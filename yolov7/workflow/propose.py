"""yolov7-cvat propose: the identity proposals of `predict --assign` + `complete`, for idtrackerai-validator's Annotate tab

Instead of CVAT tasks, the frames are the failed scenes of the validation pipeline (<experiment>/flyhostel/validation/
scene_qc.csv, made by make-identogram + auto-annotate-qc; scenes of up to MAX_SCENE_LENGTH frames, as export-images),
read from the raw video, in chronological order. Per scene (split at chunks) the flies are detected (flyfinder), linked
into tracks (cvat.link) and given identities by assign.plan with complete's rules (apply medium, extra boxes), carrying
what each clip found about idtrackerai's local identities into the next clips of the chunk. Frames a person already saved
in the Annotate tab are kept: their labels fix the tracks through them (as labels a person gave in CVAT).

The proposals are written to the tab's database (<experiment>/flyhostel/validation/validator_annotations.db):
    proposal_frames (frame_number, flags)  frames proposed, with what a person has to do there (JSON list of texts,
                                           the CVAT issues of assign)
    proposals       (frame_number, x0, y0, x1, y1, label, note, confidence)   raw frame px; label "" = not assigned
    proposal_runs   (t, model, scenes, frames, summary)
Frames of the scenes proposed again replace their earlier proposals; saved boxes (boxes / frames) are never changed.

    yolov7-cvat propose --experiment FlyHostel3_12X_2026-11-10_17-00-00
    yolov7-cvat propose --experiment FlyHostel3_12X_2026-11-10_17-00-00 --limit 20 --dry-run

Prints "@@PROGRESS <fraction>" lines for the validator's job.
"""

import argparse
import collections
import csv
import json
import os
import re
import sqlite3
import time
from pathlib import Path

import numpy as np

from yolov7.workflow import assign as A
from yolov7.workflow import cvat as C
from yolov7.workflow.common import REPO_ROOT

DB_NAME = 'validator_annotations.db'
MAX_SCENE_LENGTH = 5000
FIXED_IOU = 0.5  # a track box this close to a box a person saved takes its label

SCHEMA = """
    CREATE TABLE IF NOT EXISTS boxes (frame_number INTEGER, x0 REAL, y0 REAL, x1 REAL, y1 REAL, label TEXT);
    CREATE INDEX IF NOT EXISTS boxes_fn ON boxes (frame_number);
    CREATE TABLE IF NOT EXISTS frames (frame_number INTEGER PRIMARY KEY, author TEXT, updated REAL, n_boxes INTEGER);
    CREATE TABLE IF NOT EXISTS log (t REAL, author TEXT, frame_number INTEGER, boxes TEXT);
    CREATE TABLE IF NOT EXISTS proposal_frames (frame_number INTEGER PRIMARY KEY, flags TEXT);
    CREATE TABLE IF NOT EXISTS proposals (frame_number INTEGER, x0 REAL, y0 REAL, x1 REAL, y1 REAL, label TEXT,
                                          note TEXT, confidence TEXT);
    CREATE INDEX IF NOT EXISTS proposals_fn ON proposals (frame_number);
    CREATE TABLE IF NOT EXISTS proposal_runs (t REAL, model TEXT, scenes INTEGER, frames INTEGER, summary TEXT);
"""


FRAMES_RE = re.compile(r'\b(frames?) (\d+)(?:-(\d+))?')
FORWARD = 600  # frames: a frame this close ahead in the same chunk is reached by decoding forward, not by seeking


class SequentialVideo(C.RawVideo):
    """RawVideo that decodes forward to a frame a little ahead instead of seeking (a seek decodes from the last
    keyframe, ~4 s on these videos), as the scenes are read in order"""

    def read(self, frame_numbers):
        import cv2
        out = []
        for fn in frame_numbers:
            chunk = fn // self.chunksize
            if chunk not in self.index:
                npz = self.dir / f'{chunk:06d}.npz'
                self.index[chunk] = np.load(npz)['frame_number'] if npz.is_file() else np.array([], int)
            idx = np.flatnonzero(self.index[chunk] == fn)
            if not len(idx):
                out.append(None)
                continue
            idx = int(idx[0])
            pos = getattr(self, 'pos', None)
            if self.cap is None or self.cap[0] != chunk or pos is None or not 0 <= idx - pos <= FORWARD:
                if self.cap is not None:
                    self.cap[1].release()
                self.cap = (chunk, cv2.VideoCapture(str(self.dir / f'{chunk:06d}.mp4')))
                self.cap[1].set(cv2.CAP_PROP_POS_FRAMES, idx)
                pos = idx
            while pos < idx:
                self.cap[1].grab()
                pos += 1
            ok, frame = self.cap[1].read()
            self.pos = pos + 1
            out.append(C.ensure_square(frame) if ok else None)
        return out


def raw_frames(text, fn_of):
    """The texts of assign speak of CVAT frames (here: positions in the list of scene frames): raw frame numbers"""
    def sub(m):
        a = fn_of.get(int(m.group(2)), m.group(2))
        if m.group(3):
            return f'{m.group(1)} {a}-{fn_of.get(int(m.group(3)), m.group(3))}'
        return f'{m.group(1)} {a}'
    return FRAMES_RE.sub(sub, text) if text else text


class CachedFiller:
    """assign.Filler on the detections made once at FILL_CONF (no GPU needed: works in the planning processes)"""

    def __init__(self, cache):
        self.cache = cache  # raw frame -> (n, 5) detections at FILL_CONF

    def __call__(self, fn, existing, expected):
        d = self.cache.get(fn)
        if d is None or not len(d):
            return None
        d = d.copy()
        if len(existing):
            d = d[C.iou_matrix(d[:, :4], existing).max(1) < 0.2]
        if not len(d):
            return None
        c = (d[:, :2] + d[:, 2:4]) / 2
        j = int(np.linalg.norm(c - expected, axis=1).argmin())
        return d[j], float(np.linalg.norm(c[j] - expected))


G = {}  # what the planning processes read (inherited by fork)


def plan_chunk(chunk):
    """assign.plan on the tracks of one chunk: chunks are independent (local identities, decided fragments and the
    carry-over all live within a chunk) -> (chunk, rows, assignment, carry)"""
    tracks = {t: tr for t, tr in G['tracks'].items() if G['fn_of'][tr[0][0]] // G['chunksize'] == chunk}
    carry, assignment = {}, {}
    rows = A.plan(tracks, G['names'], G['blank'], G['machine'], G['nfly'], CachedFiller(G['cache']), apply='medium',
                  fixed={t: v for t, v in G['fixed'].items() if t in tracks}, extra_boxes=True, carry=carry,
                  assignment=assignment)
    return chunk, rows, assignment, carry


def progress(fraction):
    print(f'@@PROGRESS {fraction:.4f}', flush=True)


def read_scenes(path, limit=None):
    scenes = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('pass') == 'True':
                continue
            length = int(float(row['length']))
            if length > MAX_SCENE_LENGTH:
                continue
            scenes.append((int(float(row['scene_start'])), length))
    scenes.sort()
    # overlapping scenes: one frame belongs to one clip
    out, end = [], -1
    for start, length in scenes:
        start2 = max(start, end + 1)
        if start2 < start + length:
            out.append((start2, start + length - start2))
            end = start + length - 1
    return out[:limit] if limit else out


def saved_frames(db):
    """{frame: [(box, label)]} of the frames a person saved"""
    if not db.is_file():
        return {}
    with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as con:
        try:
            rows = con.execute('SELECT b.frame_number, b.x0, b.y0, b.x1, b.y1, b.label FROM boxes b '
                               'JOIN frames f ON f.frame_number = b.frame_number').fetchall()
        except sqlite3.Error:
            return {}
    out = collections.defaultdict(list)
    for fn, x0, y0, x1, y1, label in rows:
        out[fn].append((np.array([x0, y0, x1, y1], float), label or ''))
    return out


def fixed_from_saved(tracks, fn_of, saved):
    """{track: identity} for the tracks whose box matches a numbered box a person saved (one identity per track)"""
    found = collections.defaultdict(set)
    for t, tr in tracks.items():
        for i, b in tr:
            boxes = saved.get(fn_of[i])
            if not boxes:
                continue
            numbered = [(sb, lab) for sb, lab in boxes if lab.isdigit()]
            if not numbered:
                continue
            iou = C.iou_matrix(b[None], np.array([sb for sb, _ in numbered]))[0]
            j = int(iou.argmax())
            if iou[j] >= FIXED_IOU:
                found[t].add(int(numbered[j][1]))
    return {t: ids.pop() for t, ids in found.items() if len(ids) == 1}


def build(tracks, names, fn_of, chunksize, rows, assignment):
    """-> ({frame: [box dicts]}, {frame: [flags]})"""
    boxes = collections.defaultdict(list)
    flags = collections.defaultdict(list)
    by_seg = {}
    rows = [dict(r, reason=r.get('reason', ''), needed=r.get('needed', '')) for r in rows]
    for r in rows:
        if r.get('track') is not None and r['action'] != 'add box':
            by_seg[(r['track'], r['first'])] = r
    for s in A.segments(tracks, fn_of, chunksize):
        key = (s['track'], s['boxes'][0][0])
        idn, margin, conf = assignment.get(key, (0, 0.0, 'low'))
        r = by_seg.get(key)
        label, note = (str(idn) if idn else ''), ''
        if r is not None:
            if r['action'] == 'label track':
                label = str(r['identity'])
                note = f"{r['needed']}: {r['reason']}" if r['needed'] else r['reason']
                conf = r['confidence']
            elif r['needed']:
                label = ''
                note = f"{r['needed']}: {r['reason']}"
                conf = r['confidence']
            else:
                label = ''
                note = r['reason']
        elif idn == 0:
            note = 'not one of the flies (duplicate or spurious box)'
        for i, _, b in s['boxes']:
            boxes[fn_of[i]].append(dict(box=[float(v) for v in b[:4]], label=label, note=note, confidence=conf))
    for r in rows:
        fn = fn_of.get(r['first'])
        if fn is None:
            continue
        if r['action'] == 'add box' or (r.get('track') is None and r.get('box') is not None):
            boxes[fn].append(dict(box=[float(v) for v in r['box']], label=str(r['identity']) if r['action'] == 'add box' else '',
                                  note=f"{r['needed']}: {r['reason']}" if r['needed'] else r['reason'],
                                  confidence=r['confidence']))
        if r['needed']:
            flags[fn].append(f"{r['needed']}: {r['reason']}")
    for bs in boxes.values():
        for b in bs:
            b['note'] = raw_frames(b['note'], fn_of)
    flags = {fn: [raw_frames(t, fn_of) for t in ts] for fn, ts in flags.items()}
    # one box per local identity and frame: a duplicate label loses it (flagged)
    for fn, bs in boxes.items():
        seen = {}
        for b in bs:
            if b['label'].isdigit():
                if b['label'] in seen:
                    flags.setdefault(fn, []).append(f"two boxes proposed for fly {b['label']}: one left unassigned")
                    b['note'] = (b['note'] + '; ' if b['note'] else '') + f"also proposed as fly {b['label']}"
                    b['label'] = ''
                else:
                    seen[b['label']] = b
    return boxes, flags


def main(argv=None):
    p = argparse.ArgumentParser(prog='yolov7-cvat propose', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiment', required=True, help='FlyHostel<N>_<X>X_<date>[_arena<N>]')
    p.add_argument('--videos', default=os.environ.get('FLYHOSTEL_VIDEOS', '/flyhostel_data/videos'))
    p.add_argument('--db', help=f'default: <experiment>/flyhostel/validation/{DB_NAME}')
    p.add_argument('--scenes', help='default: <experiment>/flyhostel/validation/scene_qc.csv')
    p.add_argument('--model', default=C.DEFAULT_MODEL)
    p.add_argument('--conf', type=float, default=0.5)
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--device', default='0')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--workers', type=int, default=8, help='processes planning chunks in parallel')
    p.add_argument('--limit', type=int, help='only the first N scenes')
    p.add_argument('--dry-run', action='store_true', help='print the summary, write nothing')
    p.add_argument('--log-dir', default=str(REPO_ROOT / 'logs' / 'cvat'))
    args = p.parse_args(argv)

    exp_dir = C.experiment_dir(args.experiment, args.videos)
    m = C.EXP_RE.search(args.experiment)
    if exp_dir is None or not m:
        raise SystemExit(f'{args.experiment} not found under {args.videos}')
    db = exp_dir / f'{m.group(0)}.db'
    nfly = A.number_of_flies(exp_dir, m)
    validation = exp_dir / 'flyhostel' / 'validation'
    out_db = Path(args.db) if args.db else validation / DB_NAME
    scenes = read_scenes(args.scenes or validation / 'scene_qc.csv', args.limit)
    if not scenes:
        raise SystemExit('no scenes to propose')
    video = SequentialVideo(exp_dir)
    chunksize = video.chunksize
    names = [f'{fn:010d}_s{start:010d}.png' for start, length in scenes for fn in range(start, start + length)]
    total = len(names)
    print(f'{args.experiment}: {nfly} flies, {len(scenes)} scenes, {total} frames', flush=True)

    args.weights = C.resolve_weights(args.model)
    detector = C.Detector(str(args.weights), args.img_size, args.device)
    clips = C.clips_of(names, chunksize)
    fn_of = {i: fn for cl in clips for i, fn in cl}
    tracks, done, cache = {}, 0, {}
    progress(0)
    for clip in clips:
        frames = video.read([fn for _, fn in clip])
        dets = []
        for k in range(0, len(clip), args.batch):
            part = frames[k:k + args.batch]
            ok = [f for f in part if f is not None]
            # one pass at the low threshold of the filler: after NMS, the boxes above args.conf are those a pass at
            # args.conf gives (a lower box never suppresses a higher one)
            found = iter(detector(ok, min(A.FILL_CONF, args.conf))) if ok else iter(())
            for (_, fn), f in zip(clip[k:k + args.batch], part):
                d = next(found) if f is not None else np.zeros((0, 6))
                cache[fn] = d[:, :5]
                dets.append(d[d[:, 4] >= args.conf][:, :4])
        for tr in C.link(dets):
            tracks[len(tracks)] = [(clip[t][0], dets[t][j]) for t, j in tr]
        done += len(clip)
        progress(0.7 * done / total)

    saved = saved_frames(out_db)
    fixed = fixed_from_saved(tracks, fn_of, saved)
    machine = A.Machine(db, [fn for cl in clips for _, fn in cl], chunksize, 1.0)
    G.update(tracks=tracks, names=names, blank=[False] * total, machine=machine, nfly=nfly, fixed=fixed, cache=cache,
             fn_of=fn_of, chunksize=chunksize)
    chunks = sorted({fn_of[tr[0][0]] // chunksize for tr in tracks.values()})
    carry, assignment, rows = {}, {}, []
    import multiprocessing
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(max(1, min(args.workers, len(chunks)))) as pool:
        for n, (chunk, r, a, c) in enumerate(pool.imap_unordered(plan_chunk, chunks)):
            rows += r
            assignment.update(a)
            carry.update(c)
            progress(0.7 + 0.25 * (n + 1) / len(chunks))
    boxes, flags = build(tracks, names, fn_of, chunksize, rows, assignment)
    counts = collections.Counter((r['action'], r['confidence']) for r in rows)
    carried = sum(bool(mp) for lst in carry.values() for _, mp in lst)
    summary = dict(experiment=args.experiment, flies=nfly, scenes=len(scenes), frames=total, tracks=len(tracks),
                   kept_from_saved=len(fixed), flagged_frames=len(flags),
                   clip_ends_with_swaps_carried=carried,
                   decisions={f'{a} ({c})': n for (a, c), n in sorted(counts.items())})
    print(json.dumps(summary), flush=True)
    if args.dry_run:
        return summary

    out_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(out_db, timeout=60) as con:
        con.executescript(SCHEMA)
        con.execute('CREATE TEMP TABLE fns (fn INTEGER PRIMARY KEY)')
        con.executemany('INSERT INTO fns VALUES (?)', [(fn,) for fn in fn_of.values()])
        con.execute('DELETE FROM proposals WHERE frame_number IN (SELECT fn FROM fns)')
        con.execute('DELETE FROM proposal_frames WHERE frame_number IN (SELECT fn FROM fns)')
        con.executemany('INSERT INTO proposal_frames VALUES (?, ?)',
                        [(fn, json.dumps(flags.get(fn, []))) for fn in fn_of.values()])
        con.executemany('INSERT INTO proposals VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                        [(fn, *b['box'], b['label'], b['note'], b['confidence'])
                         for fn, bs in boxes.items() for b in bs])
        con.execute('INSERT INTO proposal_runs VALUES (?, ?, ?, ?, ?)',
                    (time.time(), str(args.model), len(scenes), total, json.dumps(summary)))
    progress(1)
    print(f'written to {out_db}', flush=True)
    return summary


if __name__ == '__main__':
    main()
