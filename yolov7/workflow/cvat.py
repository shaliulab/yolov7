"""yolov7-cvat: pre-populate CVAT identity-annotation tasks with detections, as tracks

Every fly detected by the model becomes a CVAT track with a box on each frame, labelled UNASSIGNED. Detections
that clearly belong to the same fly in consecutive frames of a clip share a track, so giving one box an identity
(1, 2, ...) gives it to the whole track. Where linking is ambiguous (flies touching, a fly appearing or disappearing)
the track ends and a new one starts: give both the same identity, or use Split (Alt+M) if a track ever swaps flies.

Detection runs on the raw video frames of the experiment (the CVAT frames are resized copies with overlays drawn on
them) when they are on disk and line up with the CVAT frame, otherwise on the CVAT frames. Blank separator frames
between clips are skipped.

    yolov7-cvat predict 2366 2367                       # flyfinder3.0 (DEFAULT_MODEL)
    yolov7-cvat predict 2366 --model flyfinder --dry-run  # the production flyfinder instead
    yolov7-cvat predict --project FlyHostel3_2X_2026-10-06_13-00-00
    yolov7-cvat predict --assign --experiment FlyHostel1_3X_2026-09-22_14-00-00  # + identities (yolov7-cvat assign)

Credentials come from $CVAT_USERNAME / $CVAT_PASSWORD, the server from $CVAT_SERVER (default http://localhost:8080).
Tasks that already have annotations are skipped unless --append is passed. --assign then gives the tracks their
identities from idtrackerai and lists what a person still has to do as CVAT issues (yolov7-cvat assign --help).
"""

import argparse
import collections
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

from yolov7.workflow.common import REPO_ROOT, now

EXP_RE = re.compile(r'(FlyHostel\d+)_(\d+X)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})')
# <experiment>_<chunk>_<clip start frame>_<index in clip>.png, or <frame>_<what>.png
CLIP_RE = re.compile(r'_(\d{10})_(\d{6})\.\w+$')
FRAME_RE = re.compile(r'(?:^|_)(\d{10})(?:_|\.)')
# <frame>_s<clip start frame>.png: frame number first, as integrate-human-annotations --multisex reads it
SCENE_RE = re.compile(r'^(\d{10})_s(\d{10})\.\w+$')

# A detection continues a track from the previous frame when they are each other's nearest box, closer than MAX_STEP
# box diagonals, and every other box in either frame is clearly further away (2 x distance + MARGIN diagonals)
MAX_STEP = 1.5
MARGIN = 0.2
# Pinned rather than the production version: predictions to review can come from a model not yet in production
DEFAULT_MODEL = 'flyfinder:2026-10-15_flyfinder3.0'
BLANK_STD = 2  # separator frames are uniform
MAX_SHIFT = 0.5  # raw-frame detections must be this close (in box diagonals) to the CVAT-frame detections
# How imgstore made the non square raw frames square when it wrote the clips shown in CVAT (its SQUARE_METHOD):
# "crop" drops the end of the longest side, "pad" adds black to the bottom / right. Both keep the top left corner
FLYHOSTEL_SQUARE_METHOD = os.environ.get('FLYHOSTEL_SQUARE_METHOD', 'crop')


def connect(server):
    import logging
    from cvat_sdk import make_client
    logging.getLogger('cvat_sdk').setLevel(logging.ERROR)
    m = re.match(r'(https?://[^:/]+)(?::(\d+))?', server)
    if not m:
        raise SystemExit(f'Cannot parse CVAT server {server!r}')
    try:
        credentials = (os.environ['CVAT_USERNAME'], os.environ['CVAT_PASSWORD'])
    except KeyError:
        raise SystemExit('Set CVAT_USERNAME and CVAT_PASSWORD')
    return make_client(m.group(1), port=int(m.group(2)) if m.group(2) else None, credentials=credentials)


def iou_matrix(a, b):
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


class Detector:
    def __init__(self, weights, imgsz=640, device='0'):
        import torch
        from yolov7.models.experimental import attempt_load
        from yolov7.utils.torch_utils import select_device
        self.torch = torch
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'
        self.model = attempt_load(weights, map_location=self.device).eval()
        if self.half:
            self.model.half()
        self.stride = int(self.model.stride.max())
        self.imgsz = imgsz

    def __call__(self, frames, conf, iou=0.45):
        """BGR images of the same shape -> one (n, 6) array per image: x1 y1 x2 y2 conf class, in pixels"""
        from yolov7.utils.datasets import letterbox
        from yolov7.utils.general import non_max_suppression, scale_coords
        torch = self.torch
        ims = [letterbox(f, self.imgsz, stride=self.stride)[0][:, :, ::-1].transpose(2, 0, 1) for f in frames]
        x = torch.from_numpy(np.ascontiguousarray(np.stack(ims))).to(self.device)
        x = (x.half() if self.half else x.float()) / 255
        with torch.no_grad():
            # class-agnostic: a fly scored as both fly and blurry is one box
            pred = non_max_suppression(self.model(x)[0], conf, iou, agnostic=True)
        out = []
        for p, f in zip(pred, frames):
            p[:, :4] = scale_coords(x.shape[2:], p[:, :4], f.shape).round()
            out.append(p.cpu().numpy())
        return out


class RawVideo:
    """Frames of a FlyHostel experiment by frame number, from <videos>/<FlyHostelN>/<NX>/<date>/<chunk>.mp4"""

    def __init__(self, exp_dir):
        import yaml
        self.dir = Path(exp_dir)
        with open(self.dir / 'metadata.yaml') as f:
            self.chunksize = int(yaml.safe_load(f)['__store']['chunksize'])
        self.index = {}
        self.cap = None

    def read(self, frame_numbers):
        """Consecutive frame numbers -> list of BGR frames (None where missing)"""
        import cv2
        out = []
        pos = None
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
            if self.cap is None or self.cap[0] != chunk or pos != idx:
                if self.cap is not None:
                    self.cap[1].release()
                self.cap = (chunk, cv2.VideoCapture(str(self.dir / f'{chunk:06d}.mp4')))
                self.cap[1].set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap[1].read()
            pos = idx + 1
            out.append(ensure_square(frame) if ok else None)
        return out


def ensure_square(frame, method=FLYHOSTEL_SQUARE_METHOD):
    """The raw frame as imgstore shows it in the CVAT clips, so one scale maps it to the (square) CVAT frame"""
    h, w = frame.shape[:2]
    if h == w:
        return frame
    if method == 'pad':
        side = max(h, w)
        return np.pad(frame, [(0, side - h), (0, side - w)] + [(0, 0)] * (frame.ndim - 2))
    side = min(h, w)
    return frame[:side, :side]


def experiment_dir(task_name, videos):
    m = EXP_RE.search(task_name)
    if not m:
        return None
    d = Path(videos) / m.group(1) / m.group(2) / m.group(3)
    return d if (d / 'metadata.yaml').is_file() else None


def clips_of(names, chunksize=None, breaks=()):
    """[[(cvat frame, raw frame number)]] of consecutive raw frames, in CVAT frame order

    With chunksize, a clip that runs into the next chunk is split there: local identities are only defined within a
    chunk, so a track must never span two. A clip is also split at the CVAT frames in breaks (the first frame of each
    job): CVAT cuts a track that spans two jobs itself, and a cut track can no longer be exported.
    """
    clips, key = [], None
    for i, name in enumerate(names):
        m = CLIP_RE.search(name)
        s = SCENE_RE.match(os.path.basename(name))
        if m:
            k, fn = name[:m.start(2)], int(m.group(1)) + int(m.group(2))
        elif s:
            k, fn = s.group(2), int(s.group(1))
        else:
            m = FRAME_RE.search(name)
            k, fn = name, int(m.group(1)) if m else None
        same_chunk = chunksize is None or fn is None or fn // chunksize == (fn - 1) // chunksize
        if clips and k == key and fn is not None and clips[-1][-1][1] == fn - 1 and same_chunk and i not in breaks:
            clips[-1].append((i, fn))
        else:
            clips.append([(i, fn)])
        key = k
    return clips


def cvat_frames(task):
    """All frames of an image task as grayscale arrays, from its compressed chunks"""
    import cv2
    frames = []
    for k in range((task.size + task.data_chunk_size - 1) // task.data_chunk_size):
        _, response = task.api.retrieve_data(task.id, number=k, quality='compressed', type='chunk',
                                             _parse_response=False)
        with zipfile.ZipFile(io.BytesIO(response.data)) as z:
            for name in sorted(z.namelist()):
                frames.append(cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_GRAYSCALE))
    if len(frames) != task.size:
        raise RuntimeError(f'task {task.id}: {len(frames)} frames in chunks, {task.size} expected')
    return frames


def same_flies(raw, cvat, detector, conf):
    """Whether the raw frame shows the flies where the CVAT frame does (right frame number, same geometry)"""
    a, = detector([raw], conf)
    b, = detector([np.repeat(cvat[:, :, None], 3, 2)], conf)
    a[:, :4] *= cvat.shape[1] / raw.shape[1]
    if not len(a) or not len(b):
        return len(a) == len(b)
    ca, cb = (a[:, :2] + a[:, 2:4]) / 2, (b[:, :2] + b[:, 2:4]) / 2
    diag = np.hypot(a[:, 2] - a[:, 0], a[:, 3] - a[:, 1])
    # most flies, not all: the overlays on the CVAT frame can hide a fly or add a spurious one
    return bool((np.linalg.norm(ca[:, None] - cb[None], axis=2).min(1) / diag <= MAX_SHIFT).mean() >= 0.5)


def link(dets):
    """Detections of consecutive frames (list of (n, 4) boxes) -> tracks [[(t, i)]]

    A box continues a track only when the match is unambiguous, so tracks break wherever flies touch or a fly appears
    or disappears instead of risking a swap.
    """
    tracks, open_ = [], {}  # open_: box index in the previous frame -> its track
    for t, boxes in enumerate(dets):
        new_open = {}
        if t and len(boxes) and open_:
            prev = dets[t - 1]
            centre = lambda b: (b[:, :2] + b[:, 2:4]) / 2
            diag = np.hypot(prev[:, 2] - prev[:, 0], prev[:, 3] - prev[:, 1])
            D = np.linalg.norm(centre(prev)[:, None] - centre(boxes)[None], axis=2) / diag[:, None]
            for i in range(len(boxes)):
                j = int(D[:, i].argmin())
                d = D[j, i]
                others = np.concatenate([np.delete(D[:, i], j), np.delete(D[j], i)])
                if j in open_ and d <= MAX_STEP and (others >= 2 * d + MARGIN).all():
                    open_[j].append((t, i))
                    new_open[i] = open_[j]
        for i in range(len(boxes)):
            if i not in new_open:
                tracks.append([(t, i)])
                new_open[i] = tracks[-1]
        open_ = new_open
    return tracks


def ensure_label(client, task, name):
    from cvat_sdk.api_client import models
    labels = {l.name: l.id for l in task.get_labels()}
    if name in labels:
        return labels[name], False
    request = dict(labels=[models.PatchedLabelRequest(name=name, color='#808080', type='any')])
    if task.project_id:
        client.projects.retrieve(task.project_id).update(models.PatchedProjectWriteRequest(**request))
    else:
        task.update(models.PatchedTaskWriteRequest(**request))
    labels = {l.name: l.id for l in client.tasks.retrieve(task.id).get_labels()}
    return labels[name], True


def upload(client, task, tracks=(), shapes=()):
    """Create tracks and shapes job by job: saved through the task, CVAT re-cuts tracks at job boundaries and a track
    ending on a job's last frame comes out broken (a duplicate keyframe that stops the export, or a stray track)"""
    from cvat_sdk.api_client import models
    from cvat_sdk.core.proxies.annotations import AnnotationUpdateAction
    for job in task.get_jobs():
        inside = lambda r: job.start_frame <= r.frame <= job.stop_frame
        t, s = [r for r in tracks if inside(r)], [r for r in shapes if inside(r)]
        for k in range(0, max(len(t), len(s)), 500):
            job.update_annotations(models.PatchedLabeledDataRequest(tracks=t[k:k + 500], shapes=s[k:k + 500]),
                                   action=AnnotationUpdateAction.CREATE)


def predict_task(client, task, detector, args):
    from cvat_sdk.api_client import models
    names = [f.name for f in task.get_meta().frames]
    frames = cvat_frames(task)
    blank = [f.std() < BLANK_STD for f in frames]
    exp_dir = experiment_dir(task.name, args.videos)
    video = RawVideo(exp_dir) if exp_dir and not args.cvat_frames else None
    chunksize = RawVideo(exp_dir).chunksize if exp_dir else args.chunksize
    if chunksize is None:
        raise SystemExit(f'task {task.id}: experiment not found under {args.videos}, pass --chunksize so tracks do '
                         'not span chunks')
    stats = collections.Counter()
    boxes = {}  # cvat frame -> (n, 6) in CVAT pixels
    clips = []
    job_starts = {j.start_frame for j in task.get_jobs()}
    for clip in clips_of(names, chunksize, job_starts):
        clip = [(i, fn) for i, fn in clip if not blank[i]]
        if not clip:
            continue
        clips.append(clip)
        raw = video.read([fn for _, fn in clip]) if video and clip[0][1] is not None else [None] * len(clip)
        if raw[0] is not None and not same_flies(raw[0], frames[clip[0][0]], detector, args.conf):
            raw = [None] * len(clip)
            stats['clips where the raw frame does not match CVAT'] += 1
        for k in range(0, len(clip), args.batch):
            part = list(zip(clip[k:k + args.batch], raw[k:k + args.batch]))
            use_raw = [(c, r) for c, r in part if r is not None]
            use_cvat = [(c, frames[c[0]]) for c, r in part if r is None]
            stats['frames detected on raw video'] += len(use_raw)
            stats['frames detected on CVAT frames'] += len(use_cvat)
            for group in (use_raw, use_cvat):
                if not group:
                    continue
                ims = [im if im.ndim == 3 else np.repeat(im[:, :, None], 3, 2) for _, im in group]
                for ((i, _), im), d in zip(group, detector(ims, args.conf)):
                    h, w = frames[i].shape[:2]
                    d[:, [0, 2]] *= w / im.shape[1]
                    d[:, [1, 3]] *= h / im.shape[0]
                    boxes[i] = d
    stats['blank frames'] = int(sum(blank))

    tracks = []
    for clip in clips:
        dets = [boxes[i][:, :4] for i, _ in clip]
        for tr in link(dets):
            tracks.append([(clip[t][0], boxes[clip[t][0]][j]) for t, j in tr])
    stats['detections'] = sum(len(b) for b in boxes.values())
    stats['tracks'] = len(tracks)
    lengths = [len(t) for t in tracks]
    clip_len = [len(c) for c in clips]
    summary = dict(task=task.id, name=task.name, frames=task.size, clips=len(clips), stats=dict(stats),
                   frames_per_clip=float(np.mean(clip_len)) if clip_len else 0,
                   frames_per_track=float(np.mean(lengths)) if lengths else 0,
                   tracks_spanning_whole_clip=sum(
                       len(t) == len(c) for c in clips for t in tracks if t[0][0] == c[0][0]),
                   model=str(args.weights), conf=args.conf, label=args.label, created=now())
    if args.dry_run:
        return summary, tracks

    label_id, added = ensure_label(client, task, args.label)
    summary['label_added'] = added
    requests = []
    for tr in tracks:
        shapes = [models.TrackedShapeRequest(type=models.ShapeType('rectangle'), frame=i, points=[float(v) for v in b[:4]],
                                             outside=False, occluded=False, keyframe=True) for i, b in tr]
        last_frame, last_box = tr[-1]
        if last_frame + 1 < task.size and last_frame + 1 not in job_starts:
            shapes.append(models.TrackedShapeRequest(type=models.ShapeType('rectangle'), frame=last_frame + 1,
                                                     points=[float(v) for v in last_box[:4]], outside=True,
                                                     occluded=False, keyframe=True))
        requests.append(models.LabeledTrackRequest(frame=tr[0][0], label_id=label_id, group=0, source='auto',
                                                   shapes=shapes, attributes=[]))
    upload(client, task, tracks=requests)
    return summary, tracks


def resolve_weights(model):
    if Path(model).is_file():
        return Path(model).resolve()
    from yolov7.workflow.models import WEIGHTS, parse_ref, registry_root
    _, version_dir = parse_ref(registry_root(argparse.Namespace(registry=None)), model)
    return version_dir / WEIGHTS


def select_tasks(client, args):
    task_ids = list(args.tasks)
    for name in filter(None, [args.project, args.experiment]):
        projects = [p for p in client.projects.list() if p.name == name]
        if args.project and len(projects) != 1:
            raise SystemExit(f'{len(projects)} CVAT projects named {name!r}')
        task_ids += [t.id for p in projects for t in p.get_tasks()]
    if args.experiment:
        # tasks named after the experiment (upload_to_cvat.sh), in a project or not
        task_ids += [t.id for t in client.tasks.list() if t.name == args.experiment]
        if not task_ids:
            raise SystemExit(f'No CVAT task or project named {args.experiment!r}')
    if not task_ids:
        raise SystemExit('Pass task ids, --project or --experiment')
    return sorted(set(task_ids))


def load_detector(args):
    args.weights = resolve_weights(args.model)
    print(f'Model: {args.weights}')
    return Detector(str(args.weights), args.img_size, args.device)


def cmd_predict(args):
    from yolov7.workflow import assign as A
    client = connect(args.server)
    task_ids = select_tasks(client, args)
    detector = load_detector(args)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    for tid in task_ids:
        task = client.tasks.retrieve(tid)
        existing = task.get_annotations()
        n = len(existing.shapes) + len(existing.tracks) + len(existing.tags)
        if n and not args.append and not args.dry_run:
            print(f'task {tid} ({task.name}): already has {n} annotations, skipped (pass --append to add predictions)')
            continue
        summary, tracks = predict_task(client, task, detector, args)
        summary['existing_annotations'] = n
        out = log_dir / f'task_{tid}{"_dry-run" if args.dry_run else ""}.json'
        summary['track_boxes'] = [[(i, [round(float(v), 1) for v in b[:5]]) for i, b in tr] for tr in tracks]
        out.write_text(json.dumps(summary, indent=1, default=int))
        s = summary['stats']
        print(f"task {tid} ({task.name}): {summary['frames']} frames in {summary['clips']} clips, "
              f"{s['detections']} detections -> {s['tracks']} tracks "
              f"({summary['frames_per_track']:.1f} frames per track, {summary['frames_per_clip']:.1f} per clip)"
              f"{' [dry run, nothing uploaded]' if args.dry_run else ''}\n  {s}\n  log: {out}")
        if args.assign:
            given = {k: [(i, b[:4]) for i, b in tr] for k, tr in enumerate(tracks)} if args.dry_run else None
            A.print_summary(A.assign_task(client, client.tasks.retrieve(tid), detector, args, tracks=given))


def cmd_assign(args):
    from yolov7.workflow import assign as A
    client = connect(args.server)
    task_ids = select_tasks(client, args)
    detector = load_detector(args)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    for tid in task_ids:
        A.print_summary(A.assign_task(client, client.tasks.retrieve(tid), detector, args))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='yolov7-cvat', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--server', default=os.environ.get('CVAT_SERVER', 'http://localhost:8080'))
    sub = parser.add_subparsers(dest='command', required=True)

    def common(p):
        p.add_argument('tasks', nargs='*', type=int, help='CVAT task ids')
        p.add_argument('--project', help='all tasks of this CVAT project (named after the experiment)')
        p.add_argument('--experiment', help='all tasks named after this experiment, or in a project named after it')
        p.add_argument('--model', default=DEFAULT_MODEL,
                       help=f'registry model[:version] or a weights file (default: {DEFAULT_MODEL}; '
                            f'"flyfinder" alone means the production version)')
        p.add_argument('--label', default='UNASSIGNED',
                       help='label of the predicted tracks (added to the project if missing); flyhostel ignores boxes '
                            'still carrying it')
        p.add_argument('--dry-run', action='store_true', help='write the logs, change nothing in CVAT')
        p.add_argument('--videos', default=os.environ.get('FLYHOSTEL_VIDEOS', '/flyhostel_data/videos'))
        p.add_argument('--img-size', type=int, default=640)
        p.add_argument('--device', default='0')
        p.add_argument('--log-dir', default=str(REPO_ROOT / 'logs' / 'cvat'))
        p.add_argument('--apply', choices=['medium', 'high'], default='medium',
                       help='lowest confidence of the identity decisions applied: medium (applied, and an issue asks to '
                            'check them) or high (only the clearest are applied; everything else is an issue for a '
                            'person, with the best guess)')

    p = sub.add_parser('predict', help='upload detections to CVAT tasks as tracks labelled UNASSIGNED')
    common(p)
    p.add_argument('--assign', action='store_true', help='then give the tracks identities (see yolov7-cvat assign)')
    p.add_argument('--conf', type=float, default=0.5, help='detection confidence threshold')
    p.add_argument('--append', action='store_true', help='also predict on tasks that already have annotations')
    p.add_argument('--cvat-frames', action='store_true', help='detect on the CVAT frames, not the raw video')
    p.add_argument('--chunksize', type=int, help='frames per chunk, when the experiment is not under --videos')
    p.add_argument('--batch', type=int, default=16)
    p.set_defaults(func=cmd_predict)

    from yolov7.workflow import assign as A
    p = sub.add_parser('assign', help='give the pre-populated tracks of untouched tasks their identities',
                       description=A.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common(p)
    p.set_defaults(func=cmd_assign)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    sys.exit(main())
