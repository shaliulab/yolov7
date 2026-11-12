"""yolov7-cvat complete: settle what `assign` left for a person wherever a rule can, on a task already assigned

After `yolov7-cvat predict --assign` (main.sh runs it with --apply high) a task carries the clear identities and a CVAT
issue for everything else. This re-plans the task around the labels it already has (the semi-auto ones of assign and
any a person gave: all kept as they are) and applies three more rules, each with a margin like assign's:
  1. medium-confidence identities are applied, with an issue asking to check them. In the validation of
     FlyHostel4_12X_2026-11-06_18-00-00 job 7699 (2026-11-10) the person agreed with all 12 medium guesses and
     overruled 2 of the 5 low ones, which stay with a person.
  2. an extra box on a fly (a duplicate, or a part of a fly cut in two) is labelled with that fly's identity instead
     of being left for deletion: it touches only boxes of one fly, lies at least half inside one of them, and the
     clip's assignment is clearly worse without that fly there. flyhostel keeps one box per identity and frame.
  3. a fly missing from the predictions because it shares one oversized box with another fly (flies in a line, one
     on the other) gets the half of that box on its side, with an issue asking to check it.
Issues of assign that these rules settle are resolved, with a comment saying what was done; the others stay open,
and nothing is added twice. Low-confidence decisions are still left to a person. Nothing is deleted.
The report is logs/cvat/task_<id>_complete.csv and the annotations are backed up first, as with assign.

    yolov7-cvat complete 2547 --job 7700                      # one job of a task (or several: --job 7701 7702)
    yolov7-cvat complete --experiment FlyHostel4_12X_2026-11-06_18-00-00 --dry-run
    yolov7-cvat complete 2547 --from-backup logs/cvat/task_2547_assign_20261110-141218_backup.json --dry-run
"""

import collections
import json
import re
from pathlib import Path

import numpy as np

from yolov7.workflow import assign as A
from yolov7.workflow import cvat as C
from yolov7.workflow.common import now

TAG = 'complete'
ISSUE_RE = re.compile(r'^\[yolov7-cvat (\w+)\] (.*?): ')
FRAMES_RE = re.compile(r'frames? (\d+)(?:-(\d+))?')
FLY_RE = re.compile(r'(?:fly|best guess) (\d+)')


def issue_key(message):
    """(kind, first, last, fly) of an issue of ours, or None: what it asks about"""
    m = ISSUE_RE.match(message)
    if not m:
        return None
    needed = m.group(2)
    f = FRAMES_RE.search(needed)
    if not f:
        return None
    first, last = int(f.group(1)), int(f.group(2) or f.group(1))
    fly = FLY_RE.search(needed)
    kind = needed.split(' ')[0]  # label, delete, add, check, review
    return kind, first, last, int(fly.group(1)) if fly else None


def row_key(r):
    """The key of the issue a row would raise (or did raise): same shape as issue_key"""
    if r['action'] == 'add box' or (r.get('track') is None and r.get('identity')):
        return 'add', int(r['first']), int(r['first']), int(r['identity'])
    if r['action'] == 'label track':
        kinds = ('label', 'delete')
    elif r.get('identity') == 0:
        kinds = ('delete',)
    else:
        kinds = ('label', 'check', 'review')
    return kinds, int(r['first']), int(r['last']), r.get('identity')


def settles(r, key):
    """Whether the row r (applied, or still needed) is about the issue key"""
    kinds, first, last, fly = row_key(r)
    kind, ifirst, ilast, ifly = key
    if kind == 'add':
        return kinds == 'add' and (first, fly) == (ifirst, ifly)
    if kinds == 'add':
        return False
    return kind in kinds and (first, last) == (ifirst, ilast)


def open_issues(client, task, jobs=None):
    """[(issue, first message)] of the open issues of ours in the task (or some of its jobs)"""
    out = []
    for j in task.get_jobs():
        if jobs and j.id not in jobs:
            continue
        page = 1
        while True:
            data, _ = client.api_client.issues_api.list(job_id=j.id, page=page, page_size=100)
            for issue in data.results:
                if issue.resolved:
                    continue
                # the oldest comment is the issue's message (the list comes newest first)
                comments, _ = client.api_client.comments_api.list(issue_id=issue.id, page_size=50)
                message = min(comments.results, key=lambda c: c.id).message if comments.results else ''
                if message.startswith('[yolov7-cvat'):
                    out.append((issue, message))
            if not data.next:
                break
            page += 1
    return out


def comment(client, issue, message):
    from cvat_sdk.api_client import models
    client.api_client.comments_api.create(models.CommentWriteRequest(issue=issue.id, message=message))


def resolve(client, issue, message):
    from cvat_sdk.api_client import models
    comment(client, issue, message)
    client.api_client.issues_api.partial_update(issue.id, patched_issue_write_request=models.PatchedIssueWriteRequest(
        resolved=True))


def task_tracks(ann, labels, nfly, unassigned):
    """({track id: [(cvat frame, box)]}, {track id: identity} of the labelled ones) of a task's annotations

    Labelled shapes (boxes added by assign or a person) are one-frame tracks with a negative id, never changed. Tracks
    and shapes with a label that is neither an identity nor unassigned (FMB, COURTSHIP, ...) are not boxes of flies.
    """
    identities = {labels[str(v)]: v for v in range(1, nfly + 1)}
    tracks, fixed = {}, {}
    for t in ann.tracks:
        if t.label_id != unassigned and t.label_id not in identities:
            continue
        vis = sorted([s for s in t.shapes if not s.outside], key=lambda s: s.frame)
        if not vis:
            continue
        tracks[t.id] = [(s.frame, np.array(s.points[:4], float)) for s in vis]
        if t.label_id in identities:
            fixed[t.id] = identities[t.label_id]
    for s in ann.shapes:
        if s.label_id not in identities or len(s.points) < 4:
            continue
        pts = np.array(s.points, float).reshape(-1, 2)
        box = np.r_[pts.min(0), pts.max(0)] if len(pts) > 2 else np.array(s.points[:4], float)
        tracks[-s.id] = [(s.frame, box)]
        fixed[-s.id] = identities[s.label_id]
    return tracks, fixed


def backup_tracks(path, labels, nfly, unassigned):
    """Tracks of a saved annotations backup (task_<id>_*_backup.json), for a dry run on that state"""
    class Obj:
        def __init__(self, d):
            self.__dict__.update(d)
    d = json.loads(Path(path).read_text())
    ann = Obj(dict(tracks=[Obj(dict(t, shapes=[Obj(s) for s in t['shapes']])) for t in d['tracks']],
                   shapes=[Obj(s) for s in d['shapes']]))
    return task_tracks(ann, labels, nfly, unassigned)


def complete_task(client, task, detector, args):
    """Plan and (unless args.dry_run) apply the completion rules to a task -> summary dict"""
    summary = dict(task=task.id, name=task.name, created=now())
    found, why = A.experiment(task, args)
    if found is None:
        return dict(summary, skipped=why)
    exp_dir, db, nfly, chunksize = found
    if nfly < 2:
        return dict(summary, skipped='single-fly experiment: nothing to assign')
    labels = {l.name: l.id for l in task.get_labels()}
    missing = [str(v) for v in range(1, nfly + 1) if str(v) not in labels]
    if args.label not in labels or missing:
        return dict(summary, skipped=f'labels {missing or [args.label]} not in the task')
    ann = task.get_annotations()
    if args.from_backup:
        if not args.dry_run:
            raise SystemExit('--from-backup only makes sense with --dry-run')
        tracks, fixed = backup_tracks(args.from_backup, labels, nfly, labels[args.label])
    else:
        tracks, fixed = task_tracks(ann, labels, nfly, labels[args.label])
    if not tracks:
        return dict(summary, skipped='no tracks: run yolov7-cvat predict --assign first')

    names = [f.name for f in task.get_meta().frames]
    frames = C.cvat_frames(task)
    blank = [f.std() < C.BLANK_STD for f in frames]
    jobs = sorted((j.start_frame, j.stop_frame, j.id) for j in task.get_jobs())
    breaks = {a for a, _, _ in jobs}
    clips = C.clips_of(names, chunksize, breaks)
    if args.job:
        unknown = set(args.job) - {j for _, _, j in jobs}
        if unknown:
            return dict(summary, skipped=f'jobs {sorted(unknown)} are not jobs of task {task.id}')
        # clips never cross a job break, so planning only the tracks of these jobs decides them the same way; the
        # idtrackerai evidence (Machine) still comes from every clip: a fragment that starts in the previous job keeps
        # the identity its earlier frames give it
        mine = lambda i: any(a <= i <= b for a, b, j in jobs if j in args.job)
        tracks = {t: [(i, b) for i, b in tr if mine(i)] for t, tr in tracks.items()}
        tracks = {t: tr for t, tr in tracks.items() if tr}
    video = C.RawVideo(exp_dir)
    first = next(fn for cl in clips for i, fn in cl if fn is not None and not blank[i])
    raw = video.read([first])[0]
    if raw is None:
        return dict(summary, skipped=f'raw frame {first} not found in {exp_dir}')
    scale = raw.shape[1] / frames[0].shape[1]
    machine = A.Machine(db, [fn for cl in clips for _, fn in cl if fn is not None], chunksize, scale)
    state = A.CarryState(args.log_dir, C.EXP_RE.search(task.name).group(0) + ('_dry-run' if args.dry_run else ''))
    carry = state.load()
    rows = A.plan(tracks, names, blank, machine, nfly, A.Filler(detector, video, scale), breaks=breaks,
                  apply='medium', fixed=fixed, extra_boxes=True, carry=carry)
    if not args.job:
        state.save(carry)
    job_of = lambda i: next(j for a, b, j in jobs if a <= i <= b)
    for r in rows:
        r['job'] = job_of(r['first'])
    if args.job:
        rows = [r for r in rows if r['job'] in args.job]
    # rows that leave nothing to a person on a track that was not touched are only informative (assign reports them
    # too); the applied ones and those needing a person are what matters here
    issues = open_issues(client, task, args.job) if not args.from_backup else []
    settled, kept, new = [], [], []
    applied = [r for r in rows if r['action'] != 'none']
    needed = [r for r in rows if r['needed']]
    for issue, message in issues:
        key = issue_key(message)
        done = [r for r in applied if key and settles(r, key)]
        still = [r for r in needed if key and settles(r, key)]
        if done and not [r for r in still if r['action'] == 'none']:
            # the row about the box the issue points at (its position), then one that settles it for good
            pos = np.array(issue.position, float) if issue.position and len(issue.position) == 4 else None
            def closeness(r):
                b = r.get('box')
                return np.abs(np.asarray(b, float) - pos).sum() if pos is not None and b is not None else np.inf
            settled.append((issue, message, min(done, key=lambda r: (closeness(r), bool(r['needed'])))))
        else:
            kept.append((issue, message))
    # an item settled but still to be checked keeps its issue, with a comment: one issue per item
    kept_keys = [issue_key(m) for _, m in kept] + [issue_key(m) for _, m, r in settled if r['needed']]
    for r in needed:
        if not any(k and settles(r, k) for k in kept_keys):
            new.append(r)
    new_ids = {id(r) for r in new}

    counts = collections.Counter((r['action'], r['confidence']) for r in rows)
    summary.update(nfly=nfly, kept=len(fixed), chunksize=chunksize, scale=round(scale, 3), db=str(db),
                   tracks=len(tracks), job=args.job,
                   labelled=sum(r['action'] == 'label track' for r in rows),
                   added=sum(r['action'] == 'add box' for r in rows),
                   to_check=sum(r['action'] != 'none' and bool(r['needed']) for r in rows),
                   needed=len(needed), issues_open=len(issues), issues_settled=sum(not r['needed'] for _, _, r in settled),
                   issues_commented=sum(bool(r['needed']) for _, _, r in settled), issues_new=len(new),
                   decisions={f'{a} ({c})': n for (a, c), n in sorted(counts.items())})
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = f'_job{"-".join(map(str, args.job))}' if args.job else ''
    stem = log_dir / f'task_{task.id}_complete{suffix}{"_dry-run" if args.dry_run else ""}'
    A.write_rows(rows, stem.with_suffix('.csv'))
    summary['report'] = str(stem.with_suffix('.csv'))
    if args.dry_run:
        return summary

    summary['backup'] = str(A.backup_annotations(ann, log_dir, task.id, TAG))
    summary.update(A.apply_rows(client, task, rows, tracks, fixed, labels, clips, chunksize, breaks,
                                frames[0].shape[:2], args.label, tag=TAG, skip_issue=lambda r: id(r) not in new_ids))
    resolved = 0
    for issue, _, r in settled:
        what = (f"labelled {r['identity']} ({r['confidence']} confidence, margin {r.get('margin', '')})"
                if r['action'] == 'label track' else f"added a box for fly {r['identity']}")
        message = f"[yolov7-cvat {TAG}] {what}: {r['reason']}"
        if r['needed']:
            comment(client, issue, f"{message}. Still to do: {r['needed']}")
        else:
            resolve(client, issue, message)
            resolved += 1
    summary['issues_resolved'] = resolved
    return summary


def print_summary(s):
    if 'skipped' in s:
        print(f"task {s['task']} ({s['name']}): complete skipped, {s['skipped']}")
        return
    dry = 'backup' not in s
    where = f"job(s) {', '.join(map(str, s['job']))} of task {s['task']}" if s['job'] else f"task {s['task']}"
    print(f"{where} ({s['name']}): {s['tracks']} tracks ({s['kept']} already labelled), {s['nfly']} flies; "
          f"{'would label' if dry else 'labelled'} {s['labelled']} tracks, {'would add' if dry else 'added'} "
          f"{s['added']} boxes; {s['needed']} items for a person ({s['to_check']} of them checks of what was changed); "
          f"issues: {s['issues_open']} open, {s['issues_settled']} {'would be ' if dry else ''}resolved, "
          f"{s['issues_commented']} {'would be ' if dry else ''}commented (to check), {s['issues_new']} new"
          f"\n  {s['decisions']}\n  report: {s['report']}"
          + (f"\n  backup: {s['backup']}" if not dry else ''))
