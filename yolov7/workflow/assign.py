"""yolov7-cvat assign: give the pre-populated tracks of a CVAT identity task their identities, as a validator would

Runs on a task that `yolov7-cvat predict` pre-populated and nobody has touched yet (with --keep-labelled, also where a
person labelled some tracks: those are kept as they are and the others decided around them). Per clip and chunk (local
identities only exist within a chunk):
  1. every box of every track is matched to idtrackerai's blob in that frame (ROI_0 + IDENTITY of the experiment's
     .db): fragment and local identity
  2. the tracks of a clip get identities 1..n jointly: tracks present in the same frame get different identities,
     idtrackerai's identity at the start of the clip is trusted (weighted by how long its fragment is: it is reliable
     on long fragments, its errors are on the short ones after crossings), fragments already decided earlier in the
     chunk vote strongly, a track starting where another one ended votes for continuity, and a box can be called
     "not one of the flies" at a cost
  3. confidence = how much worse the best assignment that changes this track's identity is
A track gets its identity (source semi-auto) when idtrackerai has the fly missing or wrong somewhere in it and no box
of that fragment was decided before in the chunk, and the decision is clear (margin >= MEDIUM). A track whose box
touches another box (within TOUCH_TOL px) is labelled too: downstream, a labelled box gives its identity to every fly
it overlaps, so touching boxes are only unambiguous when they all carry an identity. A touching box that is not one of
the flies (often a duplicate box on one fly) becomes an issue; nothing is ever deleted automatically. A fly the predictions
missed gets a box when idtrackerai has a blob for it, or the model sees it at a lower confidence. What a person still
has to do becomes a CVAT issue on that frame (and a row of logs/cvat/task_<id>_assign.csv): check the medium-confidence
labels and every added box, decide the unclear cases (low confidence: nothing is changed), add the flies nothing was
found for (mostly take-offs and jumps, where the fly is a motion-blur streak). Boxes that are not one of the flies
(duplicates, spurious detections) stay UNASSIGNED, which flyhostel ignores; they are only listed in the report.
With --apply high only the clearest decisions (margin >= HIGH) are applied: medium ones are not, and become issues
with the best guess, as the low ones do.

Before changing anything the task's annotations are saved to logs/cvat/task_<id>_assign_<time>_backup.json.
"""

import collections
import csv
import datetime
import itertools
import json
import math
import re
import sqlite3
from pathlib import Path

import numpy as np

from yolov7.workflow import cvat as C
from yolov7.workflow.common import now

MATCH_GATE = 40  # raw px between a box centre and a blob centroid
VALIDATED_WEIGHT = 20.0
FIXED_WEIGHT = 1000.0  # a track a person labelled (--keep-labelled)
ANCHOR_WEIGHT = 5.0  # idtrackerai's identity in the first frame of a clip
LATE_WEIGHT = 0.3  # ... in later frames, per box
CONTINUITY_WEIGHT = 10.0  # a clear continuation (CLEAR_MARGIN diagonals better than the alternatives)
CLEAR_MARGIN = 0.5
CLEAR_RATIO = 2.0
MAX_STEP_DIAG = 3.0
NOT_A_FLY_PENALTY = 5.0  # per box, for calling a box something other than one of the flies
MAX_STATES = 5000
HIGH, MEDIUM = 8.0, 3.0  # margins; below MEDIUM a decision is reported, not applied
RANK = {'low': 0, 'medium': 1, 'high': 2}
FILL_CONF = 0.01  # detections considered for a fly missing from the predictions
FILL_REPORT_CONF = 0.1  # ... reported for review from this confidence
FILL_APPLY_CONF = 0.25  # ... added from this confidence
FILL_MAX_STEP = 60  # CVAT px per frame between a filled box and where the fly is before/after
TOUCH_TOL = 1.0  # CVAT px: boxes closer than this touch, and all of them need an identity
DUPLICATE_INSIDE = 0.9  # a box that is not a fly, has no blob and lies this much inside a box it touches is likely a duplicate
EXTRA_INSIDE = 0.5  # an extra box at least this much inside a box of fly m can be labelled m (yolov7-cvat complete)
SPLIT_RATIO = 1.25  # a box this many times longer than the fly's usual box holds two flies (they overlap, so not 2x)
MERGE_GAP = 8  # frames: a fly seen this close before and after a merged box is where its interpolated box says


def fragment_weight(length):
    # 1 frame -> ~0.1, 100 frames -> ~0.5, 10000 frames -> ~1
    return min(1.0, 0.1 + math.log10(max(length, 1)) / 4.5)


class Machine:
    """idtrackerai's blobs (centroid, fragment, local identity) of the frames being assigned"""

    def __init__(self, db, frame_numbers, chunksize, scale):
        import pandas as pd
        self.chunksize, self.scale = chunksize, scale
        fns = ','.join(map(str, sorted(set(frame_numbers))))
        with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as con:
            self.blobs = pd.read_sql(
                'select r.frame_number fn, r.in_frame_index ifi, r.x, r.y, r.fragment, i.local_identity lid '
                'from ROI_0 r join IDENTITY i on r.frame_number=i.frame_number and r.in_frame_index=i.in_frame_index '
                f'where r.frame_number in ({fns})', con)
            self.length = {}
            for c in sorted({fn // chunksize for fn in frame_numbers}):
                for fragment, n in con.execute('select fragment, count(*) from ROI_0 where frame_number between ? and ? '
                                               'group by fragment', (c * chunksize, (c + 1) * chunksize - 1)):
                    self.length[(c, fragment)] = n
        self.by_fn = {fn: g for fn, g in self.blobs.groupby('fn')}

    def match(self, fn, boxes):
        """boxes (n, 4) CVAT px -> list of blob rows (or None) per box"""
        from scipy.optimize import linear_sum_assignment
        g = self.by_fn.get(fn)
        out = [None] * len(boxes)
        if g is None or not len(boxes):
            return out
        c = (boxes[:, :2] + boxes[:, 2:4]) / 2 * self.scale
        D = np.hypot(c[:, None, 0] - g.x.values[None], c[:, None, 1] - g.y.values[None])
        for i, j in zip(*linear_sum_assignment(D)):
            if D[i, j] <= MATCH_GATE:
                out[i] = g.iloc[j]
        return out


def segments(tracks, fn_of, chunksize, breaks=()):
    """{track id: [(cvat frame, box)]} -> parts within one chunk and one job [dict(track=id, boxes=[(i, fn, box)])]

    breaks: the first CVAT frame of each job
    """
    out = []
    for t, tr in tracks.items():
        cur = []
        for i, b in tr:
            fn = fn_of[i]
            if cur and (fn // chunksize != cur[-1][1] // chunksize or i in breaks):
                out.append(dict(track=t, boxes=cur))
                cur = []
            cur.append((i, fn, b))
        out.append(dict(track=t, boxes=cur))
    return out


def solve_clip(segs, machine, validated, nfly, anchor=None, fixed=None):
    """Assign identities 1..nfly to the segments of one clip-chunk

    validated: {(chunk, fragment): identity} decided before this clip. fixed: {track id: identity} labelled by a person.
    Returns ([(identity, margin)] per segment, whether idtrackerai had no identity at all here, a function giving the
    margin of any set of forbidden (segment, identity index) pairs) or None if the clip is too complex. Identity 0 =
    not a fly.
    """
    chunksize = machine.chunksize
    frames = sorted({i for s in segs for i, _, _ in s['boxes']})
    # match boxes to blobs frame by frame
    for s in segs:
        s['blobs'] = [None] * len(s['boxes'])
    for i in frames:
        ks = [(k, n) for k, s in enumerate(segs) for n, b in enumerate(s['boxes']) if b[0] == i]
        fn = segs[ks[0][0]]['boxes'][ks[0][1]][1]
        boxes = np.array([segs[k]['boxes'][n][2] for k, n in ks])
        for (k, n), blob in zip(ks, machine.match(fn, boxes)):
            segs[k]['blobs'][n] = blob
    # evidence E[k, id]. Like a human, trust idtrackerai at the start of the clip (and fragments decided before,
    # anywhere); later in the clip its identities are what is being checked, so they only break ties
    E = np.zeros((len(segs), nfly))
    first = frames[0]
    for k, s in enumerate(segs):
        for (i, fn, b), blob in zip(s['boxes'], s['blobs']):
            if blob is None:
                continue
            key = (fn // chunksize, int(blob.fragment))
            if key in validated:
                E[k, validated[key] - 1] += VALIDATED_WEIGHT
            elif 0 < blob.lid <= nfly:
                w = fragment_weight(machine.length.get(key, 1))
                E[k, int(blob.lid) - 1] += ANCHOR_WEIGHT * w if i == first else LATE_WEIGHT * w
    for k, s in enumerate(segs):
        if s['track'] in (fixed or {}):
            E[k, fixed[s['track']] - 1] += FIXED_WEIGHT
    arbitrary = not E.any()
    for n, v in (anchor or {}).items():
        E[n, v] += ANCHOR_WEIGHT
    # continuity at every break: match the segments ending in a frame to those starting in the next one(s), by
    # predicted position; the better the best matching is than any other, the stronger the evidence
    link_w = collections.defaultdict(float)
    ends, starts = collections.defaultdict(list), collections.defaultdict(list)
    for k, s in enumerate(segs):
        ends[s['boxes'][-1][0]].append(k)
        starts[s['boxes'][0][0]].append(k)

    def predict(k):
        bs = segs[k]['boxes']
        c = (bs[-1][2][:2] + bs[-1][2][2:]) / 2
        if len(bs) > 1:
            c = c + (c - (bs[-2][2][:2] + bs[-2][2][2:]) / 2) * 0.5
        return c

    for t in sorted(ends):
        A = ends[t]
        B = starts.get(t + 1, [])
        if len(B) < len(A):  # a fly missed in frame t + 1
            B = B + starts.get(t + 2, [])
        if not A or not B or len(A) > 6 or len(B) > 6:
            continue
        diag = np.mean([np.hypot(*(segs[k]['boxes'][-1][2][2:] - segs[k]['boxes'][-1][2][:2])) for k in A])
        P = np.array([predict(k) for k in A])
        Q = np.array([(segs[k]['boxes'][0][2][:2] + segs[k]['boxes'][0][2][2:]) / 2 for k in B])
        D = np.linalg.norm(P[:, None] - Q[None], axis=2) / diag
        n = min(len(A), len(B))
        options = []
        for ra in itertools.permutations(range(len(A)), n):
            for rb in itertools.permutations(range(len(B)), n):
                if len(A) >= len(B) and list(rb) != sorted(rb):
                    continue
                if len(A) < len(B) and list(ra) != sorted(ra):
                    continue
                options.append((sum(D[x, y] for x, y in zip(ra, rb)), list(zip(ra, rb))))
        options.sort(key=lambda o: o[0])
        cost, pairs = options[0]
        margin = (options[1][0] - cost) if len(options) > 1 else MAX_STEP_DIAG
        # clear = better than the alternative by CLEAR_MARGIN diagonals AND by CLEAR_RATIO in relative terms (in a
        # jump both flies move far and their blurred boxes say little)
        ratio = (options[1][0] / max(cost, 0.05)) if len(options) > 1 else np.inf
        clarity = min(1.0, margin / CLEAR_MARGIN) * min(1.0, (ratio - 1) / (CLEAR_RATIO - 1))
        # one matching is one piece of evidence: a swap breaks two of its links, so each carries half
        share = 1.0 if len(pairs) == 1 else 0.5
        for x, y in pairs:
            if D[x, y] <= MAX_STEP_DIAG:
                link_w[(A[x], B[y])] += CONTINUITY_WEIGHT * clarity * share
    # Exact dynamic programme over segments in order of their first frame. The only interactions are between
    # consecutive holders of the same identity (a holder must have ended before the next one starts; continuity
    # links join a segment to the next holder), so the state is the last holder of each identity.
    order = sorted(range(len(segs)), key=lambda k: (segs[k]['boxes'][0][0], k))
    start = [s['boxes'][0][0] for s in segs]
    end = [s['boxes'][-1][0] for s in segs]

    def best(forbid=()):
        forbid = set(forbid)
        states = {tuple([-1] * nfly): (0.0, ())}
        for k in order:
            new = {}
            for holders, (sc, hist) in states.items():
                for v in range(nfly + 1):
                    if (k, v) in forbid:
                        continue
                    if v == nfly:  # not one of the flies (a duplicate or spurious box): any number of them
                        gain, nh = -NOT_A_FLY_PENALTY * len(segs[k]['boxes']), holders
                    else:
                        h = holders[v]
                        if h >= 0 and end[h] >= start[k]:
                            continue
                        gain = E[k, v] + (link_w.get((h, k), 0.0) if h >= 0 else 0.0)
                        nh = holders[:v] + (k,) + holders[v + 1:]
                    cand = (sc + gain, hist + ((k, v),))
                    if nh not in new or new[nh][0] < cand[0]:
                        new[nh] = cand
            states = dict(sorted(new.items(), key=lambda x: -x[1][0])[:MAX_STATES])
            if not states:
                return None
        sc, hist = max(states.values(), key=lambda x: x[0])
        return sc, dict(hist)

    top = best()
    if top is None:
        return None
    top_score, assign = top

    def margin(forbid):
        """How much worse the best assignment that avoids every (segment, 0-based identity) pair in forbid is"""
        alt = best(forbid)
        return float(top_score - (alt[0] if alt else -np.inf))

    out = [((assign[k] + 1) % (nfly + 1), margin({(k, assign[k])})) for k in range(len(segs))]
    return out, arbitrary, margin


def confidence(margin):
    return 'high' if margin >= HIGH else 'medium' if margin >= MEDIUM else 'low'


def applies(conf, apply):
    """Whether a decision of confidence conf is applied, when decisions are applied from confidence apply on"""
    return RANK[conf] >= RANK[apply]


def label_decision(conf, idn, frames, apply):
    """(action, needed) of labelling a track idn"""
    if not applies(conf, apply):
        return 'none', f'label {fmt(frames)} (best guess {idn})'
    return 'label track', '' if conf == 'high' else f'check the label {idn} of {fmt(frames)}'


def fmt(frames):
    frames = sorted(set(frames))
    return f'frame {frames[0]}' if len(frames) == 1 else f'frames {frames[0]}-{frames[-1]}'


class Filler:
    """Finds a fly the predictions missed in a frame: a low-confidence detection on the raw frame"""

    def __init__(self, detector, video, scale):
        self.det, self.video, self.scale = detector, video, scale

    def __call__(self, fn, existing, expected):
        f = self.video.read([fn])[0] if self.video is not None else None
        if f is None:
            return None
        d = self.det([f], FILL_CONF)[0]
        d[:, :4] /= self.scale
        if len(d) and len(existing):
            d = d[C.iou_matrix(d[:, :4], existing).max(1) < 0.2]
        if not len(d):
            return None
        c = (d[:, :2] + d[:, 2:4]) / 2
        j = int(np.linalg.norm(c - expected, axis=1).argmin())
        return d[j], float(np.linalg.norm(c[j] - expected))


def plan(tracks, names, blank, machine, nfly, filler=None, validated=None, breaks=(), apply='medium', fixed=None,
         extra_boxes=False):
    """Decisions for the tracks {track id: [(cvat frame, box)]} of a task -> rows (dicts)

    action: 'label track' (track, first, last = the per-chunk part, identity), 'add box' (first, identity, box) or
    'none'; needed: what a person still has to do ('' if nothing).
    apply: lowest confidence ('medium' or 'high') of the decisions applied; the others are left to a person.
    fixed: {track id: identity} of tracks a person labelled: the other tracks are decided around them, and they are
    never changed (where the plan cannot agree with one, it is an issue).
    extra_boxes: label an extra box on a fly (a duplicate, or a part of the fly) with that fly's identity when the clip
    is clearly worse without the fly there (yolov7-cvat complete), instead of leaving it to a person.
    """
    fixed = fixed or {}
    chunksize = machine.chunksize
    clips = C.clips_of(names, chunksize, breaks)
    fn_of = {i: fn for cl in clips for i, fn in cl}
    # frames without a raw frame number (e.g. a scene of frame -1) cannot be matched to idtrackerai: left to a person
    tracks = {t: [(i, b) for i, b in tr if fn_of[i] is not None] for t, tr in tracks.items()}
    tracks = {t: tr for t, tr in tracks.items() if tr}
    segs = segments(tracks, fn_of, chunksize, breaks)
    clip_of = {i: k for k, cl in enumerate(clips) for i, _ in cl}
    by_clip = collections.defaultdict(list)
    for s in segs:
        by_clip[clip_of[s['boxes'][0][0]]].append(s)
    parts = collections.Counter(s['track'] for s in segs)
    validated = dict(validated or {})
    rows = []
    for k, cl in enumerate(clips):
        cl = [i for i, fn in cl if fn is None and not blank[i]]
        if cl:
            rows.append(dict(clip=k, chunk=None, frames=f'{cl[0]}-{cl[-1]}', raw='', action='none', confidence='low',
                             first=cl[0], last=cl[-1], reason='no raw frame number in the image name',
                             needed=f'review {fmt(cl)}'))
    for k in sorted(by_clip):
        ss = by_clip[k]
        cl = [(i, fn) for i, fn in clips[k] if not blank[i]]
        chunk = cl[0][1] // chunksize
        where = dict(clip=k, chunk=chunk, frames=f'{cl[0][0]}-{cl[-1][0]}', raw=f'{cl[0][1]}-{cl[-1][1]}')
        # a chunk idtrackerai did not identify: name the flies present in the first frame by position, as a human would
        anchor = {}
        res = solve_clip(ss, machine, validated, nfly, fixed=fixed)
        if res is not None and res[1]:
            present = sorted([n for n, s in enumerate(ss) if s['boxes'][0][0] == cl[0][0]],
                             key=lambda n: tuple(ss[n]['boxes'][0][2][:2]))
            anchor = {n: v for v, n in enumerate(present[:nfly])}
            res = solve_clip(ss, machine, validated, nfly, anchor=anchor, fixed=fixed)
        if res is None:
            rows.append(dict(where, action='none', confidence='low', first=cl[0][0], last=cl[-1][0],
                             reason='too many tracks to solve', needed=f'review frames {cl[0][0]}-{cl[-1][0]}'))
            continue
        out, _, margin_of = res
        ident = [o[0] for o in out]
        # fragments given to more than one fly in this clip: every part needs its own label
        frag_ids = collections.defaultdict(set)
        last_holder = {}
        for s, idn in zip(ss, ident):
            if idn == 0:
                continue
            for b, blob in zip(s['boxes'], s['blobs']):
                if blob is not None:
                    key = (chunk, int(blob.fragment))
                    frag_ids[key].add(idn)
                    if key not in last_holder or b[0] >= last_holder[key][0]:
                        last_holder[key] = (b[0], idn)
        shared = {key for key, ids in frag_ids.items() if len(ids) > 1}
        row_of = {}  # segment -> its row
        def track_of(n):
            s = ss[n]
            return dict(where, track=s['track'], split=parts[s['track']] > 1, first=s['boxes'][0][0],
                        last=s['boxes'][-1][0], box=s['boxes'][0][2], margin=round(float(out[n][1]), 2))

        for n, (s, (idn, margin)) in enumerate(zip(ss, out)):
            first, last = s['boxes'][0][0], s['boxes'][-1][0]
            track = track_of(n)
            if idn == 0:
                # left UNASSIGNED, so ignored downstream: reported, no action needed
                row_of[n] = dict(track, action='none', identity=0, confidence='low',
                                 reason='more boxes than flies: this track is not one of the flies (duplicate or spurious)',
                                 needed='')
                continue
            reasons, keys = [], []
            for (i, fn, b), blob in zip(s['boxes'], s['blobs']):
                if blob is None:
                    # idtrackerai missed the fly in this frame: an UNASSIGNED box is ignored, so it needs the identity
                    reasons.append((i, 'fly missing in idtrackerai'))
                    continue
                key = (chunk, int(blob.fragment))
                keys.append(key)
                if key in shared:
                    reasons.append((i, f'fragment {key[1]} is on {len(frag_ids[key])} flies in this clip'))
                elif key in validated:
                    if validated[key] != idn:
                        reasons.append((i, f'fragment {key[1]} decided as {validated[key]} earlier in the chunk'))
                elif blob.lid != idn:
                    reasons.append((i, f'fragment {key[1]} has local identity {int(blob.lid)}'))
            if not reasons:
                continue
            conf = 'high' if n in anchor else confidence(margin)
            summary = collections.OrderedDict()
            for i, r in reasons:
                summary.setdefault(r, []).append(i)
            why = '; '.join(f'{r} ({fmt(fr)})' for r, fr in summary.items())
            if n in anchor:
                why += '; chunk not identified by idtrackerai, identity chosen'
            action, needed = label_decision(conf, idn, [first, last], apply)
            row_of[n] = dict(track, action=action, identity=idn, confidence=conf, reason=why, needed=needed)
            if applies(conf, apply):
                for key in keys:
                    if key not in shared:
                        validated.setdefault(key, idn)
        # a fragment that switched flies is decided, from here on, as the fly that holds it at the end
        for key in shared:
            validated[key] = last_holder[key][1]
        added = missing_flies(ss, ident, cl, where, chunk, machine, nfly, validated, filler, apply)
        overlapping(ss, out, row_of, added, track_of, apply, margin_of if extra_boxes else None)
        rows += [row_of[n] for n in sorted(row_of)] + added
    out = []
    for r in rows:
        if r.get('track') in fixed and r['action'] == 'label track':
            if r['identity'] == fixed[r['track']]:
                continue
            r = dict(r, action='none', needed=f"check frames {r['first']}-{r['last']}",
                     reason=f"labelled {fixed[r['track']]} by a person, the plan says {r['identity']}: {r['reason']}")
        out.append(r)
    return out


def touching(a, b, tol=TOUCH_TOL):
    """Whether two boxes overlap or touch (within tol px)"""
    return a[0] <= b[2] + tol and b[0] <= a[2] + tol and a[1] <= b[3] + tol and b[1] <= a[3] + tol


def extra_box_clusters(ss, out, by_frame, added_ids):
    """{segment: (identity m, members)} for every group of mutually touching segments that involves one fly only

    A group is the segments connected by touching boxes (in any frame); it involves one fly when the identities the
    solver gave its members, and those of the added boxes they touch, are a single m: the other members are extra boxes
    on fly m. added_ids: {frame: [(box, identity)]} of the boxes added for missing flies.
    """
    adj = collections.defaultdict(set)
    touch_added = collections.defaultdict(set)
    for i, boxes in by_frame.items():
        segs = [(n, b) for n, b, _ in boxes if n is not None]
        for (n1, b1), (n2, b2) in itertools.combinations(segs, 2):
            if touching(b1, b2):
                adj[n1].add(n2)
                adj[n2].add(n1)
        for n, b in segs:
            for ab, m in added_ids.get(i, []):
                if touching(b, ab):
                    touch_added[n].add(m)
    clusters = {}
    seen = set()
    for n in adj:
        if n in seen:
            continue
        members, stack = set(), [n]
        while stack:
            k = stack.pop()
            if k in members:
                continue
            members.add(k)
            stack.extend(adj[k])
        seen |= members
        ids = {out[k][0] for k in members if out[k][0] > 0} | {m for k in members for m in touch_added[k]}
        if len(ids) == 1:
            m = ids.pop()
            for k in members:
                clusters[k] = (m, members)
    return clusters


def overlapping(ss, out, row_of, added, track_of, apply='medium', margin_of=None):
    """Label every track whose box touches another box in some frame, even by one pixel

    Downstream, a labelled box gives its identity to every fly whose blob it overlaps, so where boxes touch every one
    of them needs its own identity (even where idtrackerai has it right), otherwise two flies can end up with one
    identity. A box that is not one of the flies and touches another is left to a person (an issue: delete it if it is a
    duplicate, otherwise label it); nothing is deleted automatically.

    track_of(n): the row fields of segment n
    margin_of(forbid): the solver's margin function (solve_clip). With it, an extra box on a fly is labelled with that
    fly's identity: it touches boxes of one fly m only, lies at least EXTRA_INSIDE inside one of them, and the clip is
    clearly worse (margin >= MEDIUM) with none of the touching boxes as fly m. Downstream, flyhostel keeps one box per
    identity and frame, so the duplicate is harmless, and a box on a part of the fly is right.
    """
    by_frame = collections.defaultdict(list)  # frame -> [(segment or None for an added box, box)]
    added_ids = collections.defaultdict(list)
    for n, s in enumerate(ss):
        for (i, _, b), blob in zip(s['boxes'], s['blobs']):
            by_frame[i].append((n, b, blob))
    for r in added:
        if r['action'] == 'add box':
            by_frame[r['first']].append((None, np.asarray(r['box'], float), None))
            added_ids[r['first']].append((np.asarray(r['box'], float), r['identity']))
    def inside(a, b):  # fraction of the smaller box inside the other
        inter = np.prod(np.clip(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0, None))
        return float(inter / max(min(np.prod(a[2:] - a[:2]), np.prod(b[2:] - b[:2])), 1e-9))

    clusters = extra_box_clusters(ss, out, by_frame, added_ids) if margin_of else {}
    cluster_margin = {}

    def extra_box(n, b, boxes, i):
        """(identity, margin) when the box b of segment n is an extra box on a fly, else None"""
        if n not in clusters:
            return None
        m, members = clusters[n]
        same = [ob for k, ob, _ in boxes if k != n and k in members and touching(b, ob)]
        same += [ab for ab, am in added_ids.get(i, []) if am == m and touching(b, ab)]
        if not same or max(inside(b, ob) for ob in same) < EXTRA_INSIDE:
            return None
        key = frozenset(members)
        if key not in cluster_margin:
            cluster_margin[key] = margin_of({(k, m - 1) for k in members})
        return m, cluster_margin[key]

    for i, boxes in sorted(by_frame.items()):
        for n, b, blob in boxes:
            if n is None or (n in row_of and (row_of[n]['action'] != 'none' or row_of[n]['needed'])):
                continue  # labelled, or already left to a person
            others = [ob for m, ob, _ in boxes if m != n and touching(b, ob)]
            if not others:
                continue
            idn, margin = out[n]
            first, last = ss[n]['boxes'][0][0], ss[n]['boxes'][-1][0]
            why = f'box touches another box (frame {i})'
            extra = extra_box(n, b, boxes, i) if idn == 0 or confidence(margin) == 'low' else None
            if extra is not None and confidence(extra[1]) != 'low':
                m, cm = extra
                conf = confidence(cm)
                action, needed = label_decision(conf, m, [first, last], apply)
                row_of[n] = dict(track_of(n), action=action, identity=m, confidence=conf, margin=round(cm, 2),
                                 reason=f'extra box on fly {m} (frame {i}): touches only boxes of fly {m}, '
                                        f'{"not one of the flies" if idn == 0 else "unclear which box is the fly"} '
                                        'otherwise; a duplicate or a part of the fly, harmless with its identity',
                                 needed=needed)
                continue
            if idn == 0:
                # never deleted automatically: a person decides
                frac = max(inside(b, ob) for ob in others)
                likely = blob is None and frac >= DUPLICATE_INSIDE
                row_of[n] = dict(track_of(n), action='none', identity=0, confidence='low',
                                 reason=why + (f'; likely a duplicate: no blob, {frac:.0%} inside the other box' if likely
                                               else '; more boxes than flies: this one fits worst'),
                                 needed=f'delete the box in {fmt([first, last])} if it is a duplicate, otherwise label it')
                continue
            conf = confidence(margin)
            action, needed = label_decision(conf, idn, [first, last], apply)
            row_of[n] = dict(track_of(n), action=action, identity=idn, confidence=conf, reason=why, needed=needed)


def missing_flies(ss, ident, cl, where, chunk, machine, nfly, validated, filler, apply='medium'):
    rows = []
    counts = collections.Counter(b[0] for s in ss for b in s['boxes'])
    scale = machine.scale
    for i, fn in cl:
        if counts.get(i, 0) >= nfly:
            continue
        here = [n for n, s in enumerate(ss) if any(b[0] == i for b in s['boxes'])]
        for m in sorted(set(range(1, nfly + 1)) - {ident[n] for n in here}):
            before = [(b[0], b[2]) for n, s in enumerate(ss) if ident[n] == m for b in s['boxes'] if b[0] < i]
            after = [(b[0], b[2]) for n, s in enumerate(ss) if ident[n] == m for b in s['boxes'] if b[0] > i]
            ref = [x for x in (max(before, key=lambda x: x[0]) if before else None,
                               min(after, key=lambda x: x[0]) if after else None) if x is not None]
            row = dict(where, first=i, last=i, identity=m)
            if not ref:
                rows.append(dict(row, action='none', confidence='low', reason=f'fly {m} not detected in frame {i}',
                                 needed=f'add box for fly {m} in frame {i}'))
                continue
            expected = np.mean([(b[:2] + b[2:]) / 2 for _, b in ref], axis=0)
            size = np.mean([b[2:] - b[:2] for _, b in ref], axis=0)
            guess = np.r_[expected - size / 2, expected + size / 2]
            gap = max(abs(j - i) for j, _ in ref)
            # does idtrackerai have this fly? a blob no prediction box was matched to, near where it should be, and not
            # inside the box of another fly (a take-off streak or a touching fly segmented as two blobs: the fly is
            # not where that blob says)
            existing = np.array([b[2] for s in ss for b in s['boxes'] if b[0] == i]).reshape(-1, 4)
            g = machine.by_fn.get(fn)
            used = {int(blob.ifi) for s in ss for b, blob in zip(s['boxes'], s['blobs']) if b[0] == i and blob is not None}
            inside_box = lambda r: any((e[0] <= r.x / scale <= e[2]) and (e[1] <= r.y / scale <= e[3]) for e in existing)
            spare = [] if g is None else [
                r for r in g.itertuples() if r.ifi not in used and not inside_box(r) and
                np.hypot(r.x / scale - expected[0], r.y / scale - expected[1]) <= FILL_MAX_STEP * gap]
            if spare:
                blob = min(spare, key=lambda r: np.hypot(r.x / scale - expected[0], r.y / scale - expected[1]))
                key = (chunk, int(blob.fragment))
                if validated.get(key, blob.lid) == m:
                    continue  # idtrackerai has the fly, with the right identity: nothing to do
                c = np.array([blob.x, blob.y]) / scale
                ok = applies('medium', apply)
                rows.append(dict(row, action='add box' if ok else 'none', box=np.r_[c - size / 2, c + size / 2],
                                 confidence='medium',
                                 reason=f'fly {m} not in the predictions in frame {i}; idtrackerai has it (fragment '
                                        f'{key[1]}) as local identity {int(blob.lid)}',
                                 needed=f'check the box added for fly {m} in frame {i}' if ok else
                                 f'add box for fly {m} in frame {i} (where idtrackerai has a blob, drawn here)'))
                continue
            merged = split_box(ss, ident, i, m, expected, size, gap)
            if merged is not None:
                box, n, ratio, how = merged
                ok = applies('medium', apply)
                rows.append(dict(row, action='add box' if ok else 'none', box=box, confidence='medium',
                                 reason=f'fly {m} not in the predictions in frame {i}; the box of fly {n} there is '
                                        f'{ratio:.2f}x the usual length of fly {n}: two flies in one box, fly {m} gets '
                                        f'{how}',
                                 needed=f'check the box added for fly {m} in frame {i} (shares a box with fly {n})' if ok
                                 else f'add box for fly {m} in frame {i} (shares a box with fly {n}, drawn here)'))
                continue
            found = filler(fn, existing, expected) if filler else None
            if found is not None and found[0][4] >= FILL_REPORT_CONF and found[1] <= FILL_MAX_STEP * gap:
                d, dist = found
                ok = d[4] >= FILL_APPLY_CONF and applies('medium', apply)
                rows.append(dict(row, action='add box' if ok else 'none', box=d[:4], confidence='medium' if ok else 'low',
                                 reason=f'fly {m} not in the predictions in frame {i}; detection at confidence '
                                        f'{d[4]:.2f}, {dist:.0f}px from where fly {m} is before/after',
                                 needed=f'check the box added for fly {m} in frame {i}' if ok else
                                 f'add box for fly {m} in frame {i} (a detection at confidence {d[4]:.2f} is drawn here)'))
            else:
                rows.append(dict(row, action='none', box=guess, confidence='low',
                                 reason=f'fly {m} not in the predictions in frame {i}, no detection found where expected '
                                        '(often a take-off or jump: a motion-blur streak)',
                                 needed=f'add box for fly {m} in frame {i}'))
    return rows


def split_box(ss, ident, i, m, expected, size, gap):
    """A box for the missing fly m in frame i, where it shares an oversized box with another fly
    -> (box, that fly, length ratio, how the box was drawn) or None

    Two flies in a line (or one on the other) are often one detection: a box SPLIT_RATIO times longer than the usual
    box of the fly it was given to, where fly m is expected. Fly m gets its box interpolated from where it is before
    and after when both are within MERGE_GAP frames (as a validator does with keyframes; the flies barely move while
    together), otherwise the half of the shared box nearest to where it is expected.
    """
    guess = np.r_[expected - size / 2, expected + size / 2]
    best = None
    for n, s in enumerate(ss):
        if ident[n] in (0, m):
            continue
        here = [b for f, _, b in s['boxes'] if f == i]
        if not here or not touching(here[0], guess):
            continue
        b = here[0]
        usual = np.median([max(ob[2:] - ob[:2]) for t, seg in enumerate(ss) if ident[t] == ident[n]
                           for _, _, ob in seg['boxes']])
        ratio = max(b[2:] - b[:2]) / max(usual, 1e-9)
        if ratio < SPLIT_RATIO:
            continue
        axis = int(np.argmax(b[2:] - b[:2]))
        mid = (b[axis] + b[axis + 2]) / 2
        a, c = b.copy(), b.copy()
        a[axis + 2], c[axis] = mid, mid
        half = min((a, c), key=lambda h: np.linalg.norm((h[:2] + h[2:]) / 2 - expected))
        box, how = (guess, 'its box interpolated from before and after') if gap <= MERGE_GAP else (half, 'the half nearest to it')
        if best is None or ratio > best[2]:
            best = (box, ident[n], float(ratio), how)
    return best


def server_tracks(ann, unassigned, identities=None):
    """({track id: [(cvat frame, box)]} of the pre-populated tracks, {track id: identity} of those a person labelled),
    or (None, the reason the task cannot be assigned)

    identities: {label id: identity} of the identity labels, to keep the tracks labelled with them (--keep-labelled);
    otherwise any labelled or edited track makes the task touched.
    """
    if ann.shapes or ann.tags:
        return None, f'{len(ann.shapes) + len(ann.tags)} shapes/tags'
    if identities is None:
        touched = [t for t in ann.tracks if t.label_id != unassigned or t.source != 'auto']
        if touched:
            return None, f'{len(touched)} tracks already labelled or edited'
    else:
        other = [t for t in ann.tracks if t.label_id != unassigned and t.label_id not in identities]
        if other:
            return None, f'{len(other)} tracks with a label that is neither {unassigned!r} nor an identity'
    fixed = {t.id: identities[t.label_id] for t in ann.tracks if identities and t.label_id in identities}
    tracks = {}
    for t in ann.tracks:
        vis = sorted([s for s in t.shapes if not s.outside], key=lambda s: s.frame)
        if vis:
            tracks[t.id] = [(s.frame, np.array(s.points[:4], float)) for s in vis]
    return tracks, fixed


def experiment(task, args):
    """(experiment dir, .db, number of flies, chunksize) of a task named after its experiment"""
    m = C.EXP_RE.search(task.name)
    exp_dir = C.experiment_dir(task.name, args.videos)
    if not m or exp_dir is None:
        return None, f'experiment of task {task.name!r} not found under {args.videos}'
    db = exp_dir / f'{m.group(0)}.db'
    if not db.is_file():
        return None, f'{db} not found'
    nfly = number_of_flies(exp_dir, m)
    if nfly is None:
        return None, f'number of flies of arena {m.group(4)} not found in {exp_dir}'
    return (exp_dir, db, nfly, C.RawVideo(exp_dir).chunksize), None


def number_of_flies(exp_dir, m):
    """Flies in the experiment: <N>X of its name, or for an arena (whose <N>X counts the flies of all arenas) the
    number of animals idtrackerai is configured with in arenas/<N>/<N>.conf (else the .toml of the arena)"""
    if not m.group(4):
        return int(m.group(2)[:-1])
    conf = exp_dir / f'{m.group(4)}.conf'
    if conf.is_file():
        return int(json.loads(conf.read_text())['_number_of_animals']['value'])
    toml = exp_dir / f'{m.group(3)}.toml'
    if toml.is_file():
        found = re.search(r'^number_of_animals\s*=\s*(\d+)', toml.read_text(), re.M)
        if found:
            return int(found.group(1))
    return None


def assign_task(client, task, detector, args, frames=None, tracks=None):
    """Plan and (unless args.dry_run) apply identities to a pre-populated task -> summary dict

    tracks: {track id: [(cvat frame, box)]} to plan on instead of the task's (a dry run of predict uploads none)
    """
    from cvat_sdk.api_client import models
    summary = dict(task=task.id, name=task.name, created=now())
    found, why = experiment(task, args)
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
    fixed = {}
    if tracks is None:
        keep = getattr(args, 'keep_labelled', False)
        identities = {labels[str(v)]: v for v in range(1, nfly + 1)} if keep else None
        tracks, why = server_tracks(ann, labels[args.label], identities)
        if tracks is None:
            return dict(summary, skipped=f'{why}: assign only runs on untouched pre-populated tasks'
                                         + ('' if keep else ' (or pass --keep-labelled)'))
        fixed = why
    elif not args.dry_run:
        raise ValueError('tracks are only given for a dry run')

    names = [f.name for f in task.get_meta().frames]
    frames = frames if frames is not None else C.cvat_frames(task)
    blank = [f.std() < C.BLANK_STD for f in frames]
    jobs = sorted((j.start_frame, j.stop_frame, j.id) for j in task.get_jobs())
    breaks = {a for a, _, _ in jobs}
    clips = C.clips_of(names, chunksize, breaks)
    video = C.RawVideo(exp_dir)
    first = next(fn for cl in clips for i, fn in cl if fn is not None and not blank[i])
    raw = video.read([first])[0]
    scale = raw.shape[1] / frames[0].shape[1] if raw is not None else None
    if scale is None:
        return dict(summary, skipped=f'raw frame {first} not found in {exp_dir}')
    machine = Machine(db, [fn for cl in clips for _, fn in cl if fn is not None], chunksize, scale)
    rows = plan(tracks, names, blank, machine, nfly, Filler(detector, video, scale), breaks=breaks,
                apply=getattr(args, 'apply', 'medium'), fixed=fixed)

    job_of = lambda i: next(j for a, b, j in jobs if a <= i <= b)
    for r in rows:
        r['job'] = job_of(r['first'])
    counts = collections.Counter((r['action'], r['confidence']) for r in rows)
    summary.update(nfly=nfly, kept=len(fixed), chunksize=chunksize, scale=round(scale, 3), db=str(db), tracks=len(tracks),
                   labelled=sum(r['action'] == 'label track' for r in rows),
                   to_check=sum(r['action'] != 'none' and bool(r['needed']) for r in rows),
                   added=sum(r['action'] == 'add box' for r in rows),
                   needed=sum(bool(r['needed']) for r in rows),
                   split=len({r['track'] for r in rows if r.get('split')}),
                   decisions={f'{a} ({c})': n for (a, c), n in sorted(counts.items())})

    log_dir = Path(args.log_dir)
    stem = log_dir / f'task_{task.id}_assign{"_dry-run" if args.dry_run else ""}'
    write_rows(rows, stem.with_suffix('.csv'))
    summary['report'] = str(stem.with_suffix('.csv'))
    if args.dry_run:
        return summary

    summary['backup'] = str(backup_annotations(ann, log_dir, task.id, 'assign'))
    summary.update(apply_rows(client, task, rows, tracks, fixed, labels, clips, chunksize, breaks, frames[0].shape[:2],
                              args.label))
    return summary


def backup_annotations(ann, log_dir, task_id, what):
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    backup = Path(log_dir) / f'task_{task_id}_{what}_{stamp}_backup.json'
    backup.write_text(json.dumps(ann.to_dict(), default=str))
    return backup


def apply_rows(client, task, rows, tracks, fixed, labels, clips, chunksize, breaks, hw, unassigned, tag='assign',
               skip_issue=lambda r: False):
    """Apply the rows of plan() to the task: labelled tracks re-created with their identity, boxes added, and a CVAT
    issue for what a person still has to do -> dict(issues=n)

    Tracks with a negative id are shapes (one-frame tracks), never changed. skip_issue(row): whether the issue of a row
    exists already (yolov7-cvat complete).
    """
    from cvat_sdk.api_client import models
    jobs = sorted((j.start_frame, j.stop_frame, j.id) for j in task.get_jobs())
    job_of = lambda i: next(j for a, b, j in jobs if a <= i <= b)
    # tracks that change: those labelled, and those crossing a chunk or a job (older predictions): one new track per
    # part
    fn_of = {i: fn for cl in clips for i, fn in cl}
    label_of = {(r['track'], r['first'], r['last']): r['identity'] for r in rows if r['action'] == 'label track'}
    parts = collections.Counter(s['track'] for s in segments(tracks, fn_of, chunksize, breaks))
    changed = ({t for t, _, _ in label_of} | {t for t, n in parts.items() if n > 1}) - set(fixed)
    changed = {t for t in changed if t >= 0}
    rect = lambda frame, box, outside=False: models.TrackedShapeRequest(
        type=models.ShapeType('rectangle'), frame=frame, points=[float(v) for v in box], outside=outside,
        occluded=False, keyframe=True)
    create = []
    for s in segments({t: tracks[t] for t in sorted(changed)}, fn_of, chunksize, breaks):
        a, b = s['boxes'][0][0], s['boxes'][-1][0]
        ident = label_of.get((s['track'], a, b))
        shapes = [rect(i, box) for i, _, box in s['boxes']]
        if b + 1 < task.size and b + 1 not in breaks:
            shapes.append(rect(b + 1, s['boxes'][-1][2], outside=True))
        create.append(models.LabeledTrackRequest(frame=a, label_id=labels[str(ident)] if ident else labels[unassigned],
                                                 group=0, source='semi-auto' if ident else 'auto', shapes=shapes,
                                                 attributes=[]))
    boxes = [models.LabeledShapeRequest(type=models.ShapeType('rectangle'), frame=int(r['first']),
                                        label_id=labels[str(r['identity'])], points=[float(v) for v in r['box']],
                                        occluded=False, outside=False, group=0, source='semi-auto', attributes=[])
             for r in rows if r['action'] == 'add box']
    if changed:
        task.remove_annotations(ids=sorted(changed))
    C.upload(client, task, tracks=create, shapes=boxes)
    # what a person still has to do: an issue on the frame, around the box concerned (the whole frame if none)
    h, w = hw
    issues = 0
    for r in rows:
        if not r['needed'] or skip_issue(r):
            continue
        box = r.get('box')
        box = [0, 0, w, h] if box is None else [float(np.clip(v, 0, (w, h)[k % 2])) for k, v in enumerate(box)]
        client.issues.create(models.IssueWriteRequest(
            frame=int(r['first']), job=int(job_of(int(r['first']))), position=box,
            message=f"[yolov7-cvat {tag}] {r['needed']}: {r['reason']}"))
        issues += 1
    return dict(issues=issues)


def write_rows(rows, path):
    cols = ['job', 'clip', 'chunk', 'frames', 'raw', 'action', 'confidence', 'track', 'split', 'first', 'last',
            'identity', 'margin', 'box', 'reason', 'needed']
    with open(path, 'w', newline='') as f:
        out = csv.DictWriter(f, cols, extrasaction='ignore')
        out.writeheader()
        for r in sorted(rows, key=lambda r: (r['first'], r.get('track') or 0)):
            r = dict(r)
            if r.get('box') is not None:
                r['box'] = ' '.join(f'{v:.0f}' for v in r['box'])
            out.writerow(r)


def print_summary(s):
    if 'skipped' in s:
        print(f"task {s['task']} ({s['name']}): assign skipped, {s['skipped']}")
        return
    done = 'would label' if 'backup' not in s else 'labelled'
    print(f"task {s['task']} ({s['name']}): {s['tracks']} tracks, {s['nfly']} flies; {done} {s['labelled']} tracks, "
          f"{'would add' if 'backup' not in s else 'added'} {s['added']} boxes, {s['split']} tracks split at chunks; "
          f"{s['needed']} items for a person ({s['to_check']} of them checks of what was changed)" +
          (f", {s['issues']} CVAT issues" if 'issues' in s else '') +
          f"\n  {s['decisions']}\n  report: {s['report']}" + (f"\n  backup: {s['backup']}" if 'backup' in s else ''))
