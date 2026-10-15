# TrackSplitter.py

import bisect

import numpy as np


def _runs_from_votes(values, window, min_run):
    """
    Smooth a 0/1 vote sequence with a majority window, then merge any run
    shorter than min_run into its longer neighbor. Returns [[label, i0, i1]].
    """
    n = len(values)
    half = window // 2
    smooth = []
    for i in range(n):
        seg = values[max(0, i - half): i + half + 1]
        twice = 2 * sum(seg)
        smooth.append(values[i] if twice == len(seg) else int(twice > len(seg)))

    runs = []
    for i, s in enumerate(smooth):
        if runs and runs[-1][0] == s:
            runs[-1][2] = i
        else:
            runs.append([s, i, i])

    while len(runs) > 1:
        lengths = [r[2] - r[1] + 1 for r in runs]
        k = int(np.argmin(lengths))
        if lengths[k] >= min_run:
            break
        if k == 0:
            nb = 1
        elif k == len(runs) - 1:
            nb = k - 1
        else:
            nb = k - 1 if lengths[k - 1] >= lengths[k + 1] else k + 1
        runs[k][0] = runs[nb][0]
        merged = []
        for r in runs:
            if merged and merged[-1][0] == r[0]:
                merged[-1][2] = r[2]
            else:
                merged.append(list(r))
        runs = merged
    return runs


def split_on_team_flips(track_boxes_per_frame, det_votes,
                        window=5, min_run=6, min_agreement=0.75):
    """
    Split a raw track where the team judged from jersey color changes and
    stays changed. That pattern is an ID swap: the tracker moved the ID to
    a player on the other team.

    track_boxes_per_frame: per frame [(track_id, box)]
    det_votes: {track_id: [(frame, team 0/1)]} from clean detections only

    Returns (new_boxes_per_frame, seg_labels, info).
    seg_labels maps every segment ID of a split track to its team. The first
    segment keeps the original ID; later segments get new IDs.
    The split frame is the midpoint of the gap between the last vote of one
    team and the first vote of the other, which is usually the contact.
    """
    all_ids = {tid for frame in track_boxes_per_frame for tid, _ in frame}
    next_id = max(all_ids) + 1 if all_ids else 0

    split_frames, seg_ids, seg_labels, info = {}, {}, {}, []

    for tid, votes in det_votes.items():
        if len(votes) < 2 * min_run:
            continue
        votes = sorted(votes)
        frames = [f for f, _ in votes]
        vals = [v for _, v in votes]

        runs = _runs_from_votes(vals, window, min_run)
        if len(runs) < 2:
            continue

        # every run must be mostly one team, or the call is too ambiguous
        if any(vals[i0:i1 + 1].count(label) / (i1 - i0 + 1) < min_agreement
               for label, i0, i1 in runs):
            continue

        # Smoothing blurs where a run really ends, so place each split at the
        # point that fits the raw votes best. Ties go to the widest frame gap,
        # which is the contact where no clean crops exist.
        starts = []
        lo = runs[0][1]
        for k in range(len(runs) - 1):
            la, lb = runs[k][0], runs[k + 1][0]
            hi = runs[k + 1][2]
            best_j, best_cost, best_gap = None, None, None
            for j in range(lo + 1, hi + 1):
                cost = (sum(1 for v in vals[lo:j] if v != la)
                        + sum(1 for v in vals[j:hi + 1] if v != lb))
                gap = frames[j] - frames[j - 1]
                if (best_cost is None or cost < best_cost
                        or (cost == best_cost and gap > best_gap)):
                    best_j, best_cost, best_gap = j, cost, gap
            starts.append((frames[best_j - 1] + frames[best_j]) // 2 + 1)
            lo = best_j
        if any(s2 <= s1 for s1, s2 in zip(starts[:-1], starts[1:])):
            continue

        ids = [tid]
        for _ in starts:
            ids.append(next_id)
            next_id += 1
        split_frames[tid] = starts
        seg_ids[tid] = ids
        labels = [r[0] for r in runs]
        for sid, label in zip(ids, labels):
            seg_labels[sid] = label
        info.append({"track": tid, "split_frames": starts,
                     "segment_ids": ids, "teams": labels})

    new_boxes = []
    for f, tracks in enumerate(track_boxes_per_frame):
        row = []
        for tid, box in tracks:
            if tid in split_frames:
                tid = seg_ids[tid][bisect.bisect_right(split_frames[tid], f)]
            row.append((tid, box))
        new_boxes.append(row)
    return new_boxes, seg_labels, info