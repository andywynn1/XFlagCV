# pipeline.py

import time

import cv2
import numpy as np
from ultralytics import YOLO

from TeamAssigner import TeamAssigner, color_feats
from TrackSplitter import split_on_team_flips
from TrackStitcher import TrackStitcher


def _store_crop(track_crops, track_id, crop, n_samples, rng):
    """Keep first crop, last crop, and a reservoir sample across the track."""
    entry = track_crops.get(track_id)
    if entry is None:
        track_crops[track_id] = {"start": crop, "end": crop, "samples": [crop], "seen": 1}
        return
    entry["end"] = crop
    entry["seen"] += 1
    if len(entry["samples"]) < n_samples:
        entry["samples"].append(crop)
    else:
        j = int(rng.integers(entry["seen"]))
        if j < n_samples:
            entry["samples"][j] = crop


def _overlap_fractions(boxes):
    """For each box, the largest share of its own area covered by another box."""
    out = []
    for i, a in enumerate(boxes):
        area = max((a[2] - a[0]) * (a[3] - a[1]), 1)
        worst = 0.0
        for j, b in enumerate(boxes):
            if i == j:
                continue
            iw = min(a[2], b[2]) - max(a[0], b[0])
            ih = min(a[3], b[3]) - max(a[1], b[1])
            if iw > 0 and ih > 0:
                worst = max(worst, iw * ih / area)
        out.append(worst)
    return out


def run_full_pipeline(video_path, model_path, presnap_frame_target,
                      tracker="botsort.yaml", device="mps",
                      max_gap_frames=30, max_dist=None,
                      presnap_window=10, samples_per_track=8,
                      progress_every=25, overlap_max=0.2, nms_iou=0.5,
                      split_on_flip=True):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    t0 = time.time()
    print(f"[1/4] video: {total_frames} frames at {fps:.1f} fps. loading model...", flush=True)

    model = YOLO(model_path)
    print(f"      model loaded in {time.time() - t0:.1f}s. tracking...", flush=True)
    rng = np.random.default_rng(0)

    # ---- Pass 1: track once. Keep boxes and a few crops, not frames. ----
    track_boxes_per_frame = []   # per frame: [(track_id, (x1, y1, x2, y2))]
    track_crops = {}             # {track_id: {"start", "end", "samples", "seen"}}
    window_crops = {}            # {frame_num: {track_id: crop}} near the presnap target
    det_feats = {}               # {track_id: [(frame_num, color features)]} clean detections

    lo = max(0, presnap_frame_target - presnap_window)
    hi = presnap_frame_target + presnap_window

    results_gen = model.track(
        source=video_path, tracker=tracker,
        iou=nms_iou, conf=0.4, device=device,
        stream=True, verbose=False,
    )

    t_track = time.time()
    for frame_num, r in enumerate(results_gen):
        if frame_num % progress_every == 0:
            el = time.time() - t_track
            print(f"      frame {frame_num}/{total_frames}  ({el:.1f}s elapsed, "
                  f"{(frame_num + 1) / max(el, 1e-6):.1f} fps)", flush=True)
        frame = r.orig_img
        fh, fw = frame.shape[:2]
        frame_tracks = []

        if r.boxes.id is not None:
            for box, track_id, cls_id in zip(r.boxes.xyxy.cpu().numpy(),
                                              r.boxes.id.cpu().numpy(),
                                              r.boxes.cls.cpu().numpy()):
                if r.names[int(cls_id)] != "player":
                    continue
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, fw), min(y2, fh)
                if x2 <= x1 or y2 <= y1:
                    continue
                frame_tracks.append((int(track_id), (x1, y1, x2, y2)))

        # Crops are only kept when the box is mostly free of other players.
        # Crops taken during contact mix two jerseys and vote for the wrong team.
        overlaps = _overlap_fractions([b for _, b in frame_tracks])
        in_window = lo <= frame_num <= hi
        for (tid, (x1, y1, x2, y2)), frac in zip(frame_tracks, overlaps):
            clean = frac <= overlap_max
            if not (clean or in_window):
                continue
            crop = frame[y1:y2, x1:x2].copy()   # copy so the full frame can be freed
            if clean:
                _store_crop(track_crops, tid, crop, samples_per_track, rng)
                det_feats.setdefault(tid, []).append((frame_num, color_feats(crop)))
            if in_window:
                window_crops.setdefault(frame_num, {})[tid] = crop

        track_boxes_per_frame.append(frame_tracks)

    if not track_boxes_per_frame:
        raise ValueError(f"No frames read from {video_path}")

    print(f"[2/4] tracking done: {len(track_boxes_per_frame)} frames in "
          f"{time.time() - t_track:.1f}s, "
          f"{len({t for f in track_boxes_per_frame for t, _ in f})} raw track IDs", flush=True)

    # ---- Team assignment ----
    # Fit on the frame near the target with the most detected players.
    if not window_crops:
        raise ValueError(
            f"No player detections within {presnap_window} frames of "
            f"frame {presnap_frame_target} (video has {len(track_boxes_per_frame)} frames)"
        )
    best_frame = max(window_crops, key=lambda f: len(window_crops[f]))

    print(f"[3/4] fitting teams on frame {best_frame} "
          f"({len(window_crops[best_frame])} players)...", flush=True)
    t_team = time.time()
    assigner = TeamAssigner(device=device)
    assigner.fit(window_crops[best_frame])
    mode, score = assigner.fit_color(window_crops)
    if mode is None:
        sep = "n/a" if score is None else f"{score:.2f}"
        print(f"      color model off (teams not separable by color, separation {sep}): "
              "using the embedding classifier", flush=True)
    else:
        print(f"      color model on: {mode} mode (team separation {score:.2f}, higher is better)",
              flush=True)
    print(f"      fit done in {time.time() - t_team:.1f}s. voting on other tracks...", flush=True)
    assigner.assign_tracks({tid: e["samples"] for tid, e in track_crops.items()})

    print(f"      teams done in {time.time() - t_team:.1f}s", flush=True)

    # ---- Split tracks where the jersey color flips (tracker ID swaps) ----
    splits = []
    if split_on_flip and assigner.color_mode is not None:
        det_votes = {}
        for tid, lst in det_feats.items():
            votes = [(f, assigner.vote_feats(ft)) for f, ft in lst]
            votes = [(f, v) for f, v in votes if v is not None]
            if votes:
                det_votes[tid] = votes
        track_boxes_per_frame, seg_labels, splits = split_on_team_flips(
            track_boxes_per_frame, det_votes)
        for seg_id, label in seg_labels.items():
            if seg_id not in assigner.locked:      # presnap labels stay locked
                assigner.track_teams[seg_id] = label
        for sp in splits:
            print(f"      split raw ID {sp['track']} at frames {sp['split_frames']} "
                  f"-> IDs {sp['segment_ids']}, teams {sp['teams']}", flush=True)
        if not splits:
            print("      no team flips found inside any track", flush=True)
    all_ids = {t for f in track_boxes_per_frame for t, _ in f}
    no_team = sorted(t for t in all_ids if assigner.get_team(t) is None)
    print(f"      {len(no_team)} of {len(all_ids)} raw IDs have no team yet: {no_team}", flush=True)

    # ---- Stitching ----
    print("[4/4] stitching...", flush=True)
    stitcher = TrackStitcher(max_gap_frames=max_gap_frames, max_dist=max_dist)
    summaries = stitcher.build_summaries(track_boxes_per_frame, track_crops)
    remap = stitcher.stitch(summaries, assigner=assigner)
    print(f"      done. total {time.time() - t0:.1f}s", flush=True)

    return {
        "video_path": video_path,
        "fps": fps,
        "track_boxes_per_frame": track_boxes_per_frame,
        "assigner": assigner,
        "remap": remap,
        "team_summary": assigner.summary(),
        "presnap_frame_used": best_frame,
        "stitch_report": stitcher.report,
        "splits": splits,
    }