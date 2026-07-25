# pipeline.py

from ultralytics import YOLO
from TeamAssigner import TeamAssigner
from TrackStitcher import TrackStitcher
import cv2


def run_full_pipeline(video_path, model_path, presnap_frame_target,
                       tracker="botsort.yaml", device="mps",
                       max_gap_frames=30, max_dist=150):

    model = YOLO(model_path)

    # ---- Pass 1: run tracking once, cache everything ----
    all_frames_cache = []
    track_boxes_per_frame = []

    results_gen = model.track(
        source=video_path, tracker=tracker,
        iou=0.5, conf=0.4, device=device,
        stream=True, verbose=False,
    )

    for frame_num, r in enumerate(results_gen):
        frame = r.orig_img
        all_frames_cache.append(frame)

        if r.boxes.id is None:
            track_boxes_per_frame.append([])
            continue

        frame_tracks = []
        for box, track_id, cls_id in zip(r.boxes.xyxy.cpu().numpy(),
                                          r.boxes.id.cpu().numpy(),
                                          r.boxes.cls.cpu().numpy()):
            if r.names[int(cls_id)] != "player":
                continue
            track_id = int(track_id)
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            frame_tracks.append((track_id, (x1, y1, x2, y2), crop))

        track_boxes_per_frame.append(frame_tracks)

    # ---- Team assignment (presnap, once) ----
    presnap_crops_by_track = {
        tid: crop for tid, box, crop in track_boxes_per_frame[presnap_frame_target]
    }
    assigner = TeamAssigner(device=device)
    assigner.fit(presnap_crops_by_track)

    # ---- Stitching (cleans up broken IDs) ----
    stitcher = TrackStitcher(max_gap_frames=max_gap_frames, max_dist=max_dist)
    summaries = stitcher.build_summaries(track_boxes_per_frame)
    remap = stitcher.stitch(summaries, assigner=assigner)

    return {
        "frames": all_frames_cache,
        "track_boxes_per_frame": track_boxes_per_frame,
        "assigner": assigner,
        "remap": remap,
        "team_summary": assigner.summary(),
    }
