import cv2

TEAM_COLORS = {
    0: (60, 60, 235),     # team A color
    1: (235, 140, 60),    # team B color
    None: (0, 220, 220),  # unresolved color
}


def _draw_labeled_box(frame, box, label, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    label_y1 = max(y1 - th - 10, 0)
    cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, max(y1 - 6, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def render_annotated_video(result, output_path):
    """Re-reads the source video, so no frames are held in memory."""
    track_boxes_per_frame = result["track_boxes_per_frame"]
    assigner = result["assigner"]
    remap = result["remap"]
    fps = result["fps"]

    cap = cv2.VideoCapture(result["video_path"])
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame_tracks in track_boxes_per_frame:
        ok, frame = cap.read()
        if not ok:
            break

        for track_id, box in frame_tracks:
            canonical_id = remap.get(track_id, track_id)
            team_id = assigner.get_team(canonical_id)
            color = TEAM_COLORS[team_id]
            label = f"#{canonical_id}" if team_id is None else f"T{team_id} #{canonical_id}"
            _draw_labeled_box(frame, box, label, color)

        out.write(frame)

    cap.release()
    out.release()
    print(f"saved to {output_path}")