import cv2

TEAM_COLORS = {
    0: (60, 60, 235),     # team A 
    1: (235, 140, 60),    # team B 
    None: (0, 220, 220),  # unresolved 
}

def _draw_labeled_box(frame, box, label, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    label_y1 = max(y1 - th - 10, 0)
    cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

def render_annotated_video(result, output_path, fps=30):
    frames = result["frames"]
    track_boxes_per_frame = result["track_boxes_per_frame"]
    remap = result["remap"]
    resolved_teams = result["resolved_teams"] 

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame_num, frame_tracks in enumerate(track_boxes_per_frame):
        frame = frames[frame_num].copy()
        for track_id, box, crop in frame_tracks:
            canonical_id = remap.get(track_id, track_id)
            team_id = resolved_teams.get(canonical_id)
            color = TEAM_COLORS.get(team_id, TEAM_COLORS[None])
            label = f"#{canonical_id}" if team_id is None else f"T{team_id} #{canonical_id}"
            _draw_labeled_box(frame, box, label, color)
        out.write(frame)

    out.release()
    print(f"saved to {output_path}")
