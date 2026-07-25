# TrackStitcher.py

import numpy as np

""""""
class TrackStitcher:
    def __init__(self, max_gap_frames=30, max_dist=150):
        self.max_gap_frames = max_gap_frames
        self.max_dist = max_dist

    def build_summaries(self, track_boxes_per_frame):
        summaries = {}
        for frame_num, frame_tracks in enumerate(track_boxes_per_frame):
            for track_id, box, crop in frame_tracks:
                if track_id not in summaries:
                    summaries[track_id] = {
                        "start_frame": frame_num, "end_frame": frame_num,
                        "start_box": box, "end_box": box,
                        "start_crop": crop, "end_crop": crop,
                    }
                else:
                    summaries[track_id]["end_frame"] = frame_num
                    summaries[track_id]["end_box"] = box
                    summaries[track_id]["end_crop"] = crop
        return summaries

    def _center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _dist(self, p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

    def stitch(self, summaries, assigner=None, embed_fn=None):
        track_ids = list(summaries.keys())
        remap = {tid: tid for tid in track_ids}
        track_ids_by_start = sorted(track_ids, key=lambda t: summaries[t]["start_frame"])

        for ending_id in track_ids_by_start:
            end_frame = summaries[ending_id]["end_frame"]
            end_pos = self._center(summaries[ending_id]["end_box"])

            candidates = []
            for starting_id in track_ids:
                if starting_id == ending_id:
                    continue
                gap = summaries[starting_id]["start_frame"] - end_frame
                if gap <= 3 or gap > self.max_gap_frames:
                    continue
                dist = self._dist(end_pos, self._center(summaries[starting_id]["start_box"]))
                if dist > self.max_dist:
                    continue
                candidates.append((starting_id, gap, dist))

            if not candidates:
                continue
            chosen_id = candidates[0][0] if len(candidates) == 1 else \
                self._break_tie(ending_id, candidates, summaries, assigner, embed_fn)

            remap[chosen_id] = remap[ending_id]

        return remap

    def _break_tie(self, ending_id, candidates, summaries, assigner, embed_fn):
        scored = []
        ending_team = assigner.get_team(ending_id) if assigner else None
        ending_embed = embed_fn(summaries[ending_id]["end_crop"]) if embed_fn else None

        for starting_id, gap, dist in candidates:
            team_match = False
            appearance_score = 0.0

            if assigner and ending_team is not None:
                predicted_team = int(assigner.classifier.predict([summaries[starting_id]["start_crop"]])[0])
                team_match = (predicted_team == ending_team)

            if embed_fn and ending_embed is not None:
                start_embed = embed_fn(summaries[starting_id]["start_crop"])
                appearance_score = float(np.dot(ending_embed, start_embed) /
                    (np.linalg.norm(ending_embed) * np.linalg.norm(start_embed) + 1e-8))

            scored.append((starting_id, dist, team_match, appearance_score))

        scored.sort(key=lambda s: (not s[2], -s[3], s[1]))
        return scored[0][0]
