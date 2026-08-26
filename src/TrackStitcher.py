# TrackStitcher.py
import numpy as np
from scipy.optimize import linear_sum_assignment

BIG_COST = 1e6         
FEASIBLE_CUTOFF = 50.0  


class TrackStitcher:
    """
    Reunites track IDs that got broken (e.g. during player contact) by matching
    each track's ending position/appearance against every other track's starting
    position/appearance, and solving the matching optimally (not greedily).
    """
    def __init__(self, max_gap_frames=30, max_speed_px_per_frame=15):
        self.max_gap_frames = max_gap_frames
    
        self.max_speed_px_per_frame = max_speed_px_per_frame

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
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def _pair_cost(self, ending_id, starting_id, summaries, assigner, embed_fn):
        """Lower = better match. Returns BIG_COST if the pair is infeasible."""
        end_frame = summaries[ending_id]["end_frame"]
        start_frame = summaries[starting_id]["start_frame"]
        gap = start_frame - end_frame

        if gap <= 0 or gap > self.max_gap_frames:
            return BIG_COST

        dist = self._dist(self._center(summaries[ending_id]["end_box"]),
                           self._center(summaries[starting_id]["start_box"]))
      
        budget = self.max_speed_px_per_frame * gap
        if dist > budget:
            return BIG_COST
        dist_cost = dist / budget  

        team_cost = 0.0
        if assigner is not None:
            end_team = assigner.classifier.predict([summaries[ending_id]["end_crop"]])[0]
            start_team = assigner.classifier.predict([summaries[starting_id]["start_crop"]])[0]
            team_cost = 0.0 if int(end_team) == int(start_team) else 1.0

        appearance_cost = 0.5 
        if embed_fn is not None:
            e_embed = embed_fn(summaries[ending_id]["end_crop"])
            s_embed = embed_fn(summaries[starting_id]["start_crop"])
            sim = float(np.dot(e_embed, s_embed) /
                        (np.linalg.norm(e_embed) * np.linalg.norm(s_embed) + 1e-8))
            appearance_cost = (1 - sim) / 2 
            
        return dist_cost + 2.0 * team_cost + appearance_cost

    def stitch(self, summaries, assigner=None, embed_fn=None):
        track_ids = list(summaries.keys())
        n = len(track_ids)
        idx_of = {tid: i for i, tid in enumerate(track_ids)}

        cost = np.full((n, n), BIG_COST, dtype=float)
        for i, ending_id in enumerate(track_ids):
            for j, starting_id in enumerate(track_ids):
                if ending_id == starting_id:
                    continue
                cost[i, j] = self._pair_cost(ending_id, starting_id, summaries, assigner, embed_fn)

        row_idx, col_idx = linear_sum_assignment(cost)

      
        parent = {tid: tid for tid in track_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, j in zip(row_idx, col_idx):
            if cost[i, j] < FEASIBLE_CUTOFF:
                ending_id, starting_id = track_ids[i], track_ids[j]
                union(ending_id, starting_id)

        
        remap = {}
        for tid in track_ids:
            root = find(tid)
            remap[tid] = root
        return remap
