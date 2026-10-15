# TrackStitcher.py

import numpy as np
from scipy.optimize import linear_sum_assignment

INVALID = 1e6


class TrackStitcher:
    """
    Joins track fragments that belong to one player.

    A link (A -> B) means B starts after A ends, close to where A was
    heading, on the same team. Links are chosen globally so each track
    has at most one successor and one predecessor.
    """

    def __init__(self, max_gap_frames=30, max_dist=None,
                 base_heights=1.5, heights_per_frame=0.05,
                 gap_weight=0.02, unmatched_cost=0.6,
                 embed_weight=0.5, vel_frames=5, max_project_frames=10):
        self.max_gap_frames = max_gap_frames
        self.max_dist = max_dist                  # optional absolute pixel cap
        self.base_heights = base_heights          # search radius at gap 0, in box heights
        self.heights_per_frame = heights_per_frame  # radius growth per gap frame
        self.gap_weight = gap_weight
        self.unmatched_cost = unmatched_cost      # lower = stricter linking
        self.embed_weight = embed_weight
        self.vel_frames = vel_frames
        self.max_project_frames = max_project_frames
        self.report = {}

    # ---------- summaries ----------

    def build_summaries(self, track_boxes_per_frame, track_crops=None):
        summaries = {}
        for frame_num, frame_tracks in enumerate(track_boxes_per_frame):
            for track_id, box in frame_tracks:
                center = self._center(box)
                s = summaries.get(track_id)
                if s is None:
                    summaries[track_id] = {
                        "start_frame": frame_num, "end_frame": frame_num,
                        "start_box": box, "end_box": box,
                        "count": 1,
                        "recent": [(frame_num, center)],
                    }
                else:
                    s["end_frame"] = frame_num
                    s["end_box"] = box
                    s["count"] += 1
                    s["recent"].append((frame_num, center))
                    if len(s["recent"]) > self.vel_frames:
                        s["recent"].pop(0)

        if track_crops:
            for track_id, s in summaries.items():
                crops = track_crops.get(track_id)
                if crops:
                    s["start_crop"] = crops["start"]
                    s["end_crop"] = crops["end"]
        return summaries

    def _center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _dist(self, p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def _velocity(self, summary):
        recent = summary["recent"]
        if len(recent) < 2:
            return (0.0, 0.0)
        (f0, c0), (f1, c1) = recent[0], recent[-1]
        dt = max(f1 - f0, 1)
        return ((c1[0] - c0[0]) / dt, (c1[1] - c0[1]) / dt)

    # ---------- stitching ----------

    def stitch(self, summaries, assigner=None, embed_fn=None):
        ids = list(summaries.keys())
        n = len(ids)
        remap = {tid: tid for tid in ids}
        self.report = {"raw_tracks": n, "links": 0, "team_rejections": 0}
        if n < 2:
            self.report["canonical_tracks"] = n
            return remap

        end_embeds, start_embeds = {}, {}
        if embed_fn:
            for tid in ids:
                s = summaries[tid]
                if "end_crop" in s:
                    end_embeds[tid] = embed_fn(s["end_crop"])
                    start_embeds[tid] = embed_fn(s["start_crop"])

        cost = np.full((n, n), INVALID)

        for i, e_id in enumerate(ids):
            e = summaries[e_id]
            end_center = self._center(e["end_box"])
            box_h = max(e["end_box"][3] - e["end_box"][1], 1)
            vx, vy = self._velocity(e)
            e_team = assigner.get_team(e_id) if assigner else None

            for j, s_id in enumerate(ids):
                if i == j:
                    continue
                s = summaries[s_id]
                gap = s["start_frame"] - e["end_frame"]
                if gap < 1 or gap > self.max_gap_frames:
                    continue

                # hard team filter
                s_team = assigner.get_team(s_id) if assigner else None
                if e_team is not None and s_team is not None and e_team != s_team:
                    self.report["team_rejections"] += 1
                    continue

                # project the end position forward along its last velocity
                steps = min(gap, self.max_project_frames)
                projected = (end_center[0] + vx * steps, end_center[1] + vy * steps)
                dist = self._dist(projected, self._center(s["start_box"]))

                limit = box_h * (self.base_heights + self.heights_per_frame * gap)
                if self.max_dist is not None:
                    limit = min(limit, self.max_dist)
                if dist > limit:
                    continue

                c = dist / limit + self.gap_weight * gap

                if e_id in end_embeds and s_id in start_embeds:
                    a, b = end_embeds[e_id], start_embeds[s_id]
                    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
                    c += self.embed_weight * (1.0 - cos)

                cost[i, j] = c

        links = self._solve(cost)
        self.report["links"] = len(links)

        parent = {ids[j]: ids[i] for i, j in links}

        def root(tid):
            while tid in parent:
                tid = parent[tid]
            return tid

        remap = {tid: root(tid) for tid in ids}
        self.report["canonical_tracks"] = len(set(remap.values()))

        # every track in a chain takes the chain's team, so a short fragment
        # with no label of its own inherits it
        if assigner is not None and hasattr(assigner, "propagate_chains"):
            weights = {tid: summaries[tid]["count"] for tid in ids}
            self.report["team_conflicts"] = assigner.propagate_chains(remap, weights)
        return remap

    def _solve(self, cost):
        """
        Global one-to-one matching with an 'unmatched' option. A link is
        made only when it is cheaper than leaving both tracks unlinked.
        """
        n = cost.shape[0]
        u = self.unmatched_cost
        big = np.zeros((2 * n, 2 * n))
        big[:n, :n] = cost
        big[:n, n:] = np.where(np.eye(n) > 0, u, INVALID)
        big[n:, :n] = np.where(np.eye(n) > 0, u, INVALID)
        # bottom-right block stays 0 (dummy to dummy)

        rows, cols = linear_sum_assignment(big)
        return [(r, c) for r, c in zip(rows, cols)
                if r < n and c < n and cost[r, c] < INVALID]