# TeamAssigner.py

from collections import defaultdict

import cv2
import numpy as np
from sports.common.team import TeamClassifier


def _torso(crop, x0, x1):
    h, w = crop.shape[:2]
    if h < 8 or w < 4:
        return None
    t = crop[int(0.15 * h):int(0.60 * h), int(x0 * w):int(x1 * w)]
    return t if t.size else None


def _hue_hist(crop, sat=70, val=50):
    """
    Hue histogram of the torso band, chromatic pixels only, as a share of all
    torso pixels. Blur and distance shrink the mass but keep the hue shape.
    Works when the teams wear different colors.
    """
    torso = _torso(crop, 0.15, 0.85)
    if torso is None:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0]
    colored = (hsv[..., 1] >= sat) & (hsv[..., 2] >= val)
    hist = np.histogram(hue[colored], bins=12, range=(0, 180))[0] / hue.size
    # hue is circular (red sits on both ends), so smooth across the wrap
    kernel = np.array([1, 2, 3, 2, 1]) / 9.0
    return np.convolve(np.pad(hist, 2, mode="wrap"), kernel, mode="valid")


def _full_feats(crop, grass_h=(35, 85), grass_s=35):
    """
    Brightness plus color: the share of dark, white, and gray pixels and a hue
    histogram, with grass pixels removed. Works when the teams differ in
    lightness (white vs black) and not in hue.
    """
    torso = _torso(crop, 0.2, 0.8)
    if torso is None:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    keep = ~((H >= grass_h[0]) & (H <= grass_h[1]) & (S >= grass_s) & (V >= 40))
    n = int(keep.sum())
    if n < 25:
        return None
    H, S, V = H[keep], S[keep], V[keep]
    dark = V < 70
    white = (~dark) & (S < 50) & (V >= 160)
    gray = (~dark) & (S < 50) & (V < 160)
    chroma = (~dark) & (S >= 50)
    hue = np.histogram(H[chroma], bins=12, range=(0, 180))[0].astype(float)
    hue = np.convolve(np.pad(hue, 2, mode="wrap"), np.array([1, 2, 3, 2, 1]) / 9.0, mode="valid")
    return np.concatenate([[dark.sum(), white.sum(), gray.sum()], hue]) / n


def color_feats(crop):
    """Both color descriptors for a crop: (hue histogram, brightness+color). Either can be None."""
    return (_hue_hist(crop), _full_feats(crop))


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _tv(a, b):
    """Total variation distance between two histograms: 0 identical, 1 disjoint."""
    return 0.5 * float(np.abs(a - b).sum())


class TeamAssigner:
    """
    Assigns each tracked player to one of two teams.

    Step 1: fit() on one crop per player from the pre-snap frame. Those
            labels are locked.
    Step 2: fit_color() learns each team's jersey look from the presnap
            window. It uses hue when the teams differ in hue and
            brightness plus color when they do not, or switches off.
    Step 3: assign_tracks() labels every other track by majority vote over
            clean crops.

    Usage:
        assigner = TeamAssigner(device="mps")
        assigner.fit(presnap_crops_by_track)       # {track_id: crop}
        assigner.fit_color(window_crops)           # {frame: {track_id: crop}}
        assigner.assign_tracks(samples_by_track)   # {track_id: [crops]}
        team_id = assigner.get_team(track_id)      # 0, 1, or None
    """

    def __init__(self, device="mps", min_samples=3, min_agreement=0.75,
                 color_min_mass=0.01, color_min_sim=0.5, color_max_overlap=0.7,
                 color_min_sep=0.35, color_min_margin=0.12):
        self.classifier = TeamClassifier(device=device)
        self.min_samples = min_samples        # clean crops needed before a vote counts
        self.min_agreement = min_agreement    # share of crops that must agree
        self.track_teams = {}   # {track_id: 0 or 1}
        self.track_votes = {}   # {track_id: (n_crops, agreement, voted_team)}
        self.locked = set()     # presnap tracks, labeled by fit()
        self.color_min_mass = color_min_mass        # hue mode: chromatic share needed to vote
        self.color_min_sim = color_min_sim          # hue mode: min cosine similarity to a team
        self.color_max_overlap = color_max_overlap  # hue mode: teams must differ in hue by this much
        self.color_min_sep = color_min_sep          # full mode: min distance between teams
        self.color_min_margin = color_min_margin    # full mode: min gap between the two distances
        self.color_mode = None       # "hue", "full", or None
        self.color_centroids = None
        self.fitted = False

    def fit(self, presnap_crops_by_track):
        track_ids = list(presnap_crops_by_track.keys())
        crops = list(presnap_crops_by_track.values())

        if len(crops) < 2:
            raise ValueError(f"Need at least 2 players to fit teams, got {len(crops)}")

        self.classifier.fit(crops)
        predictions = self.classifier.predict(crops)

        self.track_teams = {
            track_id: int(team_id)
            for track_id, team_id in zip(track_ids, predictions)
        }
        self.locked = set(track_ids)
        self.fitted = True

    def fit_color(self, crops_by_frame):
        """
        Learn each team's jersey look from the presnap window, using the labels
        fit() gave. Returns (mode, separation): mode is "hue", "full", or None
        (off), and separation is how far apart the teams are (higher is better).
        """
        feats = {0: [], 1: []}
        for frame_crops in crops_by_frame.values():
            for track_id, crop in frame_crops.items():
                team = self.track_teams.get(track_id)
                if team is not None:
                    feats[team].append(color_feats(crop))

        self.color_mode, self.color_centroids = None, None

        # 1) teams that differ in hue
        hue = {t: [_unit(f[0]) for f in feats[t]
                   if f[0] is not None and f[0].sum() >= self.color_min_mass] for t in (0, 1)}
        if len(hue[0]) >= 3 and len(hue[1]) >= 3:
            c0 = _unit(np.mean(hue[0], axis=0))
            c1 = _unit(np.mean(hue[1], axis=0))
            overlap = float(c0 @ c1)
            if overlap <= self.color_max_overlap:
                self.color_mode, self.color_centroids = "hue", (c0, c1)
                return "hue", 1.0 - overlap

        # 2) teams that differ in brightness (white vs black)
        full = {t: [f[1] for f in feats[t] if f[1] is not None] for t in (0, 1)}
        if len(full[0]) >= 3 and len(full[1]) >= 3:
            c0, c1 = np.mean(full[0], axis=0), np.mean(full[1], axis=0)
            sep = _tv(c0, c1)
            if sep >= self.color_min_sep:
                self.color_mode, self.color_centroids = "full", (c0, c1)
                return "full", sep
            return None, sep
        return None, None

    def _color_vote(self, crop):
        """0, 1, or None when the crop gives no clear answer."""
        return self.vote_feats(color_feats(crop))

    def vote_feats(self, feats):
        """Team vote from stored color features (hue, full): 0, 1, or None."""
        if self.color_mode is None or feats is None:
            return None
        c0, c1 = self.color_centroids
        if self.color_mode == "hue":
            h = feats[0]
            if h is None or h.sum() < self.color_min_mass:
                return None
            u = _unit(h)
            sims = np.array([u @ c0, u @ c1])
            if sims.max() < self.color_min_sim:
                return None
            return int(sims.argmax())
        f = feats[1]
        if f is None:
            return None
        d0, d1 = _tv(f, c0), _tv(f, c1)
        if abs(d0 - d1) < self.color_min_margin:
            return None
        return 0 if d0 < d1 else 1

    def assign_tracks(self, samples_by_track):
        """
        Label every track not already locked by fit(). With the color model on,
        each clean crop votes by color. Otherwise one batched classifier call.
        A track needs enough agreeing votes, or it stays unresolved
        (get_team returns None).
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before assign_tracks()")

        votes = defaultdict(list)

        if self.color_mode is not None:
            for track_id, samples in samples_by_track.items():
                if track_id in self.track_teams:
                    continue
                for crop in samples:
                    v = self._color_vote(crop)
                    if v is not None:
                        votes[track_id].append(v)
        else:
            flat_crops, owners = [], []
            for track_id, samples in samples_by_track.items():
                if track_id in self.track_teams:
                    continue
                for crop in samples:
                    flat_crops.append(crop)
                    owners.append(track_id)
            if not flat_crops:
                return
            predictions = self.classifier.predict(flat_crops)
            for track_id, team_id in zip(owners, predictions):
                votes[track_id].append(int(team_id))

        for track_id, vote_list in votes.items():
            n = len(vote_list)
            counts = np.bincount(vote_list, minlength=2)
            agreement = float(counts.max()) / n
            self.track_votes[track_id] = (n, round(agreement, 2), int(counts.argmax()))
            # No label is better than a wrong one: too few clean crops or a
            # split vote leaves the track unresolved (yellow).
            if n < self.min_samples or agreement < self.min_agreement:
                continue
            self.track_teams[track_id] = int(counts.argmax())

    def propagate_chains(self, remap, weights):
        """
        After stitching, give every track in a chain the chain's team.
        Presnap labels outweigh everything; otherwise the track with more
        frames wins. Returns the number of chains where labeled members
        disagreed (a sign of a bad link or a bad label).
        """
        chains = defaultdict(list)
        for track_id, root in remap.items():
            chains[root].append(track_id)

        conflicts = 0
        for members in chains.values():
            if len(members) < 2:
                continue
            score = np.zeros(2)
            for track_id in members:
                team = self.track_teams.get(track_id)
                if team is None:
                    continue
                score[team] += 1e6 if track_id in self.locked else weights.get(track_id, 1)
            if score.sum() == 0:
                continue
            if score[0] > 0 and score[1] > 0:
                conflicts += 1
            team = int(score.argmax())
            for track_id in members:
                self.track_teams[track_id] = team
        return conflicts

    def get_team(self, track_id):
        """Returns the team (0 or 1) or None"""
        return self.track_teams.get(track_id)

    def summary(self):
        """debug"""
        if not self.fitted:
            return "not fitted yet"
        values, counts = np.unique(list(self.track_teams.values()), return_counts=True)
        return dict(zip(values.tolist(), counts.tolist()))