# TeamAssigner.py

import numpy as np
from sports.common.team import TeamClassifier

#presnap team detector
class TeamAssigner:
    """
    Assigns each tracked player to one of two teams, using a single
    representative crop per player from the pre-snap window.

    Usage:
        assigner = TeamAssigner(device="mps")
        assigner.fit(presnap_crops_by_track)   # dict: {track_id: crop}
        team_id = assigner.get_team(track_id)  # 0 or 1, locked forever after fit
    """

    def __init__(self, device="mps"):
        self.classifier = TeamClassifier(device=device)
        self.track_teams = {}   # {track_id: 0 or 1} — set once, in fit()
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
        self.fitted = True

    def get_team(self, track_id):
        """Returns the locked team (0 or 1) or None"""
        return self.track_teams.get(track_id)

    def summary(self):
        """debug"""
        if not self.fitted:
            return "not fitted yet"
        values, counts = np.unique(list(self.track_teams.values()), return_counts=True)
        return dict(zip(values.tolist(), counts.tolist()))