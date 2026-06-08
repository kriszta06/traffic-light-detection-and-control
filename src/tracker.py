from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .detector import Detection


@dataclass
class _Track:
    cx: float
    cy: float
    label: str
    hits: int = 1
    last_seen_frame: int = 0
    last_det: Detection = None  


class Tracker:
    def __init__(self,
                 min_hits: int = 2,
                 max_age: int = 3,
                 match_dist: int = 40):
        self.min_hits = min_hits
        self.max_age = max_age
        self.match_dist2 = match_dist * match_dist
        self.tracks: List[_Track] = []
        self.frame = 0

    @staticmethod
    def _centre(d: Detection):
        return (d.x1 + d.x2) / 2.0, (d.y1 + d.y2) / 2.0

    def update(self, dets: List[Detection]) -> List[Detection]:
        self.frame += 1

        used_tracks = set()
        used_dets = set()
        for di, d in enumerate(dets):
            cx, cy = self._centre(d)
            best, best_d2 = -1, self.match_dist2 + 1
            for ti, t in enumerate(self.tracks):
                if ti in used_tracks or t.label != d.label:
                    continue
                d2 = (t.cx - cx) ** 2 + (t.cy - cy) ** 2
                if d2 < best_d2:
                    best_d2, best = d2, ti
            if best >= 0:
                t = self.tracks[best]
                t.cx, t.cy = cx, cy
                t.hits += 1
                t.last_seen_frame = self.frame
                t.last_det = d
                used_tracks.add(best)
                used_dets.add(di)


        for di, d in enumerate(dets):
            if di in used_dets:
                continue
            cx, cy = self._centre(d)
            self.tracks.append(_Track(cx=cx, cy=cy, label=d.label,
                                      hits=1,
                                      last_seen_frame=self.frame,
                                      last_det=d))


        self.tracks = [t for t in self.tracks
                       if self.frame - t.last_seen_frame <= self.max_age]

        return [t.last_det for t in self.tracks if t.hits >= self.min_hits]

    def reset(self):
        self.tracks.clear()
        self.frame = 0
