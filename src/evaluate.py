from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .detector import Detection


@dataclass
class GroundTruth:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str  


def _tag_to_label(tag: str) -> str | None:
    t = tag.lower()
    if t.startswith("stop"):
        return "red"
    if t.startswith("go"):
        return "green"
    return None  


def load_annotations(csv_path: str) -> Dict[str, List[GroundTruth]]:

    out: Dict[str, List[GroundTruth]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            label = _tag_to_label(row["Annotation tag"])
            if label is None:
                continue
            fname = os.path.basename(row["Filename"])
            try:
                x1 = int(row["Upper left corner X"])
                y1 = int(row["Upper left corner Y"])
                x2 = int(row["Lower right corner X"])
                y2 = int(row["Lower right corner Y"])
            except (KeyError, ValueError):
                continue
            out[fname].append(GroundTruth(x1, y1, x2, y2, label))
    return out


def _iou(a: Tuple[int, int, int, int],
         b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


@dataclass
class FrameResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0


def evaluate_frame(dets: List[Detection],
                   gts: List[GroundTruth],
                   iou_threshold: float = 0.3) -> FrameResult:
    """Greedy per-class matching."""
    res = FrameResult()
    used_gt: set[int] = set()
    dets_sorted = sorted(dets, key=lambda d: d.score, reverse=True)
    for d in dets_sorted:
        best_iou, best_idx = 0.0, -1
        for i, g in enumerate(gts):
            if i in used_gt or g.label != d.label:
                continue
            iou = _iou((d.x1, d.y1, d.x2, d.y2),
                       (g.x1, g.y1, g.x2, g.y2))
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= iou_threshold and best_idx >= 0:
            res.tp += 1
            used_gt.add(best_idx)
        else:
            res.fp += 1
    res.fn = len(gts) - len(used_gt)
    return res


@dataclass
class Summary:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def add(self, fr: FrameResult) -> None:
        self.tp += fr.tp
        self.fp += fr.fp
        self.fn += fr.fn

    def __str__(self) -> str:
        return (f"TP={self.tp}  FP={self.fp}  FN={self.fn}  "
                f"Precision={self.precision:.3f}  "
                f"Recall={self.recall:.3f}  F1={self.f1:.3f}")
