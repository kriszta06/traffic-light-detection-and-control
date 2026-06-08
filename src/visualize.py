"""Draw detection boxes on a frame."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .detector import Detection


COLOR = {
    "red":   (0, 0, 255),
    "green": (0, 255, 0),
}


def draw_detections(frame_bgr, dets, thickness=2, show_score=True):
    out = frame_bgr.copy()
    for d in dets:
        c = COLOR.get(d.label, (255, 255, 255))
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), c, thickness)
        label = f"{d.label} {d.score:.2f}" if show_score else d.label
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(d.y1 - 4, th + 2)
        cv2.rectangle(out, (d.x1, ty - th - 2),
                      (d.x1 + tw + 2, ty + bl - 2), c, -1)
        cv2.putText(out, label, (d.x1 + 1, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


_GT_COLOR = {"red": (255, 255, 0), "green": (0, 255, 255)}


def draw_ground_truth(frame_bgr, gts, thickness=1):
    out = frame_bgr.copy()
    for x1, y1, x2, y2, label in gts:
        c = _GT_COLOR.get(label, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), c, thickness, cv2.LINE_AA)
        cv2.putText(out, "gt:" + label, (x1, max(0, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
    return out
