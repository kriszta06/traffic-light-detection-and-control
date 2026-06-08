from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

RED_STRICT_LO_1 = np.array([0,   70, 150], dtype=np.uint8)
RED_STRICT_HI_1 = np.array([8,  255, 255], dtype=np.uint8)
RED_STRICT_LO_2 = np.array([172, 70, 150], dtype=np.uint8)
RED_STRICT_HI_2 = np.array([180, 255, 255], dtype=np.uint8)
RED_WIDE_LO_1   = np.array([0,   70, 150], dtype=np.uint8)
RED_WIDE_HI_1   = np.array([18, 255, 255], dtype=np.uint8)
RED_WIDE_LO_2   = np.array([160, 70, 150], dtype=np.uint8)
RED_WIDE_HI_2   = np.array([180, 255, 255], dtype=np.uint8)
RED_ANCHOR_DILATE = 3
RED_WIDE_MAX_Y    = 0.32      

GREEN_LOWER = np.array([40,  60, 150], dtype=np.uint8)
GREEN_UPPER = np.array([95, 255, 255], dtype=np.uint8)

MIN_AREA            = 8
MAX_AREA_FRAC       = 0.02
MIN_ASPECT          = 0.5
MAX_ASPECT          = 1.6
MIN_EXTENT          = 0.5
VERTICAL_MAX_FRAC   = 0.48   
DARK_SURROUND_VMAX  = 95

ABOVE_DARK_VMAX      = 110
ABOVE_HEIGHT_FACTOR  = 1.5
ABOVE_CHECK_MIN_Y    = 0.30
PAIR_Y_TOL           = 0.5
PAIR_SIZE_TOL        = 0.6
PAIR_MIN_DIST_FACTOR = 1.5
PAIR_MAX_DIST_FACTOR = 25.0

SKY_BULB_MAX_Y       = 0.25
SKY_BULB_MIN_MEAN_V  = 215      
SKY_BULB_MIN_MEAN_S  = 170       


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    score: float

    def as_tuple(self):
        return (self.x1, self.y1, self.x2, self.y2, self.label, self.score)

def _red_mask(hsv):
    H = hsv.shape[0]
    strict = cv2.bitwise_or(
        cv2.inRange(hsv, RED_STRICT_LO_1, RED_STRICT_HI_1),
        cv2.inRange(hsv, RED_STRICT_LO_2, RED_STRICT_HI_2))
    wide = cv2.bitwise_or(
        cv2.inRange(hsv, RED_WIDE_LO_1, RED_WIDE_HI_1),
        cv2.inRange(hsv, RED_WIDE_LO_2, RED_WIDE_HI_2))
    k = np.ones((RED_ANCHOR_DILATE, RED_ANCHOR_DILATE), np.uint8)
    grown = cv2.bitwise_and(wide, cv2.dilate(strict, k))
    out = strict.copy()
    top = int(H * RED_WIDE_MAX_Y)
    out[:top] = cv2.bitwise_or(out[:top], grown[:top])
    return out


def _green_mask(hsv):
    return cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)


def _clean(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask

def _dark_surround_score(v, x, y, w, h):
    H, W = v.shape
    pad = max(w, h)
    x0 = max(0, x - pad);  y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
    region = v[y0:y1, x0:x1].astype(np.int32)
    if region.size == 0:
        return 0.0
    bx0, by0 = x - x0, y - y0
    region[by0:by0 + h, bx0:bx0 + w] = -1
    ring = region[region >= 0]
    if ring.size == 0:
        return 0.0
    mean_v = float(ring.mean())
    return float(np.clip(1.0 - (mean_v / DARK_SURROUND_VMAX), 0.0, 1.0))


def _above_mean_v(v, x, y, w, h):
    H, W = v.shape
    pad_x = max(1, w // 4)
    cx0 = max(0, x - pad_x)
    cx1 = min(W, x + w + pad_x)
    ah = int(round(h * ABOVE_HEIGHT_FACTOR))
    ay1 = y
    ay0 = max(0, ay1 - ah)
    if ay1 - ay0 < 2:
        return 0.0
    col = v[ay0:ay1, cx0:cx1]
    if col.size == 0:
        return 0.0
    return float(col.mean())


def _filter_blobs(mask, hsv, label, frame_shape):
    H, W = frame_shape
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    max_area = MAX_AREA_FRAC * H * W
    n, labels_img, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_AREA or area > max_area:
            continue
        aspect = w / max(h, 1)
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue
        extent = area / float(w * h)
        if extent < MIN_EXTENT:
            continue
        cy = y + h / 2.0
        if cy / H > VERTICAL_MAX_FRAC:
            continue
        ds = _dark_surround_score(v, x, y, w, h)
        if cy / H < SKY_BULB_MAX_Y:
            blob_pix = labels_img[y:y + h, x:x + w] == i
            if not blob_pix.any():
                continue
            mean_v = float(v[y:y + h, x:x + w][blob_pix].mean())
            mean_s = float(s[y:y + h, x:x + w][blob_pix].mean())
            if mean_v < SKY_BULB_MIN_MEAN_V or mean_s < SKY_BULB_MIN_MEAN_S:
                continue
        else:
            if ds < 0.2:
                continue
        if label == "red" and cy / H > ABOVE_CHECK_MIN_Y:
            if _above_mean_v(v, x, y, w, h) > ABOVE_DARK_VMAX:
                continue
        vfac = 1.0 - 0.45 * (cy / H) / VERTICAL_MAX_FRAC
        score = float(np.clip((0.5 * extent + 0.5 * ds) * vfac, 0.0, 1.0))
        out.append(Detection(int(x), int(y), int(x + w), int(y + h), label, score))
    return out




def _reject_brake_pairs(reds):
    if len(reds) < 2:
        return reds
    rejected = set()
    for i, a in enumerate(reds):
        if i in rejected:
            continue
        ha = a.y2 - a.y1
        wa = a.x2 - a.x1
        cya = (a.y1 + a.y2) / 2.0
        cxa = (a.x1 + a.x2) / 2.0
        for j in range(i + 1, len(reds)):
            if j in rejected:
                continue
            b = reds[j]
            hb = b.y2 - b.y1
            wb = b.x2 - b.x1
            cyb = (b.y1 + b.y2) / 2.0
            cxb = (b.x1 + b.x2) / 2.0
            ref_h = (ha + hb) / 2.0
            ref_w = (wa + wb) / 2.0
            if ref_h <= 0:
                continue
            if abs(cya - cyb) / ref_h > PAIR_Y_TOL:
                continue
            size_diff = abs(ha - hb) / ref_h + abs(wa - wb) / max(ref_w, 1)
            if size_diff > PAIR_SIZE_TOL:
                continue
            dx = abs(cxa - cxb)
            if not (PAIR_MIN_DIST_FACTOR * ref_h <= dx <= PAIR_MAX_DIST_FACTOR * ref_h):
                continue
            rejected.add(i)
            rejected.add(j)
            break
    return [d for k, d in enumerate(reds) if k not in rejected]


def _expand(d, factor, W, H):
    w = d.x2 - d.x1
    h = d.y2 - d.y1
    dx = max(1, int(round(w * factor)))
    dy = max(1, int(round(h * factor * 1.5)))
    return Detection(
        max(0, d.x1 - dx), max(0, d.y1 - dy),
        min(W, d.x2 + dx), min(H, d.y2 + dy),
        d.label, d.score,
    )


def _iou(a, b):
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aa = (a.x2 - a.x1) * (a.y2 - a.y1)
    ab = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / float(aa + ab - inter)


def _nms(dets, iou_threshold=0.3):
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    keep = []
    for d in dets:
        if all(_iou(d, k) < iou_threshold for k in keep):
            keep.append(d)
    return keep


def detect(frame_bgr, expand=1.0):
    """Detect red/green traffic lights. Returns list[Detection]."""
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    H, W = frame_bgr.shape[:2]
    red   = _clean(_red_mask(hsv))
    green = _clean(_green_mask(hsv))
    reds   = _filter_blobs(red,   hsv, "red",   (H, W))
    greens = _filter_blobs(green, hsv, "green", (H, W))
    reds = _reject_brake_pairs(reds)
    dets = reds + greens
    if expand > 0:
        dets = [_expand(d, expand, W, H) for d in dets]
    return _nms(dets, iou_threshold=0.3)
