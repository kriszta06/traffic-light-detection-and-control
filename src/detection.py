"""
Description: Traffic light detection and color classification using YOLO + HSV fallback.
"""

import cv2
import numpy as np
from ultralytics import YOLO


class TrafficLightDetector:

    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.traffic_light_class_id = 9
        self.area_ref = 50000.0
        self.stop_area_pixels = 20000.0

    def detect(self, frame, confidence_threshold=0.5):
        if frame is None:
            return []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(rgb_frame, imgsz=640, conf=confidence_threshold)
        detections = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            xyxy        = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids   = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(xyxy, confidences, class_ids):
                if cls_id != self.traffic_light_class_id or conf < confidence_threshold:
                    continue
                xmin, ymin, xmax, ymax = map(float, box)
                box_w    = xmax - xmin
                box_h    = ymax - ymin
                box_area = box_w * box_h

                # Shape validation: reject boxes that don't look like a traffic light housing.
                # A real housing (3 stacked bulbs) is taller than wide (h/w >= 0.8).
                # Street lamp heads and glare spots are square or wider than tall.
                # Also reject tiny boxes < 50px (too small to classify reliably).
                if box_area < 50.0:
                    continue
                if box_w > 0 and (box_h / box_w) < 0.8:
                    continue

                label = self._classify_light_color(frame, xmin, ymin, xmax, ymax)
                detections.append({
                    'box':   [xmin, ymin, xmax, ymax],
                    'conf':  float(conf),
                    'class': label,
                })

        if not detections:
            detections = self._color_fallback(frame)
        else:
            detections = self._apply_nms(detections, iou_threshold=0.45)

        return detections

    @staticmethod
    def _apply_nms(detections, iou_threshold=0.45):
        if len(detections) <= 1:
            return detections
        detections_sorted = sorted(detections, key=lambda d: d['conf'], reverse=True)
        kept = []
        for det in detections_sorted:
            xmin, ymin, xmax, ymax = det['box']
            dominated = False
            for k in kept:
                kx1, ky1, kx2, ky2 = k['box']
                ix1 = max(xmin, kx1)
                iy1 = max(ymin, ky1)
                ix2 = min(xmax, kx2)
                iy2 = min(ymax, ky2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = (max(0.0, xmax - xmin) * max(0.0, ymax - ymin) +
                         max(0.0, kx2 - kx1)  * max(0.0, ky2 - ky1) - inter)
                if inter / max(union, 1e-6) > iou_threshold:
                    dominated = True
                    break
            if not dominated:
                kept.append(det)
        return kept

    def _classify_light_color(self, frame, xmin, ymin, xmax, ymax):
        xmin = max(0, int(round(xmin)))
        ymin = max(0, int(round(ymin)))
        xmax = min(frame.shape[1] - 1, int(round(xmax)))
        ymax = min(frame.shape[0] - 1, int(round(ymax)))
        if xmax <= xmin or ymax <= ymin:
            return 'Traffic_Light'

        crop = frame[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            return 'Traffic_Light'

        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_ch = hsv[:, :, 0].astype(float)
        s_ch = hsv[:, :, 1].astype(float)
        v_ch = hsv[:, :, 2].astype(float)

        # Select pixels that are BOTH bright AND saturated (high V*S score).
        # This avoids sky (high V, low S) and dark housing (low V).
        vs_score = v_ch * s_ch
        if vs_score.max() < 1.0:
            return 'Traffic_Light'

        threshold = np.percentile(vs_score, 80)
        best_mask = vs_score >= max(threshold, 500.0)

        if best_mask.sum() < 4:
            return 'Traffic_Light'

        best_h = h_ch[best_mask]
        best_s = s_ch[best_mask]

        if float(best_s.mean()) < 40:
            return 'Traffic_Light'

        # Vote by counting pixels in each color band.
        # DO NOT use mean_h for red: red wraps around hue axis (0-15 AND 155-179).
        # mean([5, 175]) = 90 -> misclassified as green without voting.
        n_red    = int(np.sum((best_h <= 15) | (best_h >= 155)))
        n_yellow = int(np.sum((best_h > 15)  & (best_h <= 45)))
        n_green  = int(np.sum((best_h > 45)  & (best_h <= 100)))

        best_count = max(n_red, n_yellow, n_green)
        if best_count < 3:
            return 'Traffic_Light'

        if best_count == n_red:
            return 'Red_Light'
        if best_count == n_yellow:
            return 'Yellow_Light'
        return 'Green_Light'

    def _color_fallback(self, frame):
        """HSV-based fallback. Only fires when YOLO finds no traffic lights.

        Strict filters to avoid false positives (road signs, street lamps):
        1. ROI: top 45% of image only (traffic lights are mounted high)
        2. High S+V thresholds (vivid, bright pixels only)
        3. Minimum area 50px
        4. Aspect ratio 0.7 - 3.5
        5. Pixel density >= 40% inside bbox
        6. Circularity >= 0.45 (bulb is round; signs are not)
        7. Dark surround: immediate border pixels must be dark (black housing)
        """
        height, width = frame.shape[:2]
        roi_y_end   = int(height * 0.45)
        roi_x_start = int(width  * 0.15)
        roi_x_end   = int(width  * 0.85)

        roi = frame[0:roi_y_end, roi_x_start:roi_x_end]
        if roi.size == 0:
            return []
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        color_ranges = {
            'Red_Light': [
                (np.array([0,   160, 160]), np.array([10,  255, 255])),
                (np.array([165, 160, 160]), np.array([179, 255, 255])),
            ],
            'Yellow_Light': [
                (np.array([18, 160, 160]), np.array([35, 255, 255])),
            ],
            'Green_Light': [
                (np.array([42, 160, 160]), np.array([88, 255, 255])),
            ],
        }

        all_detections = []

        for label, ranges in color_ranges.items():
            mask = None
            for lower, upper in ranges:
                m    = cv2.inRange(hsv, lower, upper)
                mask = m if mask is None else cv2.bitwise_or(mask, m)

            if mask is None:
                continue

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask   = cv2.dilate(mask, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = float(w * h)

                if area < 50.0 or area > 3000.0:
                    continue

                aspect_ratio = h / max(w, 1)
                if aspect_ratio < 0.7 or aspect_ratio > 3.5:
                    continue

                colored_pixels = int(cv2.countNonZero(mask[y:y + h, x:x + w]))
                if colored_pixels / max(area, 1.0) < 0.40:
                    continue

                contour_area = cv2.contourArea(contour)
                perimeter    = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4.0 * np.pi * contour_area / (perimeter ** 2)
                    if circularity < 0.45:
                        continue

                border = 4
                sx = max(0, x - border)
                sy = max(0, y - border)
                ex = min(roi.shape[1], x + w + border)
                ey = min(roi.shape[0], y + h + border)
                surround_v = hsv[sy:ey, sx:ex, 2].astype(float)
                inner      = np.zeros(surround_v.shape, dtype=bool)
                inner[y - sy:y - sy + h, x - sx:x - sx + w] = True
                outer = surround_v[~inner]
                if outer.size > 0 and float(outer.mean()) > 120:
                    continue

                all_detections.append({
                    'box':   [float(roi_x_start + x), float(y),
                              float(roi_x_start + x + w), float(y + h)],
                    'conf':  0.30,
                    'class': label,
                })

        return all_detections
