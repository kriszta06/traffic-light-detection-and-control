"""
Description: Handles real-time traffic light detection and state classification within the simulated environment 
             using a Single-Stage CNN detector (YOLO framework).
"""

import cv2
import numpy as np
from ultralytics import YOLO

class TrafficLightDetector:
    """
    A class responsible for loading the trained CNN weights and performing inference on simulated camera streams.
    """
    
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initializes the detector by loading the network architecture and weights.
        
        Parameters:
            model_path (str): Path to the YOLO weights file (e.g., custom trained on simulator synthetic data).
        """
        self.model = YOLO(model_path)
        self.traffic_light_class_id = 9
        self.color_ranges = {
            'Red_Light': [(0, 70, 50), (10, 255, 255), (170, 255, 255)],
            'Yellow_Light': [(15, 70, 50), (35, 255, 255)],
            'Green_Light': [(36, 70, 50), (85, 255, 255)],
        }
        
    def detect(self, frame, confidence_threshold=0.5):
        """
        Runs neural network inference on the current simulation frame to detect and classify traffic lights.
        
        Parameters:
            frame (numpy.ndarray): The preprocessed image frame from the simulator.
            confidence_threshold (float): Minimum confidence score to validate a detection.
            
        Returns:
            list of dict: A list of detected bounding boxes, where each box contains coordinates 
                          [xmin, ymin, xmax, ymax], confidence score, and the predicted class label 
                          (e.g., 'Red_Light', 'Yellow_Light', 'Green_Light').
                          
        Note:
            Bounding boxes are axis-aligned relative to the simulator camera's current viewport resolution.
        """
        if frame is None:
            return []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(rgb_frame, imgsz=1280, conf=confidence_threshold)
        detections = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(xyxy, confidences, class_ids):
                if cls_id != self.traffic_light_class_id or conf < confidence_threshold:
                    continue
                xmin, ymin, xmax, ymax = map(float, box)
                label = self._classify_light_color(frame, xmin, ymin, xmax, ymax)
                detections.append({
                    'box': [xmin, ymin, xmax, ymax],
                    'conf': float(conf),
                    'class': label,
                })

        if not detections:
            detections = self._color_fallback(frame)

        return detections

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

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_h = float(hsv[:, :, 0].mean())
        mean_s = float(hsv[:, :, 1].mean())
        mean_v = float(hsv[:, :, 2].mean())

        # Relax saturation/value checks for distant/dim lights
        if mean_s < 30 or mean_v < 40:
            return 'Traffic_Light'
        # broaden hue ranges slightly to be more tolerant
        if mean_h <= 15 or mean_h >= 150:
            return 'Red_Light'
        if 15 < mean_h <= 40:
            return 'Yellow_Light'
        if 40 < mean_h <= 90:
            return 'Green_Light'
        return 'Traffic_Light'

    def _color_fallback(self, frame):
        """Fallback traffic light detection for distant or low-confidence lights."""
        height, width = frame.shape[:2]
        # expand ROI to cover more of the frame where distant lights appear (top + center)
        roi = frame[0:int(height * 0.6), 0:width]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        color_ranges = {
            # lower sat/val lower-bounds to be more sensitive to dim/distant lights
            'Red_Light': [
                (np.array([0, 60, 60]), np.array([18, 255, 255])),
                (np.array([150, 60, 60]), np.array([179, 255, 255])),
            ],
            'Yellow_Light': [
                (np.array([12, 80, 80]), np.array([40, 255, 255])),
            ],
            'Green_Light': [
                (np.array([36, 80, 80]), np.array([95, 255, 255])),
            ],
        }

        best_detection = None
        best_area = 0
        for label, ranges in color_ranges.items():
            mask = None
            for lower, upper in ranges:
                current_mask = cv2.inRange(hsv, lower, upper)
                mask = current_mask if mask is None else cv2.bitwise_or(mask, current_mask)

            # morphological cleaning to reduce noise and merge nearby small regions
            if mask is None:
                continue
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)

            # allow detection of very small blobs by also clustering non-zero points
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                # try to find sparse points and cluster them
                pts = cv2.findNonZero(mask)
                if pts is not None and len(pts) > 3:
                    x, y, w, h = cv2.boundingRect(pts)
                    contours = [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])]
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = float(w * h)
                # accept very small blobs for fallback (distant lights)
                if area <= 4.0:
                    continue
                if area > best_area:
                    best_area = area
                    best_detection = (x, y, w, h, label)

        if best_detection is None:
            return []

        x, y, w, h, label = best_detection
        xmin = int(width * 0.2) + x
        ymin = y
        xmax = xmin + w
        ymax = ymin + h
        return [{
            'box': [float(xmin), float(ymin), float(xmax), float(ymax)],
            'conf': 0.35,
            'class': label,
        }]
