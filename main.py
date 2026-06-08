"""
Description: Entry-point for the Traffic Light Detection & Control application.
"""
import glob
import os
import cv2
import sys
import time
from src.preprocessing import apply_gaussian_blur
from src.detection import TrafficLightDetector
from src.tracking import TrafficLightTracker
from src.control import VehicleControlUnit

def main():
    # -------------------------------------------------------------------------
    # DATA SOURCE — set one of the two variables below:
    #   image_folder : folder with .png / .jpg images  (leave "" to use video)
    #   video_file   : path to .mp4 / .avi file        (leave "" to use folder)
    # -------------------------------------------------------------------------
    image_folder = "data/dayTrain/dayTrain/dayClip3/frames"
    video_file   = ""

    image_paths = []
    cap = None

    if image_folder:
        # Use a single glob + extension filter to avoid duplicate paths on
        # Windows (where glob("*.jpg") and glob("*.JPG") both match the same file).
        all_files = glob.glob(os.path.join(image_folder, "*.*"))
        seen = set()
        for p in sorted(all_files):
            norm = os.path.normcase(p)   # lowercase on Windows
            if norm not in seen and p.lower().endswith(('.png', '.jpg', '.jpeg')):
                seen.add(norm)
                image_paths.append(p)

    if not image_paths and not video_file:
        for ext in ("*.mp4", "*.avi", "*.mov"):
            found = glob.glob(os.path.join("data", "**", ext), recursive=True)
            if found:
                video_file = found[0]
                print(f"Auto-detected video: {video_file}")
                break

    if not image_paths and not video_file:
        print(f"Error: no images in '{image_folder}' and no video found.")
        print("Set 'image_folder' or 'video_file' in main.py.")
        sys.exit(1)

    if video_file and not image_paths:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Error: cannot open video '{video_file}'.")
            sys.exit(1)
        print(f"Video mode: {video_file}")

    detector = TrafficLightDetector(model_path='yolov8n.pt')
    tracker  = TrafficLightTracker()
    vcu      = VehicleControlUnit()

    vcu.max_distance      = 60.0
    vcu.max_stop_distance = 5.0
    vcu.area_ref          = 15000.0
    vcu.stop_area_pixels  = 8000.0

    prev_control     = None
    smoothing_alpha  = 0.40   # increased from 0.20 for more responsive transitions
    stop_distance    = vcu.max_stop_distance
    pause_duration   = 2.0

    print(f"Starting simulation: {len(image_paths)} images. Press 'q' to exit.")

    def frame_source():
        if image_paths:
            for img_path in image_paths:
                f = cv2.imread(img_path)
                if f is None:
                    print(f"Warning: cannot read {img_path}. Skipping.")
                    continue
                yield f
        else:
            while True:
                ret, f = cap.read()
                if not ret:
                    break
                yield f
            cap.release()

    for frame in frame_source():
        if frame is None:
            continue

        blurred_frame = apply_gaussian_blur(frame, kernel_size=(5, 5))
        detections    = detector.detect(blurred_frame, confidence_threshold=0.35)

        # Select the relevant light BEFORE tracking so Kalman follows the correct object
        target_light = vcu.select_relevant_traffic_light(detections, lane_info={'image_shape': frame.shape})

        tracker.predict()
        if target_light is not None:
            cx, cy, w, h = tracker.update(target_light['box'])
            target_light['box'] = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

        control_payload = vcu.generate_command(target_light)

        if prev_control is None:
            prev_control = control_payload.copy()
        else:
            th = prev_control.get('throttle', 0.0) * (1 - smoothing_alpha) + control_payload.get('throttle', 0.0) * smoothing_alpha
            br = prev_control.get('brake',    0.0) * (1 - smoothing_alpha) + control_payload.get('brake',    0.0) * smoothing_alpha
            control_payload['throttle'] = float(th)
            control_payload['brake']    = float(br)
            prev_control = control_payload.copy()

        try:
            close_dist = float(control_payload.get('distance_to_light', float('inf')))
        except Exception:
            close_dist = float('inf')

        is_stop = control_payload.get('action') == 'STOP' and close_dist <= stop_distance

        if is_stop:
            print("STOP")
            control_payload['action']   = 'STOP'
            control_payload['throttle'] = 0.0
            control_payload['brake']    = 1.0
            cv2.putText(frame, "STOP",
                        (frame.shape[1]//2 - 120, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6)

        # Draw all detections
        for det in detections:
            xmin, ymin, xmax, ymax = map(int, det['box'])
            label = f"{det['class']} ({det['conf']:.2f})"
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, label, (xmin, max(0, ymin - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        action_text    = f"ACTION: {control_payload['action']}"
        telemetry_text = (f"Throttle: {control_payload['throttle']:.2f}, "
                          f"Brake: {control_payload['brake']:.2f}, "
                          f"Handbrake: {control_payload['handbrake']}, "
                          f"Dist: {control_payload['distance_to_light']:.2f}m")

        color = (0, 0, 255) if "STOP" in action_text else (0, 255, 0)
        cv2.putText(frame, action_text,    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,   color,           3)
        cv2.putText(frame, telemetry_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Traffic Light Detection", frame)

        wait_ms = int(pause_duration * 1000) if is_stop else 25
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            print("Exiting.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
