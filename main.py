"""
Description: Entry-point for the simulated Traffic Light Detection & Control application. 
             Manages the main client connection loop with the simulator, extracts camera frames, 
             pipes data through the CV pipeline, and pushes vehicle control commands back to the server.
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
    """
    Establishes the simulation client loop, synchronizes sensor callbacks, processes each frame 
    through the pipeline, and updates vehicle actuation at every simulation tick.
    
    Returns:
        None
        
    Note:
        This loop runs synchronously or asynchronously depending on the simulator's configuration 
        (e.g., synchronous mode is preferred in CARLA to ensure no frames are dropped during heavy AI inference).
    """
    
    # Path for testing with a local video file
    image_folder = "data/2011_09_26/2011_09_26_drive_0027_sync/image_02/data"
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")))

    if not image_paths:
        print(f"Error: Could not find any PNG images in path: {image_folder}")
        sys.exit(1)
    
    detector = TrafficLightDetector(model_path='yolov8n.pt')
    tracker = TrafficLightTracker()
    vcu = VehicleControlUnit()

    # User-configurable VCU distances: estimated mapping and stop threshold
    vcu.max_distance = 60.0
    vcu.max_stop_distance = 5.0
    vcu.area_ref = 15000.0
    vcu.stop_area_pixels = 8000.0

    prev_control = None
    smoothing_alpha = 0.20
    stop_distance = vcu.max_stop_distance  # meters — user-configurable stop threshold
    pause_duration = 60.0  # seconds to pause when stopping

    print("Starting simulation loop. Press 'q' to exit.")

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Could not read image {img_path}. Skipping...")
            continue
        # preprocessing

        blurred_frame = apply_gaussian_blur(frame, kernel_size=(5, 5))

        # detection

        # lower confidence to increase sensitivity for small/distant lights
        detections = detector.detect(blurred_frame, confidence_threshold=0.15)
        if not detections:
            print(f"No traffic light detections in frame: {os.path.basename(img_path)}")

        # tracking

        tracker.predict()
        if len(detections) > 0:
            cx, cy, w, h = tracker.update(detections[0]['box'])
            detections[0]['box'] = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
            if 'confirmed' not in detections[0]:
                detections[0]['confirmed'] = True

        # control

        target_light = vcu.select_relevant_traffic_light(detections, lane_info={'image_shape': frame.shape})
        
        if target_light and len(detections) > 0:
            target_light['confirmed'] = True

        control_payload = vcu.generate_command(target_light)

        # apply simple exponential smoothing to throttle/brake so the vehicle slows gradually
        if prev_control is None:
            prev_control = control_payload.copy()
        else:
            th_prev = prev_control.get('throttle', 0.0)
            br_prev = prev_control.get('brake', 0.0)
            th_new = control_payload.get('throttle', 0.0)
            br_new = control_payload.get('brake', 0.0)
            sm_th = th_prev * (1.0 - smoothing_alpha) + th_new * smoothing_alpha
            sm_br = br_prev * (1.0 - smoothing_alpha) + br_new * smoothing_alpha
            control_payload['throttle'] = float(sm_th)
            control_payload['brake'] = float(sm_br)
            prev_control = control_payload.copy()

        # if we need to stop and we're very close to the detected red light, halt and pause
        try:
            close_dist = float(control_payload.get('distance_to_light', float('inf')))
        except Exception:
            close_dist = float('inf')

        is_red_light = target_light and target_light.get('class', '') == 'Red_Light'

        if (control_payload.get('action') == 'STOP' or is_red_light) and close_dist <= stop_distance:
            print("STOP")
            # enforce a full stop in the simulated control payload
            control_payload['action'] = 'STOP'
            control_payload['throttle'] = 0.0
            control_payload['brake'] = 1.0
            # draw STOP, action, and telemetry on the frame so UI shows real stop state
            cv2.putText(frame, "STOP", (frame.shape[1] // 2 - 120, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6)

            action_text = f"ACTION: {control_payload['action']}"
            telemetry_text = (
                f"Throttle: {control_payload['throttle']:.2f},"
                f"Brake: {control_payload['brake']:.2f},"
                f"Handbrake: {control_payload['handbrake']},"
                f"Distance to Light: {control_payload['distance_to_light']:.2f}m"
            )

            cv2.putText(frame, action_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv2.putText(frame, telemetry_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            end_time = time.time() + pause_duration
            while time.time() < end_time:
                cv2.imshow("Traffic Light Detection - Video Simulation", frame)
                # wait a short time to keep GUI responsive; allow user to press 'q' to abort early
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    print("User requested exit during STOP")
                    cv2.destroyAllWindows()
                    return
                    
            print("Exiting simulation loop after STOP")
            break

        # visualization for testing

        for det in detections:
            # draw bounding box and label on the frame
            xmin, ymin, xmax, ymax = map(int, det['box'])
            label = f"{det['class']} ({det['conf']:.2f})"
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, label, (xmin, max(0, ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        action_text = f"ACTION: {control_payload['action']}"
        telemetry_text = (
            f"Throttle: {control_payload['throttle']:.2f},"
            f"Brake: {control_payload['brake']:.2f},"
            f"Handbrake: {control_payload['handbrake']},"
            f"Distance to Light: {control_payload['distance_to_light']:.2f}m"
        )

        color = (0, 0, 255) if "STOP" in action_text else (0, 255, 0)
        cv2.putText(frame, action_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(frame, telemetry_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Traffic Light Detection - Video Simulation", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("Exiting simulation loop.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


