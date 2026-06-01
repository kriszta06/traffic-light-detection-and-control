"""
Description: Entry-point for the simulated Traffic Light Detection & Control application. 
             Manages the main client connection loop with the simulator, extracts camera frames, 
             pipes data through the CV pipeline, and pushes vehicle control commands back to the server.
"""

from glob import glob
import os

import cv2
import sys
from src.preprocessing import apply_gaussian_blur, convert_color_space
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
    image_folder = "data/2011_10_03/2011_10_03_drive_0042_sync/image_02/data"
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")))

    if not image_paths:
        print(f"Error: Could not find any PNG images in path: {image_folder}")
        sys.exit(1)
    
    detector = TrafficLightDetector(model_path='yolov8n.pt') # trained YOLO model 
    tracker = TrafficLightTracker()
    vcu = VehicleControlUnit()

    print("Starting simulation loop. Press 'q' to exit.")

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Could not read image {img_path}. Skipping...")
            continue
        # preprocessing

        blurred_frame = apply_gaussian_blur(frame, kernel_size=(5, 5))
        processed_frame = convert_color_space(blurred_frame, target_space="RGB")

        # detection

        detections = detector.detect(processed_frame, confidence_threshold=0.5)

        # tracking

        tracker.predinct()
        if len(detections) > 0:
            tracker.update(detections[0]['box'])
        
        # control

        target_light = vcu.select_relevant_traffic_light(detections)
        control_payload = vcu.generate_command(target_light)

        # visualization for testing

        for det in detections:
            # draw bounding box and label on the frame
            xmin, xmax, yminn, ymax = map(int, det['box'])
            label = f"{det['class']} ({det['conf']:.2f})"
            cv2.rectangle(frame, (xmin, yminn), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, label, (xmin, yminn - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        action_text = f"ACTION: {control_payload['action']}"
        telemetry_text = f"Throttle: {control_payload['throttle']:.2f}, Brake: {control_payload['brake']:.2f}, Handbrake: {control_payload['handbrake']}, Distance to Light: {control_payload['distance_to_light']:.2f}m"

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

#TODO 1 preprocessing: in apply_gaussian_blur: Apply a standard blur filter (cv2.GaussianBlur) using the given frame, kernel size, 
# and sigma value to clean up pixel noise.

#TODO 2 preprocessing: n convert_color_space: Look at the target_space text. If it says "RGB", 
# convert the frame from BGR to RGB (cv2.COLOR_BGR2RGB). If it says "HSV", convert it to HSV. Return the converted image.

#TODO 3 detection: in __init__: Load the neural network model by passing the model_path string into the YOLO() framework, 
# and save it inside self.model.

#TODO 4 detection: in detect: Pass the image frame to the YOLO model. 
# Loop through all objects found by the AI. If the object class is a traffic light (Class ID 9) and its score is higher than the 
# confidence_threshold, extract its bounding box corners [xmin, ymin, xmax, ymax] and add them to your results list.

#TODO 5 tracking:  in __init__: Set up the internal math for the KalmanFilter to track position and speed ($x, y, vx, vy$). 
# Initialize the state transition matrix ($F$) and measurement matrix ($H$) based on the video frame timing ($dt$).

#TODO 6 tracking: in predict: Run the filter's built-in prediction step (self.kf.predict()) to estimate where the traffic light should be 
# in the current frame. Return the estimated coordinates.

#TODO 7 tracking: in update: Find the middle point ($x, y$) of the bounding box coordinates given by YOLO. 
# Send this point to the filter's update function (self.kf.update()) to correct the tracker's accuracy. Return the corrected coordinates.

#TODO 8 control: in select_relevant_traffic_light:
# Use the image dimensions to define a virtual "Driving Corridor" (the area in the middle of the screen where your lane is). 
# Calculate which traffic light box is closest to this lane center and ignore lights that are on side streets or other lanes.

#TODO 9 control: in generate_command: Look at the size (width and height) of the chosen traffic light's box. 
# As the box gets larger (meaning the car is getting closer to the light in the video), 
# use a mathematical formula to slowly increase the brake value and decrease the throttle value to create a smooth stopping motion on screen.

# DEPENDENCIES: opencv-python, ultralytics, filterpy, and numpy.