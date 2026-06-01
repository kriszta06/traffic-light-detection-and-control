# Traffic Light Detection & Control System

A real-time traffic light detection and vehicle control system that processes video files to detect traffic lights, track them across frames, and generate appropriate vehicle control commands. The pipeline is designed to work with video inputs and can be adapted for live camera feeds or simulator integration.

## Project Overview

This system implements an autonomous vehicle's perception and decision-making loop by:
1. Reading frames from a video file
2. Preprocessing images for neural network inference
3. Detecting traffic lights using YOLO deep learning model
4. Tracking detected lights across frames using Kalman filtering
5. Generating vehicle control commands based on detected traffic light states

## System Architecture & Workflow

```
Video Input
    ↓
[Preprocessing] → Gaussian Blur + Color Space Conversion
    ↓
[Detection] → YOLO Neural Network Inference
    ↓
[Tracking] → Kalman Filter Position Estimation
    ↓
[Control] → Generate Vehicle Commands
    ↓
[Visualization] → Display Results + Telemetry
    ↓
Vehicle Actuation Output
```

## Pipeline Components

### 1. **Preprocessing Module** (`src/preprocessing.py`)

Prepares raw camera frames for deep learning inference by reducing noise and normalizing color spaces.

#### Functions:

- **`apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0)`**
  - Applies Gaussian Low Pass filtering to reduce noise and artifacts
  - **TODO 1:** Implement using `cv2.GaussianBlur()` with given kernel size and sigma
  - Helps CNNs generalize better by smoothing edges in video frames
  - Input: Raw BGR/RGB frame from video
  - Output: Smoothed image ready for detection

- **`convert_color_space(image, target_space="RGB")`**
  - Converts frames between color spaces (BGR → RGB/HSV)
  - **TODO 2:** Check target_space parameter and convert using appropriate `cv2.COLOR_*` constants
    - If `target_space == "RGB"`: Convert BGR to RGB using `cv2.COLOR_BGR2RGB`
    - If `target_space == "HSV"`: Convert to HSV using `cv2.COLOR_BGR2HSV`
  - Essential for compatibility with YOLO which expects RGB input
  - Video files typically output BGR; conversion to RGB is required for YOLO

### 2. **Detection Module** (`src/detection.py`)

Performs real-time traffic light detection and classification using YOLO neural network.

#### Class: `TrafficLightDetector`

- **`__init__(model_path='yolov8n.pt')`**
  - **TODO 3:** Load YOLO model using `YOLO(model_path)` from ultralytics framework
  - Initializes the pre-trained neural network weights
  - Parameter: Path to YOLO weights (can be pre-trained or custom-trained)

- **`detect(frame, confidence_threshold=0.5)`**
  - Runs inference on preprocessed frame
  - Returns list of detections with:
    - Bounding box coordinates: `[xmin, ymin, xmax, ymax]`
    - Confidence score: Detection probability (0-1)
    - Class label: `'Red_Light'`, `'Yellow_Light'`, `'Green_Light'`
  - Filters detections below confidence threshold
  - Output: List of dictionaries containing box, class, and confidence information

### 3. **Tracking Module** (`src/tracking.py`)

Maintains temporal consistency of detected traffic lights across frames using Kalman filtering.

#### Class: `TrafficLightTracker`

Uses Kalman Filter to:
- Smooth noisy detection coordinates
- Handle brief detection drops or viewpoint changes
- Maintain stable state across simulation steps
- Prevent control instability from single-frame detection errors

- **`__init__()`**
  - Initializes Kalman Filter matrices
  - Defines state transition, measurement, and covariance parameters
  - Assumes constant velocity model relative to simulation clock

- **`predict()`**
  - Predicts next expected position of traffic light
  - Based on prior state dynamics and temporal consistency
  - Returns: Predicted bounding box `[x, y, w, h]`

- **`update(measurement)`**
  - Corrects estimated state using actual detection from current frame
  - Input: Observed bounding box coordinates from YOLO detector
  - Output: Corrected, filtered bounding box state
  - Minimizes tracking uncertainty through Bayesian inference

### 4. **Control Module** (`src/control.py`)

Translates detected traffic light states into discrete vehicle actuation commands.

#### Class: `VehicleControlUnit`

Maintains vehicle state machine and generates control payloads.

- **`select_relevant_traffic_light(tracked_objects, lane_info=None)`**
  - Filters tracked traffic lights to identify the one affecting current driving path
  - Considers lane information if available
  - Returns: Target traffic light object governing vehicle's active lane
  - Returns: None if no traffic light affects current lane

- **`generate_command(target_light)`**
  - Converts traffic light color state to control signals
  - Input: Target traffic light with current state
  - Output: Control payload containing:
    - `action`: String command ("GO", "STOP", "CAUTION")
    - `throttle`: Acceleration magnitude (0-1)
    - `brake`: Braking magnitude (0-1)
    - `handbrake`: Emergency brake flag
    - `distance_to_light`: Estimated distance to traffic light
  - Logic:
    - **Red/Yellow Light** → Generate brake command payload
    - **Green Light** → Generate acceleration command payload

## Main Execution Loop (`main.py`)

The entry point orchestrates the entire pipeline:

1. **Initialization:**
   - Opens video file (`data/traffic_light_test.mp4`)
   - Instantiates detector, tracker, and control unit
   - Sets up OpenCV display window

2. **Main Loop (per frame):**
   ```
   Frame Read
      ↓
   Preprocessing: apply_gaussian_blur() → convert_color_space()
      ↓
   Detection: detector.detect() → List[{box, class, conf}]
      ↓
   Tracking: tracker.predict() → tracker.update(detections)
      ↓
   Control: select_relevant_light() → generate_command()
      ↓
   Visualization: Draw boxes, labels, action, telemetry
      ↓
   Display & Wait for User Input
   ```

3. **Visualization Output:**
   - Bounding boxes around detected traffic lights
   - Class labels with confidence scores (e.g., "Green_Light (0.95)")
   - Action text in red (STOP) or green (GO)
   - Telemetry display:
     - Current throttle value
     - Current brake value
     - Handbrake state
     - Estimated distance to traffic light

4. **Termination:**
   - Press 'q' to exit gracefully
   - Releases video capture and destroys windows

## Data Flow Visualization

```
Raw Video Frame (BGR)
         ↓
    [Blur Filter]
         ↓
   [Color Space Convert: BGR→RGB]
         ↓
    [YOLO Detection]
    ↓        ↓        ↓
[Red] [Yellow] [Green]
    ↓        ↓        ↓
[Kalman Filter Tracking]
         ↓
[Lane Selection Filter]
         ↓
[Control Command Generation]
         ↓
[Visualization Overlay]
         ↓
Display to User
```

## Dependencies

- **OpenCV** (`cv2`): Video I/O, image processing, visualization
- **YOLO** (`ultralytics`): Deep learning inference
- **FilterPy** (`filterpy.kalman`): Kalman Filter implementation
- **NumPy** (`numpy`): Array operations

## Configuration Parameters

### Preprocessing
- Gaussian blur kernel size: `(5, 5)` (default)
- Target color space: `"RGB"` (standard for YOLO)

### Detection
- Confidence threshold: `0.5` (configurable per frame)
- YOLO model: `yolov8n.pt` (nano model, fast inference)

### Control
- Default vehicle state: `"GO"`

## Usage

```bash
python main.py
```

**Controls:**
- `q` key: Exit simulation loop
- Video will display with real-time detections and vehicle commands

## Implementation TODOs

### TODO 1 - Preprocessing (Gaussian Blur)
**File:** `src/preprocessing.py` → `apply_gaussian_blur()`
- Apply standard blur filter using `cv2.GaussianBlur()`
- Use provided frame, kernel size, and sigma value
- Clean up pixel noise from simulator rendering

### TODO 2 - Preprocessing (Color Space Conversion)
**File:** `src/preprocessing.py` → `convert_color_space()`
- Check `target_space` parameter value
- If `"RGB"`: Convert BGR → RGB using `cv2.COLOR_BGR2RGB`
- If `"HSV"`: Convert BGR → HSV using `cv2.COLOR_BGR2HSV`
- Return converted image

### TODO 3 - Detection (Model Loading)
**File:** `src/detection.py` → `TrafficLightDetector.__init__()`
- Load neural network model using `YOLO(model_path)`
- Initialize detector with ultralytics YOLO framework
- Handle custom-trained models on simulator synthetic data

## Project Status

- ✅ Architecture and pipeline structure
- ✅ Video capture and frame reading
- ✅ Visualization and telemetry display
- ⏳ Preprocessing functions (TODO 1, 2)
- ⏳ Detection model loading (TODO 3)
- ⏳ Kalman filter implementation
- ⏳ Control logic implementation
- ⏳ Lane-based traffic light selection

## Notes

- Processes video files frame-by-frame in synchronous mode
- Can be adapted for live camera feeds or simulator integration
- Kalman filtering provides robustness to temporary detection failures
- Traffic light state classification enables intelligent vehicle control
