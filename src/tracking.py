"""
Description: Manages temporal tracking of detected traffic lights across simulation steps using a Kalman Filter 
             to smooth out coordinate noise and maintain state consistency.
"""

import numpy as np
from filterpy.kalman import KalmanFilter

class TrafficLightTracker:
    """
    A Kalman Filter-based tracker designed to predict and correct the positions of traffic lights over time, 
    preventing control instability caused by occasional single-frame detection drops.
    """
    
    def __init__(self, dt=1.0):
        """
        Initializes the Kalman Filter matrices, defining the state transition, measurement, and covariance parameters 
        under a constant velocity assumption relative to the simulation clock.
        """
        self.dt = dt
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)
        self.kf.P = np.eye(4) * 500.0
        self.kf.R = np.eye(2) * 10.0
        self.kf.Q = np.eye(4) * 1e-2
        self.kf.x = np.zeros((4, 1), dtype=float)
        self.last_bbox = [0.0, 0.0, 0.0, 0.0]
        self.has_measurement = False
        
    def predict(self):
        """
        Predicts the next expected position of the traffic light bounding box based on prior state dynamics 
        and the ego-vehicle's motion telemetry provided by the simulator.
        
        Returns:
            tuple: Predicted bounding box coordinates [x, y, w, h].
        """
        self.kf.predict()
        center_x = float(self.kf.x[0, 0])
        center_y = float(self.kf.x[1, 0])
        width = self.last_bbox[2] - self.last_bbox[0]
        height = self.last_bbox[3] - self.last_bbox[1]
        return (center_x, center_y, float(width), float(height))
        
    def update(self, measurement):
        """
        Corrects the estimated state and minimizes tracking uncertainty using the actual bounding box observation 
        provided by the YOLO detector in the current frame.
        
        Parameters:
            measurement (list): Observed bounding box coordinates from the detection module.
            
        Returns:
            tuple: The corrected, filtered bounding box state.
            
        Note:
            In a simulated environment, tracking helps handle sudden viewpoint changes caused by sharp turns 
            or aggressive braking loops.
        """
        xmin, ymin, xmax, ymax = measurement
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin
        self.last_bbox = [xmin, ymin, xmax, ymax]
        measurement_vector = np.array([center_x, center_y], dtype=float)
        self.kf.update(measurement_vector)
        corrected_x = float(self.kf.x[0, 0])
        corrected_y = float(self.kf.x[1, 0])
        return (corrected_x, corrected_y, float(width), float(height))