#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 12:33:49 2025

@author: mohsen
"""

import cv2
import numpy as np
import csv
import os


# Camera parameters (adjust based on your setup)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FOV_X = 69  # Horizontal field of view in degrees
FOV_Y = 55  # Vertical field of view in degrees
Z_DISTANCE = 1.0  # Assumed working distance to marker plane (meters)

# ArUco parameters
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
ARUCO_SIZE = 0.05  # Physical size of ArUco marker in meters

CSV_FILE = "pose_data.csv"

class PoseCalculator:
    def __init__(self):
        self.reference_set = False
        self.ref_pose = None
        self.last_pose = None  # Store the last valid pose
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)


        # Calculate pixel-to-meter conversions
        self.x_scale = (2 * Z_DISTANCE * np.tan(np.radians(FOV_X/2))) / IMAGE_WIDTH
        self.y_scale = (2 * Z_DISTANCE * np.tan(np.radians(FOV_Y/2))) / IMAGE_HEIGHT
        
        # Create and initialize CSV file
        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Delta X (m)", "Delta Y (m)", "Delta Ψ (degrees)"])

    def calculate_pose(self, corners):
        """Calculate position and orientation from ArUco marker corners"""
        if len(corners) == 0:
            return None  # No markers detected
        
        corners = np.array(corners, dtype=np.float32)  # Convert to NumPy array
        
        if corners.shape[1] < 4:  # If not all four corners are detected
            return None  # Wait until all four corners appear

        center = np.mean(corners[0], axis=0)  # Compute mean position

        x = (center[1] - IMAGE_HEIGHT / 2) * self.y_scale
        y = (center[0] - IMAGE_WIDTH / 2) * self.x_scale

        # Calculate orientation using the first two corners
        vector = corners[0][1] - corners[0][0]
        psi = np.arctan2(vector[1], vector[0])

        return np.array([x, y, psi])

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, self.last_pose  # Return last valid pose
    
        # Convert to grayscale and detect markers
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT)
    
        if ids is not None and len(corners) > 0:
            for i in range(len(ids)):  # Loop through all detected markers
                cv2.polylines(frame, [np.int32(corners[i])], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # Get the center of the marker
                center = np.mean(corners[i], axis=1).astype(int)[0]
                
                # Put the ID of the marker on the image
                cv2.putText(frame, f"ID: {ids[i][0]}", (center[0] - 10, center[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
            pose = self.calculate_pose(corners)
    
            if pose is not None:  # Update only if a full marker is detected
                self.last_pose = pose  # Save last valid pose
    
                if not self.reference_set:
                    self.ref_pose = pose
                    self.reference_set = True
                    print("Reference pose set:", self.ref_pose)
                    return frame, (0, 0, 0)
                else:
                    delta = pose - self.ref_pose
                    self.save_to_csv(delta)
                    return frame, delta
    
        return frame, self.last_pose  # If incomplete detection, return last valid pose
    
    def save_to_csv(self, data):
        """Save (x, y, psi) to CSV file"""
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"{data[0]:.3f}", f"{data[1]:.3f}", f"{np.degrees(data[2]):.1f}"])


    def run(self):
        try:
            while True:
                frame, result = self.process_frame()
                if frame is None:
                    break

                if result is not None:
                    x, y, psi = result
                    print(f"Delta X: {x:.3f}m, Delta Y: {y:.3f}m, Delta Ψ: {np.degrees(psi):.1f}°")
                
                cv2.imshow('Pose Estimation', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    calculator = PoseCalculator()
    calculator.run()
