import cv2
import numpy as np
import csv
import time


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FOV_X = 30  # Horizontal field of view (degrees)
FOV_Y = 50  # Vertical field of view (degrees)
Z_DISTANCE = 0.1

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
ARUCO_SIZE = 0.05


x_scale = (2 * Z_DISTANCE * np.tan(np.radians(FOV_X/2))) / IMAGE_WIDTH
y_scale = (2 * Z_DISTANCE * np.tan(np.radians(FOV_Y/2))) / IMAGE_HEIGHT


reference_frame = False
ref_pose = []
last_pose = []

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

csv_file = open("pose_angle_data.csv", mode="a", newline="")
csv_writer = csv.writer(csv_file)

if csv_file.tell() == 0:
    csv_writer.writerow(["Timestamp", "X (m)", "Y (m)", "Psi (deg)"])
    
start_time = time.time()
try:
    while True:
        # global reference_set, ref_pose, last_pose
        
        ret, frame = cap.read()
        if not ret:
            last_pose
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT)
        # if ids is not None:
        #     print(corners)
        
        if ids is not None and len(ids) > 0:
            i = 0  
            elapsed_time = time.time() - start_time 
            cv2.polylines(frame, [np.int32(corners[i])], True, (0, 255, 0), 2)
            center = np.mean(corners[i][0], axis=0).astype(int)
            cv2.putText(frame, "Honey Bee", (center[0] - 40, center[1] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
            
            marker_corners = np.array(corners[0][0], dtype=np.float32)  

            # print(marker_corners)
            center_x = np.sum(marker_corners[:, 0])/len(marker_corners[:,0])
            center_y = np.sum(marker_corners[:, 1])/len(marker_corners[:,1])
            
            x = (center_x - IMAGE_WIDTH / 2) * x_scale  
            y = (IMAGE_HEIGHT / 2 - center_y) * y_scale  
                
            text = f"X: {x:.3f}m, Y: {y:.3f}m"
            cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 1)
            
            vector = marker_corners[1] - marker_corners[0]
            psi = np.arctan2(vector[1], vector[0])  
            psi_deg = np.degrees(psi)  
            
            cv2.putText(frame, f"Angle: {psi_deg:.1f} deg", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 1)
            csv_writer.writerow([elapsed_time, x, y, psi_deg])

        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()






        

