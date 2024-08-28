import cv2
import mediapipe as mp
import numpy as np
import time

mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

objectron = mp_objectron.Objectron(static_image_mode=False,
                                   max_num_objects=5,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.99,
                                   model_name='Cup')

prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to read frame from camera.")
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = objectron.process(image_rgb)

    if results.detected_objects:
        for detected_object in results.detected_objects:
            mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
            
            mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)
            
            bbox_3d = np.array([[landmark.x, landmark.y, landmark.z] 
                                for landmark in detected_object.landmarks_3d.landmark])
            
            h, w, _ = image.shape
            bbox_2d = np.int32(bbox_3d[:, :2] * [w, h])
            
            cv2.polylines(image, [bbox_2d[:4]], True, (0, 255, 0), 2)
            cv2.polylines(image, [bbox_2d[4:]], True, (0, 255, 0), 2)
            for i in range(4):
                cv2.line(image, tuple(bbox_2d[i]), tuple(bbox_2d[i+4]), (0, 255, 0), 2)
            
            rotation_matrix = np.array(detected_object.rotation).reshape(3, 3)
            euler_angles = cv2.Rodrigues(rotation_matrix)[0].flatten()
            orientation = f"Orientation: X:{euler_angles[0]:.2f}, Y:{euler_angles[1]:.2f}, Z:{euler_angles[2]:.2f}"
            cv2.putText(image, orientation, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(image, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('3D Cup Detection and Tracking', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()