import cv2
import mediapipe as mp
import numpy as np
import time
import trimesh
import pyrender
from scipy.spatial.transform import Rotation

mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

objectron = mp_objectron.Objectron(static_image_mode=False,
                                   max_num_objects=5,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.99,
                                   model_name='Cup')

cup_mesh = trimesh.load('cup_obj/cup_green_obj.obj')

scale_factor = 0.01
cup_mesh.apply_scale(scale_factor)

print(f"Cup: {cup_mesh}")

scene = pyrender.Scene()

cup_mesh = pyrender.Mesh.from_trimesh(cup_mesh)
scene.add(cup_mesh)

camera = pyrender.IntrinsicsCamera(fx=1000, fy=1000, cx=320, cy=240)
scene.add(camera)

renderer = pyrender.OffscreenRenderer(640, 480)

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
            # 2D landmarks
            mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)

            # 3D axis
            mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)

            # 3D bounding box
            bbox_3d = np.array([[landmark.x, landmark.y, landmark.z] 
                                for landmark in detected_object.landmarks_3d.landmark])

            # projecting 3D bounding box to 2D
            h, w, _ = image.shape
            bbox_2d = np.int32(bbox_3d[:, :2] * [w, h])

            # drawing 3D bounding box
            cv2.polylines(image, [bbox_2d[:4]], True, (0, 255, 0), 2)
            cv2.polylines(image, [bbox_2d[4:]], True, (0, 255, 0), 2)
            for i in range(4):
                cv2.line(image, tuple(bbox_2d[i]), tuple(bbox_2d[i+4]), (0, 255, 0), 2)

            # get rotation and translation
            rotation_matrix = np.array(detected_object.rotation).reshape(3, 3)
            translation = np.array(detected_object.translation)

            # converting rotation matrix to euler angles
            r = Rotation.from_matrix(rotation_matrix)
            euler_angles = r.as_euler('xyz', degrees=True)

            # display orientation!
            orientation = f"Orientation: X:{euler_angles[0]:.2f}, Y:{euler_angles[1]:.2f}, Z:{euler_angles[2]:.2f}"
            cv2.putText(image, orientation, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # creating a pose for the 3D model
            pose = np.eye(4)
            pose[:3, :3] = rotation_matrix
            pose[:3, 3] = translation

            # adding the posed mesh to the scene
            scene.add(cup_mesh, pose=pose)

            # Render the scene
            color, _ = renderer.render(scene)

            # converting the rendered image to BGR for OpenCV
            color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            # Blend the rendered image with the original image
            mask = (color_bgr != [0, 0, 0]).any(axis=2)
            image[mask] = cv2.addWeighted(image[mask], 0.5, color_bgr[mask], 0.5, 0)

            # removin the posed mesh from the scene for the next iteration
            scene.clear()
            scene.add(camera)
            scene.add(cup_mesh)

    # FPS calculation & display
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(image, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # interface display
    cv2.imshow('3D Cup Detection and AR Overlay', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
renderer.delete()