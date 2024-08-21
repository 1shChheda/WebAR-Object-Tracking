import mediapipe as mp
import cv2

class PoseEstimator:
    def __init__(self):
        self.mp_objectron = mp.solutions.objectron
        self.objectron = self.mp_objectron.Objectron(static_image_mode=False,
                                                     max_num_objects=1,
                                                     min_detection_confidence=0.5,
                                                     min_tracking_confidence=0.5,
                                                     model_name='Cup')  # Use "Cup" model

    def estimate_pose(self, image):
        # Convert the image to RGB as Objectron expects RGB input
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.objectron.process(rgb_image)
        keypoints = []

        if results.detected_objects:
            for detected_object in results.detected_objects:
                keypoints = detected_object.keypoints  # extracting keypoints (buggy, needs to be fixed)

        return keypoints
