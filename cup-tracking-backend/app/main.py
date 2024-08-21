from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.object_detection import ObjectDetector
from app.pose_estimation import PoseEstimator
import cv2
import numpy as np
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict this for security)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FrameData(BaseModel):
    frame: str

detector = ObjectDetector()
pose_estimator = PoseEstimator()

def decode_frame(base64_str):
    # to ensure the base64 string does not include the header
    base64_data = base64_str.split(",")[-1]
    image_data = base64.b64decode(base64_data)
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Could not decode image from base64 string. Ensure the string is valid.")
    
    return image

@app.post("/process_frame/")
async def process_frame(data: FrameData):
    try:
        frame = decode_frame(data.frame)
        results = detector.detect_objects(frame)
        bboxes = detector.get_cup_bbox(results)

        if bboxes:
            # assuming the first detected cup is used for pose estimation
            x_min, y_min, x_max, y_max = bboxes[0][0]
            crop = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
            keypoints = pose_estimator.estimate_pose(crop)
            return {"message": "Detection and Pose Estimation complete", "keypoints": keypoints}
        return {"message": "No cup detected"}
    except Exception as e:
        print("Error decoding frame:", e)
        return {"message": "Error decoding frame", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
