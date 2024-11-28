'''
This uses internal camera on the computer for test.
ISSUES: Prediction is too slow. The real-time camera lags.
POSSIBLE SOLUTIONS: Enable GPU Acceleration; Adjust Frame Rate

Reference Link: https://universe.roboflow.com/joshua-eni4h/charging-ports-fr5zv/model/1

To setup, install the following command in the terminal:
pip install opencv-python
python -m pip install --upgrade pip
python -m pip install depthai
pip install roboflowoak
'''

import cv2
import requests
import json
import numpy as np

# Roboflow API Key and Endpoint
API_KEY = "N7NPEN2bnowGXUuz7fCa"
MODEL_NAME = "charging-ports-fr5zv"
VERSION = "1"
API_URL = f"https://detect.roboflow.com/{MODEL_NAME}/{VERSION}?api_key={API_KEY}"

def process_frame(frame):
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Encode image as JPEG
    _, img_encoded = cv2.imencode(".jpg", frame_rgb)

    # Send image to Roboflow API
    response = requests.post(API_URL, files={"file": img_encoded.tobytes()})

    # Parse response
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.text)
        return None

if __name__ == '__main__':
    # Initialize internal camera (device 0)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 100)  # Set to 100 FPS or your camera’s max FPS

    if not cap.isOpened():
        print("Error: Could not open internal camera.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Process the frame using Roboflow API
        result = process_frame(frame)

        # Display predictions on the frame
        if result and "predictions" in result:
            for prediction in result["predictions"]:
                x = int(prediction["x"])
                y = int(prediction["y"])
                w = int(prediction["width"])
                h = int(prediction["height"])
                label = prediction["class"]
                confidence = prediction["confidence"]  # Extract confidence value

                # Draw bounding boxes
                cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
                
                # Display label and confidence
                label_text = f"{label} ({confidence:.2f})"
                cv2.putText(frame, label_text, (x - w // 2, y - h // 2 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Internal Camera", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
