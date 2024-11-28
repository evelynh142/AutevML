'''
Reference Link: https://universe.roboflow.com/joshua-eni4h/charging-ports-fr5zv/model/1

The RoboflowOak library is specifically designed to interface with 
Luxonis OAK (OpenCV AI Kit) devices, which are external cameras that provide advanced computer vision capabilities.
If you have OAK, then you can try the following steps. 
(if not, use the other file to test it on your computer's internal cammer.)

To setup, install the following command in the terminal:
pip install opencv-python
python -m pip install --upgrade pip
python -m pip install depthai
pip install roboflowoak
'''

from roboflowoak import RoboflowOak
import cv2
import time
import numpy as np

if __name__ == '__main__':
    # instantiating an object (rf) with the RoboflowOak module
    rf = RoboflowOak(model="charging-ports-fr5zv", confidence=0.05, overlap=0.5,
    version="1", api_key="N7NPEN2bnowGXUuz7fCa", rgb=True,
    depth=True, device=None, blocking=True)
    # Running our model and displaying the video output with detections
    while True:
        t0 = time.time()
        # The rf.detect() function runs the model inference
        result, frame, raw_frame, depth = rf.detect(visualize=True)
        predictions = result["predictions"]
        #{
        #    predictions:
        #    [ {
        #        x: (middle),
        #        y:(middle),
        #        width:
        #        height:
        #        depth: ###->
        #        confidence:
        #        class:
        #        mask: {
        #    ]
        #}
        #frame - frame after preprocs, with predictions
        #raw_frame - original frame from your OAK
        #depth - depth map for raw_frame, center-rectified to the center camera

        # timing: for benchmarking purposes
        t = time.time()-t0
        print("INFERENCE TIME IN MS ", 1/t)
        print("PREDICTIONS ", [p.json() for p in predictions])

        # setting parameters for depth calculation
        max_depth = np.amax(depth)
        cv2.imshow("depth", depth/max_depth)
        # displaying the video feed as successive frames
        cv2.imshow("frame", frame)

        # how to close the OAK inference window / stop inference: CTRL+q or CTRL+c
        if cv2.waitKey(1) == ord('q'):
            break
