import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from common import get_blur_score

if __name__ == "__main__":
    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect blur in the current frame
        mean_magnitude, blurry = get_blur_score(frame)
        
        # Prepare the label for the frame
        text = 'Blurry' if blurry else 'Sharp'
        color = (0, 0, 255) if blurry else (0, 255, 0)
        
        # Put the label on the frame
        cv2.putText(frame, '{}: {:.2f}'.format(text, mean_magnitude), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
        
        # Show the frame
        cv2.imshow('Webcam', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()