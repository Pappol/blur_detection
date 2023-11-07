import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def get_blur_score(image, treshold=8):
    # check if the image is grayscale, and convert it back to grayscale if necessary
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # use the zero mean image to remove the DC component
    image_zero_mean = image - np.mean(image)

    # compute the fft of the image
    ftt = np.fft.fft2(image_zero_mean)
    ftts = np.fft.fftshift(ftt)

    # remove central frequencies
    rows, cols = image.shape
    size = int(max([rows, cols]) * 0.05)
    high_pass_mask = np.ones(image.shape)
    high_pass_mask[
        rows // 2 - size : rows // 2 + size,
        cols // 2 - size : cols // 2 + size
    ] = 0

    # apply the mask to the fft obtained in the previous section and display the results
    filtered_ftts = ftts * high_pass_mask

    # compute the inverse fft
    iftts = np.fft.ifftshift(filtered_ftts)
    iftt = np.fft.ifft2(iftts)

    # compute the magnitude
    magnitude = np.abs(iftt)

    mean = np.mean(magnitude)

    # compute the mean of the magnitude
    return mean, mean < treshold


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