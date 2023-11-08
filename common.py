
import cv2
import numpy as np

def get_blur_score(image, treshold=5):
    # check if the image is grayscale, and convert it back to grayscale if necessary
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # use the zero mean image to remove the DC component
    image = image - np.mean(image)

    # compute the fft of the image
    ftt = np.fft.fft2(image)
    ftts = np.fft.fftshift(ftt)

    # remove central frequencies
    rows, cols = image.shape
    size_r = int(rows * 0.25)
    size_c = int(cols * 0.25)
    high_pass_mask = np.ones(image.shape)
    high_pass_mask[
        rows // 2 - size_r : rows // 2 + size_r,
        cols // 2 - size_c : cols // 2 + size_c
    ] = 0

    # apply the mask to the fft obtained in the previous section and display the results
    magnitude_spectrum = np.abs(ftts)
    filtered_magnitude = magnitude_spectrum * high_pass_mask

    # compute the magnitude
    magnitude = np.log(1 + filtered_magnitude)

    mean = np.mean(magnitude)

    # compute the mean of the magnitude
    return mean, mean < treshold

