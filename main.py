import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from common import get_blur_score

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--treshold', type=int, default=8)
    parser.add_argument('--path', type=str, default=None)

    args = parser.parse_args()

    # Set the treshold
    treshold = args.treshold
    path = args.path

    if path is None:
        print('Please specify a path to an image with --path <path>') 
        exit()

    frame = cv2.imread(path)

    # Detect blur in the current frame
    mean_magnitude, blurry = get_blur_score(frame)
    
    # Prepare the label for the frame
    text = 'blurry' if blurry else 'sharp'

    print('The image is {} with a blur score of {:.2f}'.format(text, mean_magnitude))
