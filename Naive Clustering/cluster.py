#!/usr/bin/env python3
import argparse
import cv2
import os
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--input", type=str, help="Input path")
    parser.add_argument("-o", "--output", type=str, help="Output path")
    return parser.parse_args()


if __name__ == '__main__':
    # Parsing Arguments
    args = parse_arguments()
    in_path = os.path.expanduser(args.input)
    out_path = os.path.expanduser(args.output)

    # Loading Image
    image = cv2.imread(in_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)

    # Parameters
    clusters = range(8, 16 + 1)
    iterations = [5, 10, 15, 20, 40, 60, 80]
    epsilons = np.arange(0.0, 1.0, 0.2)

    for k in clusters:
        # Generate Folder
        folder = f"{out_path}/k = {k}"
        exists = os.path.exists(folder)
        if not exists:
            os.mkdir(folder)

        for i in iterations:
            for e in epsilons:
                # Initialize and Perform k-means
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, i, e)
                _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

                # Convert Output
                centers = np.uint8(centers)
                labels = labels.flatten()

                # Image Composition
                composed_image = centers[labels.flatten()]
                composed_image = composed_image.reshape(image.shape)

                # Export Image
                file = f"{i}_{round(e, 2)}.png"
                cv2.imwrite(
                    f"{folder}/{file}",
                    cv2.cvtColor(composed_image, cv2.COLOR_RGB2BGR)
                )
