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


def make_folder(out_path, specification):
    composed = f"{out_path}/{specification}"
    exists = os.path.exists(composed)
    if not exists:
        os.mkdir(composed)
    return composed


def export_image(folder, filename, image):
    cv2.imwrite(
        f"{folder}/{filename}",
        image
    )


if __name__ == '__main__':
    # Parsing Arguments
    args = parse_arguments()
    in_path = os.path.expanduser(args.input)
    out_path = os.path.expanduser(args.output)

    # Loading Image
    image = cv2.imread(in_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.fastNlMeansDenoising(gray_image, None)

    # Export Grayscale Image
    export_image(
        out_path,
        "grayscale.png",
        gray_image
    )

    # Canny Edge Detection
    folder = make_folder(out_path, "Canny")
    thresholds = range(0, 250 + 10, 10)

    for t in thresholds:
        # Perform Canny Edge Detection
        canny_image = cv2.Canny(gray_image, t, t)

        # Export Image
        export_image(
            folder,
            f"{t}.png",
            canny_image
        )

    # Harris Corner Detection
    folder = make_folder(out_path, "Harris")
    block_size = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    sobel = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    k = 0.001

    # Preparation
    gray_image = np.float32(gray_image)

    for b in block_size:
        for s in sobel:
            # Perform Harris Corner Detection
            corners = cv2.cornerHarris(gray_image, b, s, k)
            corners = cv2.dilate(corners, None)
            marked_image = image.copy()
            marked_image[corners > 0.01 * corners.max()] = [255, 0, 255]

            # Export Image
            export_image(
                folder,
                f"{b}_{s}.png",
                marked_image
            )
