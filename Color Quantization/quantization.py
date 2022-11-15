#!/usr/bin/env python3
import argparse
import cv2
import os
from PIL import Image


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


if __name__ == '__main__':
    # Parsing Arguments
    args = parse_arguments()
    in_path = os.path.expanduser(args.input)
    out_path = os.path.expanduser(args.output)

    # Export Quantized Images
    palette_bits = range(1, 8 + 1)
    octree_path = make_folder(out_path, "Fast Octree")
    kmeans_path = make_folder(out_path, "k-means")

    for b in palette_bits:
        # Quantize (Fast Octree)
        quantized = Image.open(in_path).quantize(
            colors=(2 ** b),
            method=Image.FASTOCTREE
        )

        # Write
        quantized.save(
            f"{octree_path}/{b}.png",
            format="png",
            append_images=[quantized],
            save_all=True
        )

        # Quantize (Fast Octree)
        quantized = Image.open(in_path).quantize(
            colors=(2 ** b),
            method=None,
            kmeans=(2 ** b)
        )

        # Write
        quantized.save(
            f"{kmeans_path}/{b}.png",
            format="png",
            append_images=[quantized],
            save_all=True
        )
