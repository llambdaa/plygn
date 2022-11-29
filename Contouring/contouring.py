#!/usr/bin/env python3
import argparse
import cv2
import os
import datetime as date

from coloring import *
from colorspace import *
from contour import *
from palette import *
from triangulation import *
from utils import *


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to input image")
    parser.add_argument("-o", "--output", required=True, type=str, help="Path to output image")
    parser.add_argument("-c", "--colorspace", required=False, type=ColorSpace, choices=list(ColorSpace),
                        default=ColorSpace.RGB, help="Clustering color space")
    parser.add_argument("-v", "--vertex-method", required=False, type=VertexMethod, choices=list(VertexMethod),
                        default=VertexMethod.EQUAL_SPACE, help="Vertex placement method")
    parser.add_argument("-e", "--equal-distance", required=False, default=10,
                        help="Distance parameter for EQUAL_SPACE vertex placement method")
    parser.add_argument("-s", "--split-threshold", required=False, default=-1,
                        help="Triangle size threshold, above which it is split into smaller triangles")
    parser.add_argument("-b", "--bitmask-kernel", required=False, default=5,
                        help="Bitmask kernel size for morphological transformation")
    parser.add_argument("-d", "--dominant-count", required=False, default=8, help="Count of dominant colors")
    parser.add_argument("-p", "--show-plot", required=False, action='store_true',
                        help="Flag for plotting image in color space")
    parser.add_argument("-C", "--show-contour", required=False, action='store_true',
                        help="Flag for exporting images of contours")
    parser.add_argument("-T", "--show-triangulation", required=False, action='store_true',
                        help="Flag for exporting triangulation of image")
    return parser.parse_args()


if __name__ == '__main__':
    # Parsing command line arguments
    args = parse_arguments()
    in_path = os.path.expanduser(args.input)
    out_path = os.path.expanduser(args.output)
    colorspace = args.colorspace
    vertex_method = args.vertex_method
    equal_distance = int(args.equal_distance)
    split_threshold = int(args.split_threshold)
    bitmask_kernel = int(args.bitmask_kernel)
    dominant_count = int(args.dominant_count)
    flag_plot = args.show_plot
    flag_contours = args.show_contour
    flag_triangulation = args.show_triangulation

    # Load image
    image_name = os.path.basename(in_path).split('.', 1)[0]
    image = cv2.imread(in_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Transform image data to color space
    start = time()
    now = start
    print(f"1. Color Space Transformation", end='\r')
    image_as_ints, unique_ints, unique_colors, unique_counts = dedupe_colors(image)
    translated_unique_colors = to_space(unique_colors, colorspace)
    print(f"1. Color Space Transformation \t{(time() - now).total_seconds()}s")

    if flag_plot is True:
        plot(unique_colors, translated_unique_colors)

    # Perform k-means clustering
    # (=> Palette Reduction)
    now = time()
    print(f"2. Color Clustering", end='\r')
    labels = kmeans(dominant_count, translated_unique_colors, unique_counts)
    labels = expand_labels(image_as_ints, unique_ints, labels, image.shape)
    print(f"2. Color Clustering \t\t{(time() - now).total_seconds()}s")

    # Contouring
    now = time()
    print(f"3. Contouring", end='\r')
    contours = find_contours(image, dominant_count, labels, bitmask_kernel)
    print(f"3. Contouring \t\t\t{(time() - now).total_seconds()}s")

    # Triangulation
    now = time()
    print(f"4. Vertex Search", end='\r')
    match vertex_method:
        case VertexMethod.EQUAL_SPACE:
            vertices = find_vertices_equal_space(contours, equal_distance)
        case _:
            vertices = list()
    print(f"4. Vertex Search \t\t{(time() - now).total_seconds()}s")

    if flag_contours is True:
        alpha = make_folder(out_path, image_name)
        gamma = make_folder(alpha, colorspace)
        show_contours(image, contours, gamma)

    now = time()
    print(f"5. Triangulation", end='\r')
    triangulation = find_triangulation(image.shape, vertices)
    if split_threshold > 0:
        triangulation = split_triangulation(triangulation, split_threshold)
    print(f"5. Triangulation \t\t{(time() - now).total_seconds()}s")

    if flag_triangulation is True:
        alpha = make_folder(out_path, image_name)
        gamma = make_folder(alpha, colorspace)
        show_triangulation(image, triangulation, gamma)

    # Coloring
    now = time()
    print(f"6. Colorization", end='\r')
    result = colorize(image, triangulation)
    end = time()
    print(f"6. Colorization \t\t{(end - now).total_seconds()}s")
    print(45 * "-")
    print(f"Total Elapsed: \t\t\t{(end - start).total_seconds()}s")

    cv2.imwrite(
        f"{out_path}/finished.jpg",
        cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    )
