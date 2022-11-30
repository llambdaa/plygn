#!/usr/bin/env python3
import argparse
import cv2
import os
import rawpy

from utils import *
from colorspace import *
from clustering import *
from contouring import *
from triangulation import *
from colorization import *

PROCESS_STEP = 1


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to input image")
    parser.add_argument("-o", "--output", required=True, type=str, help="Path to output image")
    parser.add_argument("-c", "--colorspace", required=False, type=ColorSpace, choices=list(ColorSpace),
                        default=ColorSpace.RGB, help="Clustering color space")
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


def process(description, function, *argv):
    global PROCESS_STEP
    now = time()

    print(f"{PROCESS_STEP}. {description}", end='\r')
    (result) = function(argv)
    print(f"{PROCESS_STEP}. {description} \t\t{(time() - now).total_seconds()}s")
    PROCESS_STEP += 1
    return result


def load_image(argv):
    in_path = argv[0]
    image_name = os.path.basename(in_path).split('.', 1)[0]
    image = rawpy.imread(in_path).postprocess()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, image_name


def transform_colorspace(argv):
    image, colorspace = argv
    image_as_ints, unique_ints, unique_colors, unique_counts = dedupe_colors(image)
    translated_unique_colors = to_space(unique_colors, colorspace)
    return image_as_ints, unique_ints, unique_colors, unique_counts, translated_unique_colors


def group_by_color(argv):
    dominant_count, translated_unique_colors, unique_counts, \
    image_as_ints, unique_ints, shape = argv
    labels = kmeans(dominant_count, translated_unique_colors, unique_counts)
    labels = expand_labels(image_as_ints, unique_ints, labels, shape)
    return labels


def contouring(argv):
    image, dominant_count, labels, bitmask_kernel = argv
    contours = find_contours(image, dominant_count, labels, bitmask_kernel)
    return contours


def search_vertices(argv):
    contours, equal_distance = argv
    vertices = find_vertices(contours, equal_distance)
    return vertices


def write_contours(argv):
    out_path, image, image_name, colorspace = argv
    alpha = make_folder(out_path, image_name)
    gamma = make_folder(alpha, colorspace)
    show_contours(image, contours, gamma)


def triangulate(argv):
    shape, vertices = argv
    triangulation = find_triangulation(shape, vertices)
    return triangulation


def triangle_splitting(argv):
    triangulation, split_threshold = argv
    split = split_triangulation(triangulation, split_threshold)
    return split


def write_triangulation(argv):
    out_path, image, image_name, colorspace = argv
    alpha = make_folder(out_path, image_name)
    gamma = make_folder(alpha, colorspace)
    show_triangulation(image, triangulation, gamma)


def colorize_triangles(argv):
    image, triangulation = argv
    result = colorize(image, triangulation)
    return result


if __name__ == '__main__':
    # Parsing Arguments
    args = parse_arguments()
    in_path = os.path.expanduser(args.input)
    out_path = os.path.expanduser(args.output)
    colorspace = args.colorspace
    equal_distance = int(args.equal_distance)
    split_threshold = int(args.split_threshold)
    bitmask_kernel = int(args.bitmask_kernel)
    dominant_count = int(args.dominant_count)
    flag_plot = args.show_plot
    flag_contours = args.show_contour
    flag_triangulation = args.show_triangulation

    # Loading Image
    image, image_name = load_image(in_path)

    # Processing
    start = time()
    image_as_ints, unique_ints, \
    unique_colors, unique_counts, \
    translated_unique_colors = process("Color Space Transformation", transform_colorspace,
                                       image, colorspace)

    if flag_plot is True:
        process("Plotting", plot, unique_colors, translated_unique_colors)

    labels = process("Color Clustering", group_by_color, dominant_count,
                     translated_unique_colors, unique_counts,
                     image_as_ints, unique_ints, image.shape)

    contours = process("Contouring", contouring, image,
                       dominant_count, labels, bitmask_kernel)

    vertices = process("Vertex Search", search_vertices, contours, equal_distance)

    if flag_contours is True:
        process("Writing Contours", write_contours, out_path, image, image_name, colorspace)

    triangulation = process("Triangulation", triangulate, image.shape, vertices)

    if split_threshold > 0:
        triangulation = process("Triangle Splitting", triangle_splitting, triangulation, split_threshold)

    if flag_triangulation is True:
        process("Writing Triangulation", write_triangulation, out_path, image, image_name, colorspace)

    colorized_image = process("Triangle Colorization", colorize_triangles, image, triangulation)
    end = time()

    print(45 * "-")
    print(f"Total Time: \t\t\t{(end - start).total_seconds()}s")

    # Writing Out
    cv2.imwrite(
        f"{out_path}/{image_name}1.jpg",
        cv2.cvtColor(colorized_image, cv2.COLOR_RGB2BGR)
    )

    cv2.imwrite(
        f"{out_path}/{image_name}2.jpg",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )
