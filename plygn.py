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
from export import *

PROCESS_STEP = 1


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--input",
                        required=True,
                        type=str,
                        help="Path to input image")
    parser.add_argument("-o", "--output",
                        required=True,
                        type=str,
                        help="Path to output image")
    parser.add_argument("-c", "--colorspace",
                        required=False,
                        type=ColorSpace,
                        choices=list(ColorSpace),
                        default=ColorSpace.RGB,
                        help="Color space for clustering image data")
    parser.add_argument("-d", "--distance",
                        required=False,
                        default=10,
                        help="Preferred vertex distance")
    parser.add_argument("-s", "--splitting",
                        required=False,
                        default=-1,
                        help="Maximum triangle area before splitting into smaller triangles")
    parser.add_argument("-v", "--variance",
                        required=False,
                        default=-1.0,
                        help="Maximum allowed color variance for a triangle to be drawn")
    parser.add_argument("-n", "--noise-kernel",
                        required=False,
                        default=5,
                        help="Kernel size for noise reduction on contours")
    parser.add_argument("-k", "--kmeans",
                        required=False,
                        default=8,
                        help="Centroid count for kmeans color clustering")
    parser.add_argument("-f", "--formats",
                        required=False,
                        type=ExportFormat,
                        choices=list(ExportFormat),
                        default=[ExportFormat.JPG],
                        nargs='+',
                        help="Export formats")
    parser.add_argument("-P", "--show-plot",
                        required=False,
                        action='store_true',
                        help="Flag for plotting image in selected color space")
    parser.add_argument("-C", "--show-contour",
                        required=False,
                        action='store_true',
                        help="Flag for exporting images of contours")
    parser.add_argument("-T", "--show-triangulation",
                        required=False,
                        action='store_true',
                        help="Flag for exporting triangulation of image")
    parser.add_argument("-O", "--original",
                        required=False,
                        action='store_true',
                        help="Flag for also exporting original image in specified export formats")
    return parser.parse_args()


def process(description, function, *argv):
    global PROCESS_STEP
    now = time()

    print(f"{PROCESS_STEP}. {description}", end='\r')
    (result) = function(argv)
    print(f"{PROCESS_STEP}. {description}".ljust(35), f"{(time() - now).total_seconds()}s")
    PROCESS_STEP += 1
    return result


def transform_colorspace(argv):
    image, colorspace = argv
    image_as_ints, unique_ints, unique_colors, unique_counts = dedupe_colors(image)
    translated_unique_colors = to_space(unique_colors, colorspace)
    return image_as_ints, unique_ints, unique_colors, unique_counts, translated_unique_colors


def group_by_color(argv):
    kmeans_centroids, translated_unique_colors, unique_counts, \
        image_as_ints, unique_ints, shape = argv
    labels = kmeans(kmeans_centroids, translated_unique_colors, unique_counts)
    labels = expand_labels(image_as_ints, unique_ints, labels, shape)
    return labels


def contouring(argv):
    image, kmeans_centroids, labels, noise_kernel = argv
    contours = find_contours(image, kmeans_centroids, labels, noise_kernel)
    return contours


def search_vertices(argv):
    contours, distance = argv
    vertices = find_vertices(contours, distance)
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
    triangulation, splitting = argv
    split = split_triangulation(triangulation, splitting)
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


def load_image(path):
    image_name, image_format = os.path.basename(path).split('.', 1)
    if image_format.lower() == "nef" or image_format.lower() == "raw":
        image = rawpy.imread(in_path).postprocess()
    else:
        image = cv2.imread(in_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_name, image


if __name__ == '__main__':
    # =================================
    # ||      Parsing Arguments      ||
    # =================================
    args = parse_arguments()
    in_path = os.path.expanduser(args.input)
    out_path = os.path.expanduser(args.output)
    colorspace = args.colorspace
    distance = int(args.distance)
    splitting = int(args.splitting)
    variance = float(args.variance)
    noise_kernel = int(args.noise_kernel)
    kmeans_centroids = int(args.kmeans)
    export_formats = set(args.formats)
    flag_plot = args.show_plot
    flag_contours = args.show_contour
    flag_triangulation = args.show_triangulation
    flag_original = args.original

    # =============================
    # ||      Image Loading      ||
    # =============================
    image_name, image = load_image(in_path)
    start = time()

    # ======================================
    # ||      Color Space Operations      ||
    # ======================================
    image_as_ints, unique_ints, unique_colors, unique_counts, translated_unique_colors =\
        process("Color Space Transformation", transform_colorspace, image, colorspace)

    if flag_plot is True:
        process("Plotting", plot, unique_colors, translated_unique_colors)

    labels = process("Color Clustering", group_by_color, kmeans_centroids,
                     translated_unique_colors, unique_counts,
                     image_as_ints, unique_ints, image.shape)

    # ==================================
    # ||      Contour Operations      ||
    # ==================================
    contours = process("Contouring", contouring, image,
                       kmeans_centroids, labels, noise_kernel)

    if flag_contours is True:
        process("Writing Contours", write_contours, out_path, image, image_name, colorspace)

    vertices = process("Vertex Search", search_vertices, contours, distance)

    # ===================================
    # ||      Triangle Operations      ||
    # ===================================
    triangulation = process("Triangulation", triangulate, image.shape, vertices)

    if splitting > 0:
        triangulation = process("Triangle Splitting", triangle_splitting, triangulation, splitting)

    if flag_triangulation is True:
        process("Writing Triangulation", write_triangulation, out_path, image, image_name, colorspace)

    # ============================
    # ||      Colorization      ||
    # ============================
    colorized_image = process("Triangle Colorization", colorize_triangles, image, triangulation)

    # ============================
    # ||      Finalization      ||
    # ============================
    end = time()
    print(45 * "-", "\n", "Total Time: ".ljust(35), f"{(end - start).total_seconds()}s")
    export(f"{out_path}/{image_name}", colorized_image, image, export_formats, flag_original)
