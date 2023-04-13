#!/usr/bin/env python3
import argparse
import cv2
import os
import sys
import rawpy
import json
import qoi

from utils import *
from colorspace import *
from clustering import *
from contouring import *
from triangulation import *
from colorization import *
from export import *
from benchmark import *


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
    parser.add_argument("-B", "--benchmark",
                        required=False,
                        action='store_true',
                        help="Flag for printing and logging compression benchmarks")
    parser.add_argument("-P", "--plot",
                        required=False,
                        action='store_true',
                        help="Flag for plotting image in selected color space")
    parser.add_argument("-C", "--export-contours",
                        required=False,
                        action='store_true',
                        help="Flag for exporting images of contours")
    parser.add_argument("-T", "--export-triangulation",
                        required=False,
                        action='store_true',
                        help="Flag for exporting triangulation of image")
    parser.add_argument("-U", "--export-unprocessed",
                        required=False,
                        action='store_true',
                        help="Flag for exporting unprocessed image in specified formats for comparison")
    return parser.parse_args()


def load_image(path):
    image_name, image_format = os.path.basename(path).split('.', 1)
    if image_format.upper() in ["NEF", "RAW"]:
        image_data = rawpy.imread(path).postprocess()
    elif image_format.upper() in ["QOI"]:
        image_data = qoi.read(path)
    else:
        image_data = cv2.imread(path)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    return image_name, image_data


def logging_pre(description):
    global logging_desc
    global logging_time
    logging_desc = description
    logging_time = time()
    print(f"{logging_step}. {description}", end='\r')


def logging_post():
    global logging_step
    delta = (time() - logging_time).total_seconds()
    print(f"{logging_step}. {logging_desc}".ljust(35), f"{delta}s")
    logging_step += 1


def process_image(in_path):
    # =============================
    # ||      Image Loading      ||
    # =============================
    image_name, image_data = load_image(in_path)
    print(f"Processing '{truncate_path(in_path, 3)}'")
    start = time()

    # ======================================
    # ||      Color Space Operations      ||
    # ======================================
    logging_pre("Color Space Transformation")
    image_as_ints, unique_ints, unique_colors, unique_counts = dedupe_colors(image_data)
    translated_unique_colors = to_space(unique_colors, colorspace)
    logging_post()

    if flag_plot is True:
        logging_pre("Plotting")
        plot(unique_colors, translated_unique_colors)
        logging_post()

    logging_pre("Color Clustering")
    labels = kmeans(kmeans_centroids, translated_unique_colors, unique_counts)
    labels = expand_labels(image_as_ints, unique_ints, labels, image_data.shape)
    logging_post()

    # ==================================
    # ||      Contour Operations      ||
    # ==================================
    logging_pre("Contouring")
    contours = find_contours(image_data, kmeans_centroids, labels, noise_kernel)
    logging_post()

    if flag_contours is True:
        logging_pre("Exporting Contours")
        partial_folder = make_folder(out_path, image_name)
        export_folder = make_folder(partial_folder, colorspace)
        export_contours(image_data, contours, export_folder)
        logging_post()

    logging_pre("Vertex Search")
    vertices = find_vertices(contours, distance)
    logging_post()

    # ===================================
    # ||      Triangle Operations      ||
    # ===================================
    logging_pre("Triangulation")
    triangulation = find_triangulation(image_data.shape, vertices)
    logging_post()

    if splitting > 0:
        logging_pre("Triangle Splitting")
        triangulation = split_triangulation(triangulation, splitting)
        logging_post()

    if flag_triangulation is True:
        logging_pre("Exporting Triangulation")
        partial_folder = make_folder(out_path, image_name)
        export_folder = make_folder(partial_folder, colorspace)
        export_triangulation(image_data, triangulation, export_folder)
        logging_post()

    # ============================
    # ||      Colorization      ||
    # ============================
    logging_pre("Triangle Colorization")
    colorized_image = colorize(image_data, triangulation)
    logging_post()

    # ============================
    # ||      Finalization      ||
    # ============================
    delta = (time() - start).total_seconds()
    print(45 * "-")
    print("Total Time: ".ljust(35), f"{delta}s")

    output_basename = f"{out_path}/{image_name}"
    export(output_basename, colorized_image, image_data, export_formats, flag_unprocessed)

    # ==========================
    # ||      Benchmarking    ||
    # ==========================
    if flag_benchmark:
        result = get_benchmark_entry(in_path, output_basename, delta, export_formats, flag_unprocessed)
        benchmark_results.append(result)


def is_supported_image_format(path):
    basename = os.path.basename(path)
    if not "." in basename:
        return False

    _, format = basename.split('.', 1)
    return format.upper() in ["NEF", "RAW", "JPG", "JPEG", "PNG", "BMP"]


def logging_pre(description):
    global logging_desc
    global logging_time
    logging_desc = description
    logging_time = time()
    print(f"{logging_step}. {description}", end='\r')


def logging_post():
    global logging_step
    delta = (time() - logging_time).total_seconds()
    print(f"{logging_step}. {logging_desc}".ljust(35), f"{delta}s")
    logging_step += 1


def process_image(in_path):
    # =============================
    # ||      Image Loading      ||
    # =============================
    image_name, image_data = load_image(in_path)
    print(f"Processing '{truncate_path(in_path, 3)}'")
    start = time()

    # ======================================
    # ||      Color Space Operations      ||
    # ======================================
    logging_pre("Color Space Transformation")
    image_as_ints, unique_ints, unique_colors, unique_counts = dedupe_colors(image_data)
    translated_unique_colors = to_space(unique_colors, colorspace)
    logging_post()

    if flag_plot is True:
        logging_pre("Plotting")
        plot(unique_colors, translated_unique_colors)
        logging_post()

    logging_pre("Color Clustering")
    labels = kmeans(kmeans_centroids, translated_unique_colors, unique_counts)
    labels = expand_labels(image_as_ints, unique_ints, labels, image_data.shape)
    logging_post()

    # ==================================
    # ||      Contour Operations      ||
    # ==================================
    logging_pre("Contouring")
    contours = find_contours(image_data, kmeans_centroids, labels, noise_kernel)
    logging_post()

    if flag_contours is True:
        logging_pre("Exporting Contours")
        partial_folder = make_folder(out_path, image_name)
        export_folder = make_folder(partial_folder, colorspace)
        export_contours(image_data, contours, export_folder)
        logging_post()

    logging_pre("Vertex Search")
    vertices = find_vertices(contours, distance)
    logging_post()

    # ===================================
    # ||      Triangle Operations      ||
    # ===================================
    logging_pre("Triangulation")
    triangulation = find_triangulation(image_data.shape, vertices)
    logging_post()

    if splitting > 0:
        logging_pre("Triangle Splitting")
        triangulation = split_triangulation(triangulation, splitting)
        logging_post()

    if flag_triangulation is True:
        logging_pre("Exporting Triangulation")
        partial_folder = make_folder(out_path, image_name)
        export_folder = make_folder(partial_folder, colorspace)
        export_triangulation(image_data, triangulation, export_folder)
        logging_post()

    # ============================
    # ||      Colorization      ||
    # ============================
    logging_pre("Triangle Colorization")
    colorized_image = colorize(image_data, triangulation)
    logging_post()

    # ============================
    # ||      Finalization      ||
    # ============================
    delta = (time() - start).total_seconds()
    print(45 * "-")
    print("Total Time: ".ljust(35), f"{delta}s")

    output_basename = f"{out_path}/{image_name}"
    export(output_basename, colorized_image, image_data, export_formats, flag_unprocessed)

    # ==========================
    # ||      Benchmarking    ||
    # ==========================
    if flag_benchmark:
        result = get_benchmark_entry(in_path, output_basename, export_formats, flag_unprocessed, delta)
        benchmark_results.append(result)


def is_supported_image_format(path):
    basename = os.path.basename(path)
    if not "." in basename:
        return False

    _, format = basename.split('.', 1)
    return format.upper() in ["NEF", "RAW", "JPG", "JPEG", "PNG", "BMP"]


if __name__ == '__main__':
    # =================================
    # ||      Parsing Arguments      ||
    # =================================
    # Program arguments
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
    flag_benchmark = args.benchmark
    flag_plot = args.plot
    flag_contours = args.export_contours
    flag_triangulation = args.export_triangulation
    flag_unprocessed = args.export_unprocessed

    # Logging
    logging_step = 1
    logging_time = None
    logging_desc = None
    benchmark_results = list()

    # ========================
    # ||      Processing    ||
    # ========================
    if not os.path.exists(in_path):
        sys.exit(f"File '{in_path}' not found!")

    if os.path.isfile(in_path):
        # The given path points to a file. The image gets processed
        # if its type is supported, otherwise an error is reported.
        supported = is_supported_image_format(in_path)
        if not supported:
            sys.exit(f"File '{truncate_path(in_path, 3)}' does not have supported format!")

        process_image(in_path)

    elif os.path.isdir(in_path):
        # The given path points to a directory. It is attempted that
        # each image gets processed. If its type is not supported,
        # the file gets skipped.
        processed_images = 0
        for entry in os.listdir(in_path):
            file = os.path.join(in_path,entry)
            supported = is_supported_image_format(file)
            if not supported:
                continue
            
            # Placeholder blank line
            if processed_images > 0:
                print()

            process_image(file)
            processed_images += 1

            # Reset logging step counter
            logging_step = 1
        
        print(f">> In total {processed_images} images have been processed.")

    # ==========================
    # ||      Benchmarking    ||
    # ==========================
    if flag_benchmark:
        benchmark_path = f"{out_path}/benchmark.json"
        with open(benchmark_path, "w+") as benchmark_file:
            json.dump(benchmark_results, benchmark_file, ensure_ascii=False, indent=4)
            print(f">> Benchmark results have been written out to '{truncate_path(benchmark_path, 3)}'.")
            