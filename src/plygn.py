#!/usr/bin/env python3
import argparse
import cv2
import os
import qoi
import rawpy
import sys

from benchmark import *
from clustering import *
from colorization import *
from colorspace import *
from contouring import *
from export import *
from triangulation import *
from utils import *


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


def write_parameter_hint(path):
    with open(f"{path}/parameters.txt", 'w') as file:
        file.write(' '.join(sys.argv))


def is_supported_image_format(path):
    basename = os.path.basename(path)
    if not "." in basename:
        return False

    _, format = basename.split('.', 1)
    return format.upper() in ["NEF", "RAW", "JPG", "JPEG", "PNG", "BMP"]


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


def add_benchmark(benchmark):
    benchmark_results.append(benchmark)
    write_benchmarks(output_path, benchmark_results)


def process_image(in_path, out_path):
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
    colorized_image = colorize(image_data, triangulation, variance)
    logging_post()

    processing_time = (time() - start).total_seconds()
    print(45 * "-")
    print("Processing Time: ".ljust(35), f"{processing_time}s")

    # ======================
    # ||      Export      ||
    # ======================
    output_basename = os.path.normpath(f"{out_path}/{image_name}")
    export(output_basename, colorized_image, image_data, export_formats, flag_unprocessed)
    total_time = (time() - start).total_seconds()
    print("Total Time: ".ljust(35), f"{total_time}s")

    # ==========================
    # ||      Benchmarking    ||
    # ==========================
    if flag_benchmark:
        measurement_type = MeasurementType.SIMPLE if not flag_unprocessed else MeasurementType.COMPARATIVE
        benchmark = get_benchmark_entry(in_path, output_basename, export_formats, measurement_type, processing_time, total_time)
        add_benchmark(benchmark)


def process_images(targets, out_path):
    processed_images = 0
    for file in targets:
        if not is_supported_image_format(file):
            continue

        if processed_images > 0:
            print("\n{}\n".format("=" * 60))

        process_image(file, out_path)
        processed_images += 1

        global logging_step
        logging_step = 1

    return processed_images


if __name__ == '__main__':
    args = parse_arguments()
    input_path = os.path.expanduser(args.input)
    output_path = os.path.expanduser(args.output)
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
    benchmark_results = list()
    logging_step = 1

    if not os.path.exists(input_path):
        sys.exit(f"Target '{input_path}' has not been found!")

    write_parameter_hint(output_path)
    if os.path.isfile(input_path):
        if not is_supported_image_format(input_path):
            sys.exit(f"File '{truncate_path(input_path, 3)}' does not have a supported format!")
            
        process_image(input_path, output_path)

    elif os.path.isdir(input_path):
        targets = [os.path.join(input_path, entry) for entry in os.listdir(input_path)]
        processed_images = process_images(targets, output_path)
        print(f"[ {processed_images} images have been processed! ]")
            