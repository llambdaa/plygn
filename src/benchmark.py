import os
import torch
import numpy as np
import json
import cv2

from export import *
from plygn import load_image
from utils import *
from enum import Enum, auto
from pytorch_msssim import ms_ssim


BENCHMARK_PATH = f"{{0}}/benchmark.json"


class ResultType(Enum):
    PROCESSED   = "processed"
    UNPROCESSED = "unprocessed"

    def __str__(self):
        return self.value


class MeasurementType(Enum):
    SIMPLE      = auto()
    COMPARATIVE = auto()


def get_numpy_image(input):
    _, image = load_image(input)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    return image


def to_torch_image(numpy_image):
    return torch.from_numpy(numpy_image)
    

def get_pixel_size(image):
    _, _, height, width = image.shape
    return height * width


def get_measurement_reference(input, original_size, original_mem, processing_time, total_time):
    reference = {
        "path_original": input,
        "size_original": original_mem,
        "time_processing": processing_time,
        "time_total": total_time,
        "bpp": (original_mem / original_size)
    }
    return reference


def measure(original, compressed):
    print("SIM1. PSNR", end="\r")
    start = time()
    similarity_psnr = cv2.PSNR(original, compressed)
    delta = (time() - start).total_seconds()
    print(f"SIM1. PSNR".ljust(35), f"{delta}s")

    print("SIM2. MS-SSIM", end="\r")
    start = time()
    original = to_torch_image(original)
    compressed = to_torch_image(compressed)
    similarity_msssim = ms_ssim(original, compressed, data_range=255, size_average=False).item()
    delta = (time() - start).total_seconds()
    print(f"SIM2. MS_SSIM".ljust(35), f"{delta}s")

    return similarity_psnr, similarity_msssim


def get_measurement(path, original_image):
    result_image = get_numpy_image(path)
    result_size = get_pixel_size(result_image)
    result_mem = os.path.getsize(path)
    psnr, msssim = measure(original_image, result_image)

    measurement = {
        "path": path,
        "size": result_size,
        "psnr": psnr,
        "msssim": msssim,
        "bpp": (result_mem / result_size)
    }
    return measurement


def get_impact(processed_measurement, unprocessed_measurement):
    processed_psnr = processed_measurement["psnr"]
    unprocessed_psnr = unprocessed_measurement["psnr"]
    impact_psnr = processed_psnr - unprocessed_psnr

    processed_msssim = processed_measurement["msssim"]
    unprocessed_msssim = unprocessed_measurement["msssim"]
    impact_msssim = processed_msssim - unprocessed_msssim

    processed_bpp = processed_measurement["bpp"]
    unprocessed_bpp = unprocessed_measurement["bpp"]
    impact_bpp = (1 - (processed_bpp / unprocessed_bpp)) * 100

    impact = {
        "psnr": impact_psnr,
        "msssim": impact_msssim,
        "bpp": impact_bpp
    }
    return impact


def get_format_entry(output, format, measurement_type, original_image):
    format_suffix = str(format).lower()
    path_template = f"{output}_{{0}}.{format_suffix}"
    format_entry = {}

    print("> Processed:")
    processed_path = path_template.format(ResultType.PROCESSED)
    processed_measurement = get_measurement(processed_path, original_image)
    format_entry[str(ResultType.PROCESSED)] = processed_measurement

    if (measurement_type is MeasurementType.COMPARATIVE):
        print("> Unprocessed:")
        unprocessed_path = path_template.format(ResultType.UNPROCESSED)
        unprocessed_measurement = get_measurement(unprocessed_path, original_image)
        format_entry[str(ResultType.UNPROCESSED)] = unprocessed_measurement

        impact = get_impact(processed_measurement, unprocessed_measurement)
        format_entry["impact"] = impact

    return format_entry


def get_benchmark_entry(input, output, formats, measurement_type, processing_time, total_time):
    print("\nBenchmarking:")
    original_image = get_numpy_image(input)
    original_size = get_pixel_size(original_image)
    original_mem = os.path.getsize(input)
    benchmark_entry = {}

    reference = get_measurement_reference(input, original_size, original_mem, processing_time, total_time)
    benchmark_entry["reference"] = reference

    for format in formats:
        format_entry = get_format_entry(output, format, measurement_type, original_image)
        benchmark_entry[str(format)] = format_entry

    return benchmark_entry


def write_benchmarks(output, benchmarks):
    path = BENCHMARK_PATH.format(output)
    with open(path, "w+") as benchmark_file:
        json.dump(benchmarks, benchmark_file, ensure_ascii=False, indent=4)
        print("\nBenchmarks have been written to:")
        print(truncate_path(path, 3))
