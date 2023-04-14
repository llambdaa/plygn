import os
import torch
import numpy as np
import json

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


def get_image(input):
    _, image = load_image(input)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    image = torch.from_numpy(image)
    return image


def get_measurement_header(input, original_size, time):
    header = {
        "path": input,
        "size": original_size,
        "time": time
    }
    return header


def get_measurement(path, original_image, original_size):
    result_image = get_image(path)
    result_size = os.path.getsize(path)
    similarity = ms_ssim(original_image, result_image, data_range=255).item()
    compression = 1 - (result_size / original_size)

    measurement = {
        "path": path,
        "size": result_size,
        "similarity": similarity,
        "compression": compression,
    }
    return measurement


def get_comparison(processed_measurement, unprocessed_measurement):
    processed_size = processed_measurement["size"]
    unprocessed_size = unprocessed_measurement["size"]
    size_impact = 1 - (processed_size / unprocessed_size)

    processed_sim = processed_measurement["similarity"]
    unprocessed_sim = unprocessed_measurement["similarity"]
    similarity_impact = 1 - (processed_sim / unprocessed_sim)

    comparison = {
        "better_size": size_impact,
        "worse_similarity": similarity_impact
    }
    return comparison


def get_format_entry(output, format, measurement_type, original_image, original_size):
    format_suffix = str(format).lower()
    path_template = f"{output}_{{0}}.{format_suffix}"
    format_entry = {}

    processed_path = path_template.format(ResultType.PROCESSED)
    processed_measurement = get_measurement(processed_path, original_image, original_size)
    format_entry[str(ResultType.PROCESSED)] = processed_measurement

    if (measurement_type is MeasurementType.COMPARATIVE):
        unprocessed_path = path_template.format(ResultType.UNPROCESSED)
        unprocessed_measurement = get_measurement(unprocessed_path, original_image, original_size)
        format_entry[str(ResultType.UNPROCESSED)] = unprocessed_measurement

        comparison = get_comparison(processed_measurement, unprocessed_measurement)
        format_entry["comparison"] = comparison

    return format_entry


def get_benchmark_entry(input, output, formats, measurement_type, time):
    original_image = get_image(input)
    original_size = os.path.getsize(input)
    benchmark_entry = {}

    header = get_measurement_header(input, original_size, time)
    benchmark_entry["header"] = header

    for format in formats:
        format_entry = get_format_entry(output, format, measurement_type, original_image, original_size)
        benchmark_entry[str(format)] = format_entry

    return benchmark_entry


def write_benchmarks(output, benchmarks):
    path = BENCHMARK_PATH.format(output)
    with open(path, "w+") as benchmark_file:
        json.dump(benchmarks, benchmark_file, ensure_ascii=False, indent=4)
        print(f"Benchmarks have been written to '{truncate_path(path, 3)}'")
