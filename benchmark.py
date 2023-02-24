import os
import torch
import numpy as np

from plygn import load_image
from export import *
from enum import Enum
from pytorch_msssim import ms_ssim


class PartialResultType(Enum):
    PROCESSED = "processed"
    UNPROCESSED = "unprocessed"

    def __str__(self):
        return self.value


def get_image_tensor(input_path):
    _, image = load_image(input_path)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    image = torch.from_numpy(image)
    return image


def write_partial_result(path, partial_result_type, original_data, format_result):
    path = path.format(partial_result_type)
    image = get_image_tensor(path)
    size = os.path.getsize(path)
    original_image = original_data[0]
    original_size = original_data[1]
    similarity_to_original = ms_ssim(original_image, image, data_range=255).item()

    partial_result = {
        "path": path,
        "size": size,
        "size_ratio_to_original": size / original_size,
        "similarity_to_original": similarity_to_original
    }

    format_result[str(partial_result_type)] = partial_result


def write_partial_result_comparison(format_result):
    processed_result = format_result[str(PartialResultType.PROCESSED)]
    unprocessed_result = format_result[str(PartialResultType.UNPROCESSED)]

    # Both image format conversion/compression with and without
    # prior triangulation are performed. Both partial results
    # can now be compared in terms of size and similarity with
    # respect to the shared common original.
    processed_size = processed_result["size"]
    unprocessed_size = unprocessed_result["size"]
    format_result["size_impact"] = processed_size / unprocessed_size

    processed_similarity = processed_result["similarity_to_original"]
    unprocessed_similarity = unprocessed_result["similarity_to_original"]
    format_result["similarity_impact"] = processed_similarity / unprocessed_similarity


def get_format_result(format_path, flag_unprocessed, original_data):
    format_result = {}

    # Input Image ---> Processed Image ---> PNG/JPG/QOI
    write_partial_result(format_path, PartialResultType.PROCESSED, original_data, format_result)

    # Input Image ---> PNG/JPG/QOI
    if flag_unprocessed:
        write_partial_result(format_path, PartialResultType.UNPROCESSED, original_data, format_result)
        write_partial_result_comparison(format_result)

    return format_result


def benchmark(input_path, output_path, time, formats, flag_unprocessed):
    benchmark_result = {}
    
    # Base Information
    original_image = get_image_tensor(input_path)
    original_size = os.path.getsize(input_path)
    original_data = (original_image, original_size)
    original_object = {
        "path": input_path,
        "size": original_size,
        "time": time
    }
    benchmark_result["original"] = original_object

    # Format Specific Information
    for format in formats:
        format_name = str(format).lower()
        format_path = f"{output_path}_{{0}}.{format_name}"
        format_result = get_format_result(format_path, flag_unprocessed, original_data)
        benchmark_result[format_name] = format_result

    return benchmark_result
