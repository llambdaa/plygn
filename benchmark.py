import os

from sewar.full_ref import uqi

from plygn import load_image
from export import *


def benchmark(input_path, output_path, time, formats, flag_unprocessed):
    result = {}
    
    # Base Information
    _, original_image = load_image(input_path)
    original_size = os.path.getsize(input_path)
    original_object = {
        "path": input_path,
        "size": original_size,
        "time": time
    }
    result["original"] = original_object

    # Format Specific Information
    if ExportFormat.JPG in formats:
        jpg_object = get_format_object(original_image, original_size, output_path, "jpg", flag_unprocessed)
        result["jpg"] = jpg_object
 
    if ExportFormat.PNG in formats:
        png_object = get_format_object(original_image, original_size, output_path, "png", flag_unprocessed)
        result["png"] = png_object

    if ExportFormat.QOI in formats:
        qoi_object = get_format_object(original_image, original_size, output_path, "qoi", flag_unprocessed)
        result["qoi"] = qoi_object

    return result


def get_format_object(original_image, original_size, output_path, format, flag_unprocessed):
    format_object = {}

    # Input Image ---> Processed Image ---> PNG/JPG/QOI
    processed_path = f"{output_path}_processed.{format}"
    _, processed_image = load_image(processed_path)
    processed_size = os.path.getsize(processed_path)
    processed_uqi = uqi(original_image, processed_image)
    processed_object = {
        "path": processed_path,
        "size": processed_size,
        "size_ratio_to_original": processed_size / original_size,
        "uqi_similarity_index": processed_uqi
    }
    format_object["processed"] = processed_object

    if flag_unprocessed:
        # Input Image ---> PNG/JPG/QOI
        unprocessed_path = f"{output_path}_unprocessed.{format}"
        _, unprocessed_image = load_image(unprocessed_path)
        unprocessed_size = os.path.getsize(unprocessed_path)
        unprocessed_uqi = uqi(original_image, unprocessed_image)
        unprocessed_object = {
            "path": unprocessed_path,
            "size": unprocessed_size,
            "size_ratio_to_original": unprocessed_size / original_size,
            "uqi_similarity_index": unprocessed_uqi
        }
        format_object["unprocessed"] = unprocessed_object

        # At this point, compression happens with and without
        # previous triangulation. Its impact hence can be determined.
        format_object["size_impact"] = processed_size / unprocessed_size
        format_object["similarity_impact"] = processed_uqi / unprocessed_uqi

    return format_object