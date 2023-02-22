import os

from export import *


def benchmark(input, output, time, formats, flag_unprocessed):
    result = {}
    
    # Base Information
    original_size = os.path.getsize(input)
    original_image = {
        "name": input,
        "size": original_size,
        "time": time
    }
    result["original"] = original_image

    # Format Specific Information
    if ExportFormat.JPG in formats:
        jpg_object = get_format_object(output, "jpg", original_size, flag_unprocessed)
        result["jpg"] = jpg_object
 
    if ExportFormat.PNG in formats:
        png_object = get_format_object(output, "png", original_size, flag_unprocessed)
        result["png"] = png_object

    if ExportFormat.QOI in formats:
        qoi_object = get_format_object(output, "qoi", original_size, flag_unprocessed)
        result["qoi"] = qoi_object

    return result


def get_format_object(output, format, input_size, flag_unprocessed):
    format_object = {}

    # Input Image ---> Processed Image ---> PNG/JPG/QOI
    processed_name = f"{output}_processed.{format}"
    processed_size = os.path.getsize(processed_name)
    processed_image = {
        "name": processed_name,
        "size": processed_size,
        "ratio_to_original": processed_size / input_size,
        "similarity": "n/a"
    }
    format_object["processed"] = processed_image

    if flag_unprocessed:
        # Input Image ---> PNG/JPG/QOI
        unprocessed_name = f"{output}_unprocessed.{format}"
        unprocessed_size = os.path.getsize(unprocessed_name)
        unprocessed_image = {
            "name": unprocessed_name,
            "size": unprocessed_size,
            "ratio_to_original": unprocessed_size / input_size,
            "similarity": "n/a"
        }
        format_object["unprocessed"] = unprocessed_image

        # At this point, compression happens with and without
        # previous triangulation. Its impact hence can be determined.
        format_object["triangulation_impact"] = processed_size / unprocessed_size

    return format_object