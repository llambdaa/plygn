import qoi
import cv2
from enum import Enum


class ExportFormat(Enum):
    JPG = 'JPG'
    PNG = 'PNG'
    QOI = 'QOI'

    def __str__(self):
        return self.value


def export(path, processed, unprocessed, export_formats, export_unprocessed):
    if ExportFormat.JPG in export_formats:
        cv2.imwrite(
            f"{path}_processed.jpg",
            cv2.cvtColor(processed, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        )

        if export_unprocessed:
            cv2.imwrite(
                f"{path}_unprocessed.jpg",
                cv2.cvtColor(unprocessed, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            )

    if ExportFormat.PNG in export_formats:
        cv2.imwrite(
            f"{path}_processed.png",
            cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        )

        if export_unprocessed:
            cv2.imwrite(
                f"{path}_unprocessed.png",
                cv2.cvtColor(unprocessed, cv2.COLOR_RGB2BGR)
            )

    if ExportFormat.QOI in export_formats:
        qoi.write(f"{path}_processed.qoi", processed)
        if export_unprocessed:
            qoi.write(f"{path}_unprocessed.qoi", unprocessed)
