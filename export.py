import qoi
from enum import Enum


class ExportFormat(Enum):
    JPG = 'JPG'
    PNG = 'PNG'
    QOI = 'QOI'

    def __str__(self):
        return self.value


def export(path, processed, original, export_formats, export_original):
    if ExportFormat.JPG in export_formats:
        cv2.imwrite(
            f"{path}_processed.jpg",
            cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        )

        if export_original:
            cv2.imwrite(
                f"{path}_original.jpg",
                cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
            )

    if ExportFormat.PNG in export_formats:
        cv2.imwrite(
            f"{path}_processed.png",
            cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        )

        if export_original:
            cv2.imwrite(
                f"{path}_original.png",
                cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
            )

    if ExportFormat.QOI in export_formats:
        qoi.write(f"{path}_processed.qoi", processed)
        if export_original:
            qoi.write(f"{path}_original.qoi", original)
