from enum import Enum


class ExportFormat(Enum):
    JPG = 'jpg'
    PNG = 'png'
    QOI = 'qoi'

    def __str__(self):
        return self.value


def __export(path, processed, original, export_format, export_original):
    cv2.imwrite(
        f"{path}_processed.{export_format.value}",
        cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
    )

    if export_original:
        cv2.imwrite(
            f"{path}_original.{export_format.value}",
            cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        )


def export(path, processed, original, export_formats, export_original):
    for specific in export_formats:
        __export(path, processed, original, specific, export_original)
