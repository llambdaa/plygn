import cv2
import numpy as np

CONTOUR_THICKNESS = 1
CONTOUR_COLOR = (255, 0, 255)


def denoise_bitmask(bitmask, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Remove bitmask islands
    # inside and outside
    result = cv2.morphologyEx(bitmask, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result


def find_contours(image, cluster_count, labels, kernel_size):
    bitmask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contour_groups = list()
    for k in range(cluster_count):
        # Each bitmask pixel assigned to center k
        # is marked white, else black
        bitmask[labels == k] = 255
        bitmask[labels != k] = 0

        # If a denoise kernel size is given
        # the bitmask gets denoised
        if kernel_size > 0:
            bitmask = denoise_bitmask(bitmask, kernel_size)

        contour_group, _ = cv2.findContours(bitmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_groups.append(contour_group)

    return contour_groups


def expand_contour(line):
    result = list()
    if len(line) < 3:
        return result

    vertices = list(map(lambda x: (x[0][0], x[0][1]), line))
    a = vertices[0]

    # Put first element as last to
    # effectively close the loop
    vertices.pop(0)
    vertices.append(a)

    for b in vertices:
        if a[0] == b[0]:
            # Both vertices a and b
            # are on the same x level
            ay, by = a[1], b[1]
            result.extend([(a[0], y) for y in range(ay, by, np.sign(by - ay))])

        elif a[1] == b[1]:
            # Both vertices a and b
            # are on the same y level
            ax, bx = a[0], b[0]
            result.extend([(x, a[1]) for x in range(ax, bx, np.sign(bx - ax))])

        else:
            # Point a is stand-alone
            result.append((a[0], a[1]))

        a = b

    return result


def show_contours(image, contour_groups, out_path):
    combined_image = image.copy()
    for i, contour_group in enumerate(contour_groups):
        # For each cluster, a separate
        # output image is computed
        cluster_specific = image.copy()
        cv2.drawContours(cluster_specific, contour_group, -1, CONTOUR_COLOR, CONTOUR_THICKNESS)

        cv2.imwrite(
            f"{out_path}/cluster_{i}.png",
            cv2.cvtColor(cluster_specific, cv2.COLOR_RGB2BGR)
        )

        # Each cluster contour is also
        # written to a combined image
        cv2.drawContours(combined_image, contour_group, -1, CONTOUR_COLOR, CONTOUR_THICKNESS)

    # After drawing all contours the
    # combined contour image is exported
    cv2.imwrite(
        f"{out_path}/combined.png",
        cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
    )
