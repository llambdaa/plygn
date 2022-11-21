import cv2
import numpy
import numpy as np

from palette import *
from bresenham import *


def find_bounds(a, b, c):
    # The pixels along the edges of the triangle
    # specified by a, b and c are part of the triangle.
    # Everything in between is also inside the triangle.
    ab = bresenham(a[0], a[1], b[0], b[1])
    bc = bresenham(b[0], b[1], c[0], c[1])
    ca = bresenham(c[0], c[1], a[0], a[1])
    outline = list(ab)
    outline.extend(bc)
    outline.extend(ca)

    # Each row is mapped to x_min and x_max, which are
    # the intersection values of Bresenham lines between
    # the triangle's vertices in that row. Everything in
    # between those x-values is inside the triangle.
    y = [a[1], b[1], c[1]]
    ymin, ymax = min(y), max(y)
    bounds = [(float("inf"), float("-inf")) for _ in range(ymin, ymax + 1)]

    for x, y in outline:
        xmin, xmax = bounds[y - ymin]
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        bounds[y - ymin] = (xmin, xmax)

    return ymin, ymax, bounds


def find_color(ymin, ymax, rows, image):
    colors = [image[y][x] for y in range(ymin, ymax + 1) for x in range(rows[y - ymin][0], rows[y - ymin][1] + 1)]
    average = np.array(colors).mean(axis=0)
    return average


def colorize2(image, triangulation):
    canvas = image.copy()
    for x1, y1, x2, y2, x3, y3 in triangulation:
        a = (int(x1), int(y1))
        b = (int(x2), int(y2))
        c = (int(x3), int(y3))
        ymin, ymax, rows = find_bounds(a, b, c)
        color = find_color(ymin, ymax, rows, image)

        for y in range(ymin, ymax + 1):
            xmin, xmax = rows[y - ymin]
            for x in range(xmin, xmax + 1):
                canvas[y][x] = color

    return canvas


def colorize(image, triangulation):
    height, width, _ = image.shape
    line_info = get_line_info(width, height, triangulation)


def get_line_info(width, height, triangulation):
    line_info_matrix = np.full((height, width), -1, dtype=np.int32)
    for triangle in triangulation:
        pass


def order_vertices(coordinates):
    x1, y1, x2, y2, x3, y3 = coordinates
    # Order vertices by x coordinate
    if x1 > x2:
        x2, y2, x1, y1 = x1, y1, x2, y2

    if x2 > x3:
        x3, y3, x2, y2 = x2, y2, x3, y3

    if x1 > x2:
        x2, y2, x1, y1 = x1, y1, x2, y2

    if x2 == x3:
        # The two rightmost vertices are on the same
        # x-level. The upper is 'c', the lower is 'b'
        ax, ay = x1, y1
        if y3 < y2:
            cx, cy = x3, y3
            bx, by = x2, y2
        else:
            cx, cy = x2, y2
            bx, by = x3, y3
    elif x1 == x2:
        # The two leftmost vertices are on the same
        # x-level. The upper is 'a', the lower is 'b'
        cx, cy = x3, y3
        if y1 < y2:
            ax, ay = x1, y1
            bx, by = x2, y2
        else:
            ax, ay = x2, y2
            bx, by = x1, x1
    else:
        ax, ay, bx, by, cx, cy = x1, y1, x2, y2, x3, y3
    return ax, ay, bx, by, cx, cy

