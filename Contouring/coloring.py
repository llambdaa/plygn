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
    # Triangle edges is rasterized to pixels using Bresenham's
    # algorithm. Each edge is given an ID beforehand. That ID
    # is written at each of the line's pixel locations in a
    # helper array with the image's shape.
    height, width, _ = image.shape
    line_info = np.full((height, width), -1, dtype=np.int32)

    triangles = triangulation.simplices
    """
    for a, b, c in triangles:
        print(vertices[a], vertices[b], vertices[c])
    """

    pass
