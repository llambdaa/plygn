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
    # The triangle vertices are transformed into indices of
    # the 1D representation of the image they are placed in.
    # Then, those index representations are sorted.
    height, width, _ = image.shape
    i1, i2, i3 = to_indices(width, triangulation)

    # Then, the sorted vertex indices are transformed into
    # index differences. The differences describe the relative
    # positioning of the vertices in the image and thus the
    # concrete triangle shape.
    differences = to_differences(i1, i2, i3)
    unique_differences = np.unique(differences, return_index=True)


def to_indices(width, triangulation):
    # The triangulation result (matrix) is transposed and
    # the corresponding x- and y-coordinates are extracted
    triangulation = triangulation.T
    x1, y1, x2, y2, x3, y3 = triangulation

    # The indices are calculated using:
    #   i = y * width + x
    y1 = np.multiply(y1, width)
    y2 = np.multiply(y2, width)
    y3 = np.multiply(y3, width)

    i1 = np.add(x1, y1)
    i2 = np.add(x2, y2)
    i3 = np.add(x3, y3)

    # The vertex indices are combined again, so that
    # the three vertices of a triangle (contained in
    # i1, i2 and i3 respectively) can be sorted in
    # ascending order.
    stacked = np.column_stack((i1, i2, i3))
    sorted = np.sort(stacked, axis=1)
    i1, i2, i3 = sorted.T

    return i1, i2, i3


def to_differences(i1, i2, i3):
    d1 = np.uint64(np.subtract(i2, i1))
    d2 = np.uint64(np.subtract(i3, i2))

    # The differences 'ab' and 'bc' are combined into
    # one 64-bit int, which describes the triangle's shape
    differences = np.add(np.left_shift(d2, 32), d1)
    return differences

