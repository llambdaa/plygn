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


def colorize3(image, triangulation):
    height, width, _ = image.shape
    line_info_matrix = get_line_info_matrix(width, height, triangulation)


def get_line_info_matrix(width, height, triangulation):
    # Each triangle's vertices are ordered, so that there
    # are two leftmost vertices 'a' and 'b' and a rightmost
    # vertex 'c'. The latter defines which edges must be
    # drawn to the line info matrix, because the triangle
    # is right to them.
    line_info_matrix = np.full((height, width), -1, dtype=np.int32)
    for index, triangle in enumerate(triangulation):
        # The triangle's vertices are
        # ordered for easier checks.
        ax, ay, bx, by, cx, cy = order_vertices(triangle)

        if cy >= by:
            # Vertex 'c' below both other vertices
            points = np.array([[ax, ay], [bx, by], [cx, cy]], dtype=np.int32)

        elif cy <= ay:
            # Vertex 'c' above both other vertices
            if ax > bx:
                points = np.array([[bx, by], [cx, cy]], dtype=np.int32)
            else:
                points = np.array([[bx, by], [ax, ay], [cx, cy]], dtype=np.int32)

        else:
            # Vertex 'c' between both other vertices
            points = np.array([[ax, ay], [bx, by]], dtype=np.int32)

        cv2.polylines(line_info_matrix, [points], False, index, thickness=1)


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
        # The two rightmost vertices are on the same x-level.
        # The upper vertex is 'c', the lower is 'b'.
        if y2 < y3:
            x3, y3, x2, y2 = x2, y2, x3, y3

    if y1 != y2 and x1 != x2:
        # The two leftmost vertices are not on the same y-level.
        # Then, vertex 'a' is the upper one, 'b' the lower one.
        if y1 > y2:
            x2, y2, x1, y1 = x1, y1, x2, y2

    return x1, y1, x2, y2, x3, y3
