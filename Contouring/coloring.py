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


def colorize(image, triangulation):
    canvas = image.copy()
    for a, b, c in triangulation:
        ymin, ymax, rows = find_bounds(a, b, c)
        color = find_color(ymin, ymax, rows, image)

        for y in range(ymin, ymax + 1):
            xmin, xmax = rows[y - ymin]
            for x in range(xmin, xmax + 1):
                canvas[y][x] = color

    return canvas


"""
def colorize2(image, triangulation):
    canvas = image.copy()
    association = associate_pixels_and_triangles(triangulation)

    for triangle, pixels in association:
        color = find_color(pixels, image)

        for x, y in pixels:
            canvas[y][x] = color

    return canvas


def associate_pixels_and_triangles(triangulation):
    association = list()
    for triangle in triangulation:
        # For better performance, calculate triangle-dependent
        # terms of barycentric coordinate calculations beforehand
        # and then determine point-dependent terms
        a, b, c = triangle

        # Vertices a and c shall not be co-linear on the y-axis
        alpha = c[1] - a[1]
        if alpha == 0:
            a, b = b, a
            alpha = c[1] - a[1]

        # Compute triangle-dependent terms
        beta = a[0] * alpha
        gamma = c[0] - a[0]
        delta = c[1] - a[1]
        epsilon = b[1] - a[1]
        zeta = epsilon * gamma
        eta = (b[0] - a[0]) * alpha

        # Iterate over all pixels that
        # could be inside the triangle
        xmin, ymin, xmax, ymax = bounding_box(a, b, c)
        pixels = list()

        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                p = (x, y)

                # Compute point-dependent coordinates
                theta = (beta + ((p[1] - a[1]) * gamma) - (p[0] * delta))
                iota = zeta - eta

                u = theta / iota
                v = (p[1] - a[1] - u * epsilon) / alpha

                # If pixel is inside the triangle,
                # it becomes associated with it
                if u >= 0 and v >= 0 and (u + v) <= 1:
                    pixels.append(p)

        association.append((triangle, pixels))

    return association


def bounding_box(a, b, c):
    x = [a[0], b[0], c[0]]
    y = [a[1], b[1], c[1]]

    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    return xmin, ymin, xmax, ymax


def find_color2(pixels, image):
    colors = numpy.empty([len(pixels), 3], dtype=np.float32)

    # Extract color information at
    # pixel locations in image
    for i, (x, y) in enumerate(pixels):
        colors[i] = image[y][x]

    # Cluster color information to
    # find dominant colors in triangle
    average = colors.mean(axis=0)
    return average
"""
