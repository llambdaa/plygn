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

    # The indices of duplicate difference representations of
    # triangles are grouped.
    groups = group_duplicates(differences)
    
    """
    1) For the unique triangles (always first element of a group)
       the offsets that lie within the triangle must be found.
       That can be done by determining a bounding box (or a sequence
       of indices) that fully contains all possible pixels inside a
       triangle. Then, using vectorized operations, the barycentric
       coordinate scalars can be computed. Using a logical operation
       using numpy, the locations in the window where certain conditions
       are met, are collected into a list.
    
    2) Now, after finding all 1D pixel locations offset from the lowest
       index of a triangle, a first lookup table is built. At each index
       of a unique triangle, it holds a list or array of those offsets of
       this very triangle.
    
    3) A second lookup table is now being built. For each triangle (or its
       representation) an array entry is created. It holds the index of the
       corresponding unique triangle, of which it is a duplicate. The
       information which unique triangle represents which duplicates is stored
       in the 'groups' data structure. 
       
    4) Then, another array is constructed. It holds for each triangle at its
       index another array or list that contains the offsets starting from the
       minimum index of the triangle. That can be calculated using vectorized
       operations. Now, we have a huge 2D data structure that contains all
       concrete indices in the 1D image for each triangle, where the pixel data
       shall be retrieved from and be replaced at.
    
    5) The array is used to retrieve the pixel values at those indices using
       vectorized operations. Either that array stays 2D (slower access) or
       it is converted to one 1D array, but then the resulting pixel value
       array must be split according to the amount of pixels within one triangle
       in order to group the retrieved pixel values by triangle again.
       Theoretically, the amount of pixels inside a triangle is known.
    
    6) The averages of those pixel groups is calculated. That average will
       occur as many times as there are indices for a triangle, so that using
       the 1D version of the index array, the colors can be written directly into
       the image (numpy.put)
       
    7) The image is converted back to 2D, so that is can be written out.
    """


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


def group_duplicates(differences):
    # First, the input array is sorted, returning another
    # array of the original indices that the sorted elements
    # had in the input array. Then, the input array is sorted
    # using these very indices.
    sorted_indices = np.argsort(differences)
    sorted_differences = differences[sorted_indices]

    # Then, the array is deduplicated and the indices of the
    # unique values' first occurrences along with their count
    # are returned.
    _, index_starts, _ = np.unique(sorted_differences, return_index=True, return_counts=True)

    # Lastly, the index array is split along those delimiters.
    # The fragments will hold the indices of the sorted elements
    # into the original input array.
    groups = np.split(sorted_indices, index_starts[1:])
    return groups
