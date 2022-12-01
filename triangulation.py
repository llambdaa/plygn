import cv2
import math
import numpy as np

from numba import njit

TRIANGULATION_THICKNESS = 1
TRIANGULATION_COLOR = (0, 255, 0)


def find_vertices(contour_groups, preferred_distance):
    vertices = list()
    for contour_group in contour_groups:
        for contour in contour_group:
            # The contour must be long enough to
            # hold three distinct vertices
            length = len(contour)
            if length < 3:
                continue

            # Initially the vertex distance along a
            # contour line is equal to the preferred
            distance = preferred_distance

            # If the contour is too short to equally
            # distribute at least three vertices using
            # the preferred distance, a new distance
            # is calculated for that contour
            if length < 3 * distance:
                distance = max(1, math.floor(length / 3))

            # Theoretically, each distance-th point on the
            # contour line qualifies as a vertex for triangulation.
            # However, the last and first vertex can be placed
            # as far as 2 * distance - 2 positions along the line
            # away from each other.
            vertex_count = math.floor(length / distance)

            # That uneven distribution is resolved by sharing
            # the remaining positions after the last vertex
            # between all vertices, so that they are distributed
            # more evenly.
            remaining = length - vertex_count * distance
            shared = math.floor(remaining / vertex_count)
            distance += shared

            # Practically, the vertices are likely to be spaced
            # by more than the given distance, but the distance
            # still is the only parameter to determine the amount
            # of vertices along a contour line.vertices
            for i in range(1, vertex_count + 1):
                vertex = contour[(i * distance) - 1][0]
                x = int(vertex[0])
                y = int(vertex[1])
                vertices.append((x, y))

    return vertices


# @njit(cache=True, nogil=True)
def split_triangulation(triangulation, threshold):
    # The size of a triangle determines whether a
    # triangle must be split into smaller triangles.
    x1, y1, x2, y2, x3, y3 = triangulation.T
    sizes = np.abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) * 0.5)
    splits = np.ceil(np.log(sizes / threshold) / np.log(4)).astype(np.int32)
    splits = np.where(splits < 0, 0, splits)

    # The number of splits for each triangle determine
    # the total amount of triangles. Knowing it allows
    # for pre-allocating an array.
    bases = np.repeat(4, len(splits))
    count = np.sum(np.power(bases, splits))
    triangles = np.empty(shape=[count, 6], dtype=np.int32)
    index = 0

    # Each triangle is split according to its modifier.
    # Either it will not be split (=0) or a subroutine
    # is called, so that it is split (possibly) multiple
    # times before the smaller triangles are written back.
    for i, split in enumerate(splits):
        triangle = triangulation[i]
        if split == 0:
            # Trivial case - triangle will not be split,
            # but be written back directly
            triangles[index] = triangle
            index += 1

        elif split == 1:
            # The triangle is split one time, resulting
            # in four smaller triangles
            triangles[index:(index + 4)] = split_triangle(triangle)
            index += 4

        else:
            # The triangle is split multiple times,
            # resulting in 4^split smaller triangles
            current = split_triangle(triangle)
            for j in range(2, split + 1):
                # That container will hold the result of
                # splitting the currently inspected triangles
                new = np.empty((4 ** j, 6), dtype=np.int32)

                # Each currently inspected triangle is split into
                # four smaller triangles, which are written to the
                # collection holding the new triangles
                for k in range(len(current)):
                    offset = 4 * k
                    partial = split_triangle(current[k])
                    new[offset:(offset + 4)] = partial

                # For the next iteration, the currently inspected
                # triangles are the smaller triangles we just created
                current = new

            offset = 4 ** split
            triangles[index:(index + offset)] = current
            index += offset

    return triangles

@njit(cache=True, nogil=True)
def split_triangle(triangle):
    x1, y1, x2, y2, x3, y3 = triangle
    # The vectors between the original vertices
    # are used to build the splitting points
    dx12, dy12 = x2 - x1, y2 - y1
    dx23, dy23 = x3 - x2, y3 - y2
    dx31, dy31 = x1 - x3, y1 - y3

    # The splitting points 'a', 'b' and 'c' lie
    # directly between the original vertices.
    ax = x1 + int(dx12 * 0.5)
    ay = y1 + int(dy12 * 0.5)
    bx = x2 + int(dx23 * 0.5)
    by = y2 + int(dy23 * 0.5)
    cx = x3 + int(dx31 * 0.5)
    cy = y3 + int(dy31 * 0.5)

    # Using the splitting points,
    # four new triangles are built.
    first = [x1, y1, ax, ay, cx, cy]
    second = [ax, ay, x2, y2, bx, by]
    third = [cx, cy, bx, by, x3, y3]
    forth = [ax, ay, bx, by, cx, cy]
    return np.array([first, second, third, forth])


def find_triangulation(image_shape, vertices):
    height, width, _ = image_shape
    frame = cv2.Subdiv2D((0, 0, width, height))

    # Perform triangulation and
    # transform into more usable type
    for vertex in vertices:
        frame.insert(vertex)

    # Independent of the vertex method the
    # corners of the image are counted as
    # vertices too
    frame.insert((0, 0))
    frame.insert((0, height - 1))
    frame.insert((width - 1, 0))
    frame.insert((width - 1, height - 1))

    triangulation = np.int32(frame.getTriangleList())
    return triangulation


def show_triangulation(image, triangles, out_path):
    result = image.copy()
    for x1, y1, x2, y2, x3, y3 in triangles:
        a = (x1, y1)
        b = (x2, y2)
        c = (x3, y3)
        result = cv2.line(result, a, b, TRIANGULATION_COLOR, TRIANGULATION_THICKNESS)
        result = cv2.line(result, b, c, TRIANGULATION_COLOR, TRIANGULATION_THICKNESS)
        result = cv2.line(result, c, a, TRIANGULATION_COLOR, TRIANGULATION_THICKNESS)

    cv2.imwrite(
        f"{out_path}/triangulated.png",
        cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    )
