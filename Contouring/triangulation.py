import cv2
import math

from contour import *
from enum import Enum

TRIANGULATION_THICKNESS = 1
TRIANGULATION_COLOR = (0, 255, 0)


class VertexMethod(Enum):
    EQUAL_SPACE = 0
    VARIANCE = 1
    RANDOM = 2


def find_vertices_equal_space(contour_groups, preferred_distance):
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
