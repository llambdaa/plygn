import cv2
import numpy as np

from numba import njit


@njit(cache=True, nogil=True)
def colorize(image, triangulation):
    canvas = image.copy()

    # Triangle coordinates are converted to discrete
    # integer values and bounding boxes are calculated.
    triangulation = triangulation.astype(np.int32)
    xmin, ymin, xmax, ymax, width, height = find_bounding_boxes(triangulation)

    # Vectorized calculation of only triangle-dependent
    # components for barycentric coordinate calculations
    v0x, v0y, v1x, v1y, den = find_barycentric_components(triangulation)

    for i, (t_ax, t_ay, t_bx, t_by, t_cx, t_cy) in enumerate(triangulation):
        t_xmin, t_xmax = xmin[i], xmax[i]
        t_ymin, t_ymax = ymin[i], ymax[i]
        t_width, t_height = width[i], height[i]
        t_v0x, t_v0y = v0x[i], v0y[i]
        t_v1x, t_v1y = v1x[i], v1y[i]
        t_den = den[i]

        # The matrices 'xs' and 'ys' have the same dimensions
        # as the triangle's bounding box and hold the x- and
        # y-coordinates of the points inside it, respectively.
        xs, ys = make_coordinate_matrices(t_xmin, t_xmax, t_ymin, t_ymax, t_width, t_height)
        ones = np.ones((t_height, t_width))

        # Now, for each point inside the bounding box (or both
        # coordinate matrices), calculate the decision values.
        t_v2x = xs - t_ax
        t_v2y = ys - t_ay
        v = (t_v2x * t_v1y - t_v2y * t_v1x) / t_den
        w = (t_v2y * t_v0x - t_v2x * t_v0y) / t_den
        u = ones - v - w

        # Each position in the bounding box, where the condition
        # is met (so that the point lies within the triangle), is
        # iterated, collecting the color values into separate
        # accumulators. Then, the color average is calculated.
        r_total, g_total, b_total, size = 0, 0, 0, 0
        for ix, x in enumerate(range(t_xmin, t_xmax + 1)):
            for iy, y in enumerate(range(t_ymin, t_ymax + 1)):
                if 0 <= u[iy][ix] <= 1:
                    r, g, b = image[y][x]
                    r_total += r
                    g_total += g
                    b_total += b
                    size += 1

        r_avg = int(r_total / size)
        g_avg = int(g_total / size)
        b_avg = int(b_total / size)
        color = np.array([r_avg, g_avg, b_avg])

        # The average color is then written
        # to each point in the triangle.
        for ix, x in enumerate(range(t_xmin, t_xmax + 1)):
            for iy, y in enumerate(range(t_ymin, t_ymax + 1)):
                if 0 <= u[iy][ix] <= 1:
                    canvas[y][x] = color

    return canvas


@njit(cache=True, nogil=True)
def find_bounding_boxes(triangulation):
    x1, y1, x2, y2, x3, y3 = triangulation.T
    xmin = np.minimum(x1, np.minimum(x2, x3))
    xmax = np.maximum(x1, np.maximum(x2, x3))
    ymin = np.minimum(y1, np.minimum(y2, y3))
    ymax = np.maximum(y1, np.maximum(y2, y3))
    width = xmax - xmin + 1
    height = ymax - ymin + 1

    return xmin, ymin, xmax, ymax, width, height


@njit(cache=True, nogil=True)
def find_barycentric_components(triangulation):
    ax, ay, bx, by, cx, cy = triangulation.T
    v0x = bx - ax
    v0y = by - ay
    v1x = cx - ax
    v1y = cy - ay
    den = v0x * v1y - v1x * v0y

    return v0x, v0y, v1x, v1y, den


@njit(cache=True, nogil=True)
def make_coordinate_matrices(xmin, xmax, ymin, ymax, width, height):
    ys = np.repeat(np.arange(ymin, ymax + 1), width).reshape((-1, width))
    xs = np.zeros((height, width), dtype=np.int32)
    for j in range(height):
        xs[j] = np.arange(xmin, xmax + 1)

    return xs, ys
