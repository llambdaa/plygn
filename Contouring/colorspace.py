import cv2
import math
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from enum import Enum


class ColorSpace(Enum):
    RGB = 'RGB'
    HSV = 'HSV'
    HSL = 'HSL'

    def __str__(self):
        return self.value


def deduplicate_colors(image):
    data = image.reshape((-1, 3))
    unique, counts = np.unique(data, axis=0, return_counts=True)
    return unique, counts


def to_hsv_cylinder(pixels):
    sines = [math.sin(math.radians(2 * a)) for a in range(0, 180 + 1)]
    cosines = [math.cos(math.radians(2 * a)) for a in range(0, 180 + 1)]

    for i, (h, s, v) in enumerate(pixels):
        pixels[i][0] = s * sines[int(h)]
        pixels[i][1] = s * cosines[int(h)]


def to_hsl_cylinder(pixels):
    sines = [math.sin(math.radians(2 * a)) for a in range(0, 180 + 1)]
    cosines = [math.cos(math.radians(2 * a)) for a in range(0, 180 + 1)]

    for i, (h, l, s) in enumerate(pixels):
        pixels[i][0] = s * sines[int(h)]
        pixels[i][1] = s * cosines[int(h)]
        pixels[i][2] = l


def to_hsv(rgb):
    rgb = rgb.reshape((len(rgb), 1, 3))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv = np.float32(hsv.reshape((-1, 3)))
    return hsv


def to_hsl(rgb):
    rgb = rgb.reshape((len(rgb), 1, 3))
    hsl = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
    hsl = np.float32(hsl.reshape((-1, 3)))
    return hsl


def plot(colors, points):
    # Transpose coordinate matrix to read
    # x, y and z coordinates row-wise
    x, y, z = np.swapaxes(points, 1, 0)
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=3,
                color=colors,
            )
        )]
    )
    fig.show()


def show(image):
    figure = px.imshow(image)
    figure.show()


def to_space(colors, space):
    match space:
        case ColorSpace.HSV:
            data = to_hsv(colors)
            to_hsv_cylinder(data)
        case ColorSpace.HSL:
            data = to_hsl(colors)
            to_hsl_cylinder(data)
        case _:
            return np.float32(colors)

    return data
