import cv2
import math
import numpy as np

from operator import itemgetter
from itertools import groupby

import plotly.express as px
import plotly.graph_objects as go

from enum import Enum


class ColorSpace(Enum):
    RGB = 'RGB'
    HSV = 'HSV'
    HSL = 'HSL'

    def __str__(self):
        return self.value


def to_hsv_space(pixels):
    sines = [math.sin(math.radians(2 * a)) for a in range(0, 180)]
    cosines = [math.cos(math.radians(2 * a)) for a in range(0, 180)]

    for i, (h, s, v) in enumerate(pixels):
        pixels[i][0] = s * sines[int(h)]
        pixels[i][1] = s * cosines[int(h)]


def to_hsl_space(pixels):
    sines = [math.sin(math.radians(2 * a)) for a in range(0, 180)]
    cosines = [math.cos(math.radians(2 * a)) for a in range(0, 180)]

    for i, (h, l, s) in enumerate(pixels):
        pixels[i][0] = s * sines[int(h)]
        pixels[i][1] = s * cosines[int(h)]
        pixels[i][2] = l


def plot(image, space):
    # The incoming image RGB color data is deduplicated
    # and converted back to a 2D numpy array to be translated
    plot_colors = image.reshape((-1, 3))
    plot_colors = dict.fromkeys(map(tuple, plot_colors))
    plot_colors = np.array(list(map(np.array, plot_colors)))

    if space == ColorSpace.RGB:
        space_colors = plot_colors
    else:
        # Reshape RGB data array for vectorized
        # conversion into HSV/HSL color space
        plot_colors = plot_colors.reshape((len(plot_colors), 1, 3))
        match space:
            case ColorSpace.HSV:
                space_colors = cv2.cvtColor(plot_colors, cv2.COLOR_RGB2HSV)
            case ColorSpace.HSL:
                space_colors = cv2.cvtColor(plot_colors, cv2.COLOR_RGB2HLS)

        # Transform both colors and HSV/HSL data
        plot_colors = plot_colors.reshape((-1, 3))
        space_colors = space_colors.reshape((-1, 3))
        space_colors = np.float32(space_colors)

        # Transform into HSV/HSL cylinder
        match space:
            case ColorSpace.HSV:
                to_hsv_space(space_colors)
            case ColorSpace.HSL:
                to_hsl_space(space_colors)

    # Transpose coordinate matrix to read
    # x, y and z coordinates row-wise
    x, y, z = np.swapaxes(space_colors, 1, 0)
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=3,
                color=plot_colors,
            )
        )]
    )
    fig.show()


def show(image):
    figure = px.imshow(image)
    figure.show()


def to_space(image, space):
    if space == ColorSpace.RGB:
        return np.float32(image.reshape((-1, 3)))

    match space:
        case ColorSpace.HSV:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            data = np.float32(image.reshape((-1, 3)))
            to_hsv_space(data)
        case ColorSpace.HSL:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            data = np.float32(image.reshape((-1, 3)))
            to_hsl_space(data)
        case _:
            data = np.float32(image.reshape((-1, 3)))

    return data
