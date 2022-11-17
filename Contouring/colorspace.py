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


def dedupe_colors(image):
    # Transform color channels (R, G, B)
    # into integers for faster differentiation
    r, g, b = cv2.split(image)
    combined = (np.int32(b) << 16) + (np.int32(g) << 8) + np.int32(r)
    combined = combined.reshape(-1)

    # Determine unique colors (ints)
    # and their frequency
    unique, counts = np.unique(combined, return_counts=True)

    # Transform back into channels
    unique = unique.view(np.uint8).reshape(unique.shape + (4,))[..., :3]
    return unique, counts


def to_hsv_cylinder(pixels):
    # Transpose pixel matrix and
    # split into HSL channels
    length = len(pixels)
    pixels = np.swapaxes(pixels, 1, 0)
    h, s, v = pixels.copy()

    # Translate Hue values into
    # radians and calculate (co)sine
    h = np.multiply(h, 2 * (math.pi/180))
    sines = np.sin(h)
    cosines = np.cos(h)

    # Calculate HSL coordinates
    pixels[0] = [s[i] * sines[i] for i in range(length)]
    pixels[1] = [s[i] * cosines[i] for i in range(length)]

    # Transform pixel matrix back
    pixels = np.swapaxes(pixels, 1, 0)
    return pixels


def to_hsl_cylinder(pixels):
    # Transpose pixel matrix and
    # split into HSL channels
    length = len(pixels)
    pixels = np.swapaxes(pixels, 1, 0)
    h, l, s = pixels.copy()

    # Translate Hue values into
    # radians and calculate (co)sine
    h = np.multiply(h, 2 * (math.pi/180))
    sines = np.sin(h)
    cosines = np.cos(h)

    # Calculate HSL coordinates
    pixels[0] = [s[i] * sines[i] for i in range(length)]
    pixels[1] = [s[i] * cosines[i] for i in range(length)]
    pixels[2] = l

    # Transform pixel matrix back
    pixels = np.swapaxes(pixels, 1, 0)
    return pixels


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
            data = to_hsv_cylinder(data)
        case ColorSpace.HSL:
            data = to_hsl(colors)
            data = to_hsl_cylinder(data)
        case _:
            return np.float32(colors)

    return data
