"""Linear-light conversion for alpha compositing."""

import numpy as np


def linear(im):
    x = np.asarray(im, np.float32) / 255
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.clip(
        np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055) * 255 + 0.5,
        0,
        255,
    ).astype(np.uint8)


def luma(rgb):
    return rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
