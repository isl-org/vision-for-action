import numpy as np


def identity_func(x):
    return x


def normalize_func(scale, offset):
    def func(x):
        return x / scale - offset

    return func


def normalize_flow(max_displacement):
    def func(x):
        x = np.clip(x, -max_displacement, max_displacement)
        x = x / (2.0 * max_displacement)
        x[np.isnan(x)] = 0.0

        return x

    return func


def clip_and_scale(max_val):
    def func(x):
        x = np.clip(x, 0, max_val)
        x = x / max_val

        return x

    return func
