import numpy as np
import scipy.ndimage.filters as filters


def visualize_normals(normals):
    normals[:,:,0] = (normals[:,:,0] + 1.0) / 2.0 * 255.0
    normals[:,:,1] = (normals[:,:,1] + 1.0) / 2.0 * 255.0
    normals[:,:,2] = normals[:,:,2] * -127.0 + 128.0

    return np.uint8(np.round(normals))


def get_normals(depth, mult=100.0/14.0):
    padded_depth = depth * mult
    padded_depth = np.pad(padded_depth, 1, mode='edge')

    u = (padded_depth[1:-1,2:] - padded_depth[1:-1,:-2]) / 2.0
    v = (padded_depth[2:,1:-1] - padded_depth[:-2,1:-1]) / 2.0

    normals = np.stack([-u, -v, -np.ones(depth.shape)], -1)
    normals = normals / np.linalg.norm(normals, axis=2)[:,:,np.newaxis]

    return normals
