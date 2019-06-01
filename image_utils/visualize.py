import cv2
import numpy as np

from skimage import color


FLOW_ARROW_COLORS = [tuple(255 * int(bit) for bit in bin(i)[2:].rjust(3, '0')) for i in range(8)]


def vis_flow(flow, max_disp=256, skip=32):
    canvas = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    color_idx = 0

    for y in range(0, canvas.shape[0], skip):
        for x in range(0, canvas.shape[1], skip):
            if np.any(np.isnan(flow[y,x])):
                continue
            elif np.any(np.isinf(flow[y,x])):
                continue

            u = np.array(flow[y,x])
            dx, dy = u / (np.linalg.norm(u) + 1e-5) * min(np.linalg.norm(u) + 1e-5, max_disp)

            cv2.arrowedLine(
                    canvas, (x, y), (int(round(x + dx)), int(round(y + dy))),
                    FLOW_ARROW_COLORS[color_idx], 1)

            color_idx = (color_idx + 1) % len(FLOW_ARROW_COLORS)

    return canvas


def visualize_flow_colors(flow):
    height, width, _ = flow.shape
    norms = np.sqrt(flow[:,:,0] ** 2 + flow[:,:,1] ** 2 + 1e-8)

    hsv = np.ones((height, width, 3), dtype=np.float32)
    hsv[:,:,0] = (np.arctan2(flow[:,:,1], flow[:,:,0]) / np.pi + 1.0) / 2.0
    hsv[:,:,1] = np.minimum(1.0, norms / 100.0)

    rgb = color.hsv2rgb(hsv)
    rgb[np.any(np.isnan(flow), axis=2)] = 0.0

    return rgb


def vis_normals(normals):
    u = normals[:,:,0].copy()
    v = normals[:,:,1].copy()
    w = normals[:,:,2].copy()
    normals[:,:,0] = (u + 1.0) / 2.0 * 255.0
    normals[:,:,1] = (v + 1.0) / 2.0 * 255.0
    normals[:,:,2] = w * -127.0 + 128.0

    return np.uint8(np.round(normals))
