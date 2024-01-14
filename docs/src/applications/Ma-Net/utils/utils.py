import numpy as np


def label2colormap(label):

    m = label.astype(np.uint8)
    r, c = m.shape
    cmap = np.zeros((r, c, 3), dtype=np.uint8)
    cmap[:, :, 0] = (m & 1) << 7 | (m & 8) << 3 | (m & 64) >> 1
    cmap[:, :, 1] = (m & 2) << 6 | (m & 16) << 2 | (m & 128) >> 2
    cmap[:, :, 2] = (m & 4) << 5 | (m & 32) << 1
    return cmap
