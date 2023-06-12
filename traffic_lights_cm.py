# coding: utf-8
import numpy as np
import colorcet as cc
from matplotlib.colors import LinearSegmentedColormap

cm_data = np.asarray(cc.rainbow_bgyr_35_85_c72)
assert cm_data.shape == (256, 3)
cm_data = cm_data[cm_data.shape[0] * 20 // 80 :]
traffic_lights = LinearSegmentedColormap.from_list("traffic_lights", cm_data)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    xs, _ = np.meshgrid(np.linspace(0, 1, 80), np.linspace(0, 1, 10))
    plt.imshow(xs, cmap=traffic_lights)
    plt.show()
