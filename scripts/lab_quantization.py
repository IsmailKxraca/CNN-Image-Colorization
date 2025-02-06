"""
In this script I will quantize the ab colorspace into discrete classes.
I will use binning and clustering to identify which colors get used in reality and which are not.

I will code functions for extracting the ab values of a Lab-image and classify them into the color-classes.
"""
import numpy as np
from skimage.color import lab2rgb
import matplotlib.pyplot as plt
import cv2


# parameters for bins
bin_size = 10
min_val = -110
max_val = 110


def split_lab_channels(labimg):
    l_channel = labimg[:, :, 0]
    a_channel = labimg[:, :, 1]
    b_channel = labimg[:, :, 2]

    return l_channel, a_channel, b_channel


def lab_bins(min_val = -110, max_val = 110, bin_size = 10):
    """
    this function bins the LAB-space
    :param min_val: the minimum value of the colorspace
    :param max_val: the maximum value of the colorspace
    :param bin_size: size of the bins
    :return: an Array with the center of the bins in shape of (484, 2) (with default params)
    """
    grid_points = np.arange(min_val + bin_size / 2, max_val, bin_size)
    ab_grid = np.array(np.meshgrid(grid_points, grid_points)).T.reshape(-1, 2)
    return ab_grid


def find_nearest_bin(a_channel, b_channel, ab_grid):
    # creates a vector with every pixel as ((a1,b1),(a2,b2) ...)
    ab_values = np.stack((a_channel, b_channel), axis=-1).reshape(-1, 2)

    # calculates the Euclidean distance between every pixel and every bin-center and saves it in a new dimension
    distances = np.linalg.norm(ab_values[:, None, :] - ab_grid[None, :, :], axis=-1)

    # finds the nearest bin-center for every pixel. vector with bin-center of every class (256x256)
    nearest_bins = np.argmin(distances, axis=1)
    return nearest_bins.reshape(a_channel.shape)


def visualize_grid_colours(ab_grid, L=50):
    """
    visualizes the Colours of the bins-(centers)
    With L=50
    """
    l_values = np.full((ab_grid.shape[0], 1), L)
    lab = np.hstack((l_values, ab_grid))

    # convert lab to rgb for illustration of colors
    rgb = lab2rgb(lab.reshape(-1, 1, 3)).reshape(-1, 3)

    # extract a and b values
    a_values = ab_grid[:, 0]
    b_values = ab_grid[:, 1]

    # create diagram
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    marker_size = (10 / (ax.get_xlim()[1] - ax.get_xlim()[0]) * 2)**2

    # plot
    plt.scatter(a_values, b_values, c=rgb, s=marker_size, marker='s', edgecolor='black')
    ax.set_aspect('equal')
    plt.title("Farben der Bin-Zentren im AB-Raum")
    plt.xlabel("A-Wert")
    plt.ylabel("B-Wert")
    plt.grid(False)

    # plot diagram
    plt.show()

def quantize_ab(ab_image, ab_grid):
    # Reshape des Bildes für einfachere Berechnungen
    H, W, _ = ab_image.shape
    ab_flat = ab_image.reshape(-1, 2)  # (H*W, 2)

    # Berechne die euklidische Distanz zu jedem Bin (für alle Pixel)
    distances = np.linalg.norm(ab_flat[:, None, :] - ab_grid[None, :, :], axis=2)  # (H*W, 484)

    # Finde den Index des nächsten ab_grid-Bins (d.h. die Klasse)
    labels = np.argmin(distances, axis=1)  # (H*W,)

    return labels.reshape(H, W)  # (H, W)

ab_grid = lab_bins(min_val, max_val, bin_size)
print(ab_grid.shape)
#visualize_grid_colours(ab_grid)

test = np.load("000f2ebfe51758.jpg.npy")
rgb_image = cv2.cvtColor(test, cv2.COLOR_LAB2RGB)

# Anzeige des RGB-Bildes mit Matplotlib
plt.imshow(rgb_image)
plt.axis('off')  # Achsen ausblenden
plt.show()

#probleme: LAB images wurden mit OpenCv transformiert: OpenCv speichert in unit8 form 0-255