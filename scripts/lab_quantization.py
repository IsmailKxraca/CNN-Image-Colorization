"""
In this script I will quantize the ab colorspace into discrete classes.
I will use binning and clustering to identify which colors get used in reality and which are not.

I will code functions for extracting the ab values of a Lab-image and classify them into the color-classes.
"""
import numpy as np

# parameters for bins
bin_size = 10
min_val = -128
max_val = 127


def split_lab_channels(labimg):
    l_channel = labimg[:, :, 0]
    a_channel = labimg[:, :, 1]
    b_channel = labimg[:, :, 2]

    return l_channel, a_channel, b_channel


def lab_bins(min_val = -128, max_val = 127, bin_size = 10):
    # creating the centers of the bins;
    grid_points = np.arange(min_val + bin_size / 2, max_val, bin_size)
    ab_grid = np.array(np.meshgrid(grid_points, grid_points)).T.reshape(-1, 2)
    # two matrices with 313 values. One matrix for x and other for y values. Makes the grid
    return ab_grid


def find_nearest_bin(a_channel, b_channel, ab_grid):
    # creates a vector with every pixel as ((a1,b1),(a2,b2) ...)
    ab_values = np.stack((a_channel, b_channel), axis=-1).reshape(-1, 2)

    # calculates the Euclidean distance between every pixel and every bin-center and saves it in a new dimension
    distances = np.linalg.norm(ab_values[:, None, :] - ab_grid[None, :, :], axis=-1)

    # finds the nearest bin-center for every pixel. vector with bin-center of every class (256x256)
    nearest_bins = np.argmin(distances, axis=1)
    return nearest_bins.reshape(a_channel.shape)


ab_grid = lab_bins(min_val, max_val, bin_size)

