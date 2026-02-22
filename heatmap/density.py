import numpy as np


def compute_grid_density(mask, grid_size=20):

    h, w = mask.shape
    cell_h = h // grid_size
    cell_w = w // grid_size

    density_matrix = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            y1 = cell_h * i
            y2 = cell_h * (i + 1)
            x1 = cell_w * j
            x2 = cell_w * (j + 1)

            cell = mask[y1:y2, x1:x2]
            density_matrix[i, j] = np.sum(cell) / cell.size

    return density_matrix
