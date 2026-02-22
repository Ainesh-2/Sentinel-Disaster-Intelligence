import matplotlib.pyplot as plt
import numpy as np


def generate_heatmap(density_matrix):
    plt.figure(figsize=(6, 6))
    plt.imshow(density_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Damage Density')
    plt.title('Damage Density Heatmap')
    plt.show()
