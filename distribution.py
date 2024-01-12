import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = 'data/'

FIRE_DISTANCES_PATH = DATA_PATH + 'fire_distances.csv'
BUILDING_DISTANCES_PATH = DATA_PATH + 'building_distances.csv'
DISTRIBUTION_PLOT_PATH = 'distribution.png'

MAX_DISTANCE = 1000
NUM_RANGES = 20

def print_distribution(p_distribution, bin_edges):
    for i, p in enumerate(p_distribution):
        print(f'{bin_edges[i]:3} - {bin_edges[i+1]:4} m: {p:.5f}')

def plot_distribution(p_distribution, bin_edges):
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_centres, p_distribution)
    plt.xticks(bin_edges, rotation=70)

    plt.xlabel('distance [m]')
    plt.title('probability of propagating fire')

def main():
    fire_distances = np.loadtxt(FIRE_DISTANCES_PATH)
    building_distances = np.loadtxt(BUILDING_DISTANCES_PATH)

    bin_edges = np.arange(0, MAX_DISTANCE+1, step=MAX_DISTANCE / NUM_RANGES, dtype=np.int32)

    buildings_histogram, _ = np.histogram(building_distances, bins=bin_edges)
    fires_histogram, _ = np.histogram(fire_distances, bins=bin_edges)

    p_distribution = 0.5 * fires_histogram / buildings_histogram

    print_distribution(p_distribution, bin_edges)

    plot_distribution(p_distribution, bin_edges)
    plt.savefig(DISTRIBUTION_PLOT_PATH)
    plt.show()

if __name__ == "__main__":
    main()