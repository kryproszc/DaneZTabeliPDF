from sklearn.neighbors import KDTree
from itertools import combinations
from tqdm import tqdm
import pandas as pd
import numpy as np
import networkx as nx
import pyproj

DATA_PATH = 'data/'

BUILDINGS_PATH = DATA_PATH + 'buildings.csv'
FIRES_PATH = DATA_PATH + 'fires.csv'
# BUILDINGS_PATH = 'data/buildings_test.csv'
# FIRES_PATH = 'data/fires_test.csv'

FIRE_DISTANCES_PATH = DATA_PATH + 'fire_distances.csv'
BUILDING_DISTANCES_PATH = DATA_PATH + 'building_distances.csv'

MAX_DISTANCE = 1000
ENABLE_PROJECTION = True

def convert_coordinates(points):
    if not ENABLE_PROJECTION:
        return points
    
    source_crs = 'epsg:4326' # EPSG:4326 - standard geographic coordinate system
    target_crs = 'epsg:2180' # EPSG:2180 - projection for Poland (Transverse Mercator)
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs)

    latitudes = points[:,0]
    longitudes = points[:,1]
    results = transformer.transform(latitudes, longitudes)
    return np.column_stack(results)

def euclidean_mst(points):
    # create a graph
    G = nx.Graph()

    # add nodes to the graph
    for i, point in enumerate(points):
        G.add_node(i, pos=(point[0], point[1]))

    # compute the distances between points and add edges to the graph
    for (i, u), (j, v) in combinations(enumerate(points), 2):
        distance = ((u[0] - v[0])**2 + (u[1] - v[1])**2)**0.5
        G.add_edge(i, j, weight=distance)

    # calculate the Minimum Spanning Tree
    mst = nx.minimum_spanning_tree(G)

    # get edges of the MST
    mst_edges = list(mst.edges(data=True))

    # convert edges to indices of points with distances
    result = [[] for _ in range(len(points))]
    for edge in mst_edges:
        i, j, dist = edge
        result[i].append((j, dist['weight']))
        result[j].append((i, dist['weight']))

    # sort lists increasingly by distance
    for neighbors in result:
        neighbors.sort(key=lambda x: x[1])

    return result

def get_buildings_within_distance(kd_tree, point, distance):
    _, distances = kd_tree.query_radius([point], r=distance, return_distance=True)
    return distances[0]

def main():
    buildings_df = pd.read_csv(BUILDINGS_PATH)
    fires_df = pd.read_csv(FIRES_PATH)

    print(f'loaded {buildings_df.shape[0]} buildings')
    print(f'loaded {fires_df.shape[0]} fires')

    buildings = buildings_df.to_numpy()
    buildings = convert_coordinates(buildings)
    kd_tree = KDTree(buildings)

    building_distances = []
    fire_distances = []

    for _, day_fires in tqdm(fires_df.groupby('date')):
        day_fires = day_fires[['latitude', 'longitude']].to_numpy()
        day_fires = convert_coordinates(day_fires)

        # build Minimum Spanning Tree of all fires on a particular day
        fires_mst = euclidean_mst(day_fires)

        for fire, nearest_fires in zip(day_fires, fires_mst):
            # find all buildings to which fire could propagate
            new_fire_distances = [dist for _, dist in nearest_fires if dist < MAX_DISTANCE]
            fire_distances.extend(new_fire_distances)

            # find all buildings in proximity
            new_building_distances = get_buildings_within_distance(kd_tree, fire, MAX_DISTANCE)
            building_distances.extend(new_building_distances)

    fire_distances.sort()
    building_distances.sort()

    np.savetxt(FIRE_DISTANCES_PATH, fire_distances, fmt='%f')
    np.savetxt(BUILDING_DISTANCES_PATH, building_distances, fmt='%f')

    print(f'saved {len(fire_distances)} fire distances')
    print(f'saved {len(building_distances)} building distances')

if __name__ == "__main__":
    main()
