"""Analysis of the closest IXPs."""

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from sklearn.neighbors import BallTree, DistanceMetric

from lib.latency_lib import haversine_to_km

DEFAULT_FILE = "data/ixp_geolocation.csv"


def ixps_ball_tree(ixps_file):
    ixps = pd.read_csv(ixps_file)
    lats = np.asarray(ixps['lat']).reshape((-1, 1))
    lngs = np.asarray(ixps['lng']).reshape((-1, 1))
    coords = np.hstack((lats, lngs))
    tree = BallTree(np.deg2rad(coords),
                    metric=DistanceMetric.get_metric("haversine"))
    return tree, (lats, lngs)


def nn_distance(nn_tree, lats, lngs=None):
    """Get the distances of the provided points from the nearest ixp."""
    if lngs is not None:
        lats = np.asarray(lats).reshape(-1, 1)
        lngs = np.asarray(lngs).reshape(-1, 1)
        points = np.hstack((lats, lngs))
    else:
        points = np.asarray(lats).reshape(-1, 2)

    dist, neighbors = nn_tree.query(np.deg2rad(points))
    # Distance to kilometers, on the surface of the earth
    dist = haversine_to_km(dist)
    return dist, neighbors


# =============================================================================
# Old stuff to sample ground stations, remade

NO_DATA = -9999
Y_CORNER = 'yllcorner'
X_CORNER = 'xllcorner'
N_ROWS = 'nrows'
N_COLS = 'ncols'
CELLSIZE = 'cellsize'
GST = 'gst'
LANDING = 'landing'


def load_gdp_distribution_data(data_file):
    """Load the GDP distribution data."""
    header = parse_data_header(data_file)
    with open(data_file, 'r') as data:
        data = np.genfromtxt(data, delimiter=' ', skip_header=6)
    # Set the areas in which there is no data to 0
    data[data == header['NODATA_value']] = 0
    return header, data


def data_to_probability(matrix, filter=True):
    """Transform the data matrix into a probability distribution."""
    probability = matrix / np.sum(matrix)
    if filter:
        probability = uniform_filter(probability, 5)
        probability[probability < 0.0] = 0
        probability[matrix < 0.0] = 0
    return probability


def sample_gst_coordinates(n_gsts, gdp_matrix, gdp_header, replace=False,
                           randomize=False, filter_prob=False):
    """Sample GSTs coordinates form the gdp_matrix."""
    gdp_matrix = gdp_matrix.copy()
    gdp_matrix[gdp_matrix < 0] = 0
    probability = data_to_probability(gdp_matrix, filter_prob).flatten()
    positions = np.arange(0, gdp_matrix.shape[0] * gdp_matrix.shape[1])
    assert probability.shape == positions.shape
    gst_positions = np.random.choice(positions, size=n_gsts, replace=replace,
                                     p=probability)
    # Before returning, put again into matrix shape
    positions = np.unravel_index(gst_positions, dims=gdp_matrix.shape)
    lat = get_lat(positions[0], gdp_header, randomize)
    lng = get_lon(positions[1], gdp_header, randomize)
    return lat, lng


def get_lat(idx, header, randomize=False):
    """Get latitude from index in mat."""
    lat = header[Y_CORNER] + header[N_ROWS] - idx - 0.5
    if randomize:
        lat += np.random.normal(-0.5, 1, size=lat.shape)
    return lat


def get_lon(idx, header, randomize=False):
    lon = header[X_CORNER] + idx + 0.5
    if randomize:
        lon += np.random.normal(-0.5, 0.5, size=lon.shape)
    return lon


def parse_data_header(data_file):
    header = {}
    with open(data_file, 'r') as data:
        for _ in range(6):
            line = data.readline()
            split = line.split()
            header.update({split[0]: float(split[1])})
    return header


if __name__ == "__main__":
    tree, _ = ixps_ball_tree(DEFAULT_FILE)
    dist = nn_distance(tree, [0, 10], [0, 10])
    print(dist)
