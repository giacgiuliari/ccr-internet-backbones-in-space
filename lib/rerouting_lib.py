"""Implementation of the first-hop rerouting black-box model."""

import networkx as nx
import numpy as np
import pandas as pd
from geopy.distance import great_circle, EARTH_RADIUS
from scipy.spatial.qhull import Delaunay
from sklearn.neighbors.ball_tree import BallTree
from sklearn.neighbors.dist_metrics import DistanceMetric

from lib.latency_lib import LIGHT_IN_FIBER, LIGHT_IN_VACUUM, \
    create_baseline_graph, generator_to_matrix, compute_threshold, \
    haversine_to_km_altitude, FIBER_PATH_STRETCH, haversine_to_km, \
    gsts_optimization, make_test_constellation


def optimize_end_to_end_latency_rerouting(sat_pos, altitude, gst_pos, src_pos,
                                          min_elev, orbits, sat_per_orbit,
                                          terrestrial_gst_graph, inactive):
    # Compute satellite graph distances
    sat_sat_dist = compute_sat_sat_distance(sat_pos, altitude, orbits,
                                            sat_per_orbit)

    # Compute the BallTree for the satellites. This gives nn to satellites.
    sat_tree = BallTree(np.deg2rad(sat_pos),
                        metric=DistanceMetric.get_metric("haversine"))

    # Get the satellites that are in reach for the ground stations
    #   and their distance.
    sat_gst_ind, sat_gst_dist = compute_gst_sat_distance(altitude, min_elev,
                                                         gst_pos,
                                                         sat_tree)

    # Get the terrestrial GST -> GST distance
    gst_gst_terrestrial = gst_gst_terrestrial_distance(terrestrial_gst_graph,
                                                       gst_pos)

    # Get the satellite GST -> GST distance
    gst_gst_satellite = gsts_optimization(sat_gst_ind, sat_gst_dist,
                                          sat_sat_dist, n_gsts=gst_pos.shape[0])

    # Compute the closest active GST to the inactive ones
    nearest_active, nearest_active_dist = inactive_to_closest_active(
        inactive,
        gst_gst_terrestrial)

    # Get the closest GST to every source and its distance
    src_gst_ind, src_gst_dist = src_nearest_gst_distance(src_pos, gst_pos)

    # Put all together and get the src-dst distance matrix
    n_src = src_pos.shape[0]
    src_dst_latency = compute_src_dst_latency(n_src, inactive, src_gst_ind,
                                              src_gst_dist, nearest_active,
                                              nearest_active_dist,
                                              gst_gst_satellite)

    return src_dst_latency, nearest_active


def compute_sat_sat_distance(sat_pos, altitude, orbits, sat_per_orbit):
    sat_graph = create_baseline_graph(sat_pos, altitude, orbits, sat_per_orbit)
    sat_sat_dist = nx.all_pairs_dijkstra_path_length(sat_graph, weight='length')
    sat_sat_dist = dict(sat_sat_dist)
    sat_sat_dist = generator_to_matrix(sat_sat_dist, sat_pos.shape[0])
    return sat_sat_dist


def compute_gst_sat_distance(altitude, min_elev, gst_pos, sat_tree):
    max_sat_dist, max_radius = compute_threshold(altitude, min_elev)
    max_haversine = max_radius / EARTH_RADIUS

    # For each GST, query all the sat neighbors inside communication radius.
    sat_gst_ind, sat_gst_dist = sat_tree.query_radius(np.deg2rad(gst_pos),
                                                      max_haversine,
                                                      return_distance=True,
                                                      sort_results=True)
    dist_list = []
    for id, cur in enumerate(sat_gst_dist):
        dist = haversine_to_km_altitude(cur, altitude)
        dist_list.append(dist)
    sat_gst_dist = np.asarray(dist_list)

    return sat_gst_ind, sat_gst_dist


def src_nearest_gst_distance(src_pos, gst_pos, nn=1):
    """INCLUDES PATH STRETCH"""
    gst_tree = BallTree(np.deg2rad(gst_pos),
                        metric=DistanceMetric.get_metric("haversine"))
    src_gst_dist, src_gst_ind = gst_tree.query(np.deg2rad(src_pos), k=nn)
    src_gst_dist = haversine_to_km(src_gst_dist)
    src_gst_dist = src_gst_dist * FIBER_PATH_STRETCH
    return src_gst_ind, src_gst_dist


def delaunay_pos_graph(pos):
    """Compute a graph containing the Delaunay triangulation of the positons."""
    # Duplicate the positions left and right
    # Needed for the triangulation to wrap around earth
    new_pos = pos.copy()
    duplicate_right = pos.copy()
    duplicate_right[:, 1] = duplicate_right[:, 1] + 360
    new_pos = np.vstack((new_pos, duplicate_right))

    duplicate_left = pos.copy()
    duplicate_left[:, 1] = duplicate_left[:, 1] - 360
    new_pos = np.vstack((new_pos, duplicate_left))

    def reconvert_node(node_num, n_ixps):
        """Convert from the position in the duplicate matrix to the original."""
        return node_num - n_ixps * (node_num // n_ixps)

    # Make the Delaunay triangulation
    triangulation = Delaunay(new_pos)

    # Load the links from the triangles in a graph, with geographic distances
    n_pos = pos.shape[0]  # 578
    G = nx.Graph()
    G.add_nodes_from(np.arange(n_pos))

    for cur_tri in triangulation.simplices:
        a1, b1, c1 = cur_tri
        a = reconvert_node(a1, n_pos)
        b = reconvert_node(b1, n_pos)
        c = reconvert_node(c1, n_pos)
        ab = great_circle(pos[a, :], pos[b, :]).km
        bc = great_circle(pos[b, :], pos[c, :]).km
        ca = great_circle(pos[c, :], pos[a, :]).km
        if not ab > 0:
            print(f"{a} - {b} is {ab} in {a1, b1, c1}")
            # raise ValueError(f"{a} - {b} is {ab} in {a1, b1, c1}")
        if not bc > 0:
            print(f"{b} - {c} is {bc} in {a1, b1, c1}")
            # raise ValueError(f"{b} - {c} is {bc} in {a1, b1, c1}")
        if not ca > 0:
            print(f"{c} - {a} is {ca} in {a1, b1, c1}")
            # raise ValueError(f"{c} - {a} is {ca} in {a1, b1, c1}")
        G.add_edge(a, b, length=ab)
        G.add_edge(b, c, length=bc)
        G.add_edge(c, a, length=ca)

    return G


def gst_gst_terrestrial_distance(terrestrial_gst_graph, gst_pos):
    """Compute the distance between ground station of the terrestrial graph.

    AT THE ENDS INCLUDES PATH STRETCH.
    """
    gst_gst_terrestrial = nx.all_pairs_dijkstra_path_length(
        terrestrial_gst_graph,
        weight='length')
    gst_gst_terrestrial = dict(gst_gst_terrestrial)
    gst_gst_terrestrial = generator_to_matrix(gst_gst_terrestrial,
                                              gst_pos.shape[0])
    gst_gst_terrestrial = gst_gst_terrestrial * FIBER_PATH_STRETCH
    return gst_gst_terrestrial


def inactive_to_closest_active(inactive, gst_gst_terrestrial):
    """Compute the closest active groundstations and their distance.

    Args:
        inactive: list of indices of the gsts that are inactive.
        gst_gst_terrestrial: matrix of pairwise distances between gsts on the
            graph of land connections. PATH STRETCH ALREADY INCLUDED.
    """
    nearest_active = []
    nearest_active_dist = []
    for cur_in in inactive:
        closest = np.argsort(gst_gst_terrestrial[cur_in, :])
        found = False
        idx = 1
        while not found:
            if closest[idx] not in inactive:
                found = True
                nearest_active.append(closest[idx])
                nearest_active_dist.append(
                    gst_gst_terrestrial[cur_in, closest[idx]])
            else:
                # print(f"{cur_in} and {closest[idx]} are both inactive")
                idx += 1

    nearest_active_dist = np.asarray(nearest_active_dist)
    return nearest_active, nearest_active_dist


def compute_src_dst_latency(n_src, inactive, src_gst_ind, src_gst_dist,
                            nearest_active, nearest_active_dist, gst_gst_dist):
    src_dst_latency = np.zeros((n_src, n_src))

    # src_gst_time = src_gst_dist / LIGHT_IN_FIBER
    gst_gst_time = gst_gst_dist / LIGHT_IN_VACUUM

    gst_gst_time_rerouted = add_rerouting_to_inactive(inactive, nearest_active,
                                                      nearest_active_dist,
                                                      gst_gst_time)

    for src in range(n_src):
        # print(f"Currently doing {src} / {src_pos.shape[0]}\r", end="")
        for dst in range(src + 1, n_src):
            gst_src = src_gst_ind[src][0]
            gst_dst = src_gst_ind[dst][0]

            gst_src_dist = src_gst_dist[src][0]
            gst_dst_dist = src_gst_dist[dst][0]

            terrestrial_dist = gst_src_dist + gst_dst_dist
            terrestrial_latency = terrestrial_dist / LIGHT_IN_FIBER

            gst_gst_latency = gst_gst_time_rerouted[gst_src, gst_dst]
            total_latency = terrestrial_latency + gst_gst_latency
            src_dst_latency[src, dst] = total_latency

    src_dst_latency = src_dst_latency + src_dst_latency.T

    assert np.array_equal(src_dst_latency, src_dst_latency.T)

    return src_dst_latency


def add_rerouting_to_inactive(inactive, nearest_active, nearest_active_dist,
                              gst_gst_time):
    """Modify gst-gst time for inactive stations adding the rerouting to the
    nearest active gst."""
    gst_gst_time_rerouted = gst_gst_time.copy()

    for idx_closest, inact_idx in enumerate(inactive):
        nearest_gst_src = nearest_active[idx_closest]
        nn_active_dist = nearest_active_dist[idx_closest]
        nn_active_time = nn_active_dist / LIGHT_IN_FIBER

        gst_gst_time_rerouted[inact_idx, :] = gst_gst_time_rerouted[
                                              nearest_gst_src,
                                              :] + nn_active_time
        gst_gst_time_rerouted[:, inact_idx] = gst_gst_time_rerouted[:,
                                              nearest_gst_src] + nn_active_time

        # Make sure that the GST->SELF distance is always 0
        gst_gst_time_rerouted[inact_idx, inact_idx] = 0

    return gst_gst_time_rerouted


if __name__ == "__main__":
    # ==Show an example==
    import time
    import matplotlib.pyplot as plt
    from plotting.plot_util import scatter_on_map

    # Clockit!
    start = time.time()
    np.random.seed(42)

    # Parameters of the constellations
    H = 1300
    MIN_ANGLE = 40
    ORBITS = 32
    SAT_PER_ORBIT = 50
    INCLINATION = 53
    NUM_DEACTIVATE = 20
    CUR_CITY = 1542  # Zurich

    # Get satellite posistions
    _, sat_pos = make_test_constellation(ORBITS, SAT_PER_ORBIT,
                                         INCLINATION, H,
                                         time=0)

    # Load GSTS
    ixps = pd.read_csv("data/ixp_geolocation.csv")
    lats = np.asarray(ixps['lat']).reshape((-1, 1))
    lngs = np.asarray(ixps['lng']).reshape((-1, 1))

    # Remove IXPs that are too high in latitude
    higher = np.where(lats > 56)[0]
    lats = np.delete(lats, higher).reshape((-1, 1))
    lngs = np.delete(lngs, higher).reshape((-1, 1))

    ixp_pos = np.hstack((lats, lngs))
    print(ixp_pos.shape)
    gst_pos = np.unique(ixp_pos, axis=0)
    print(gst_pos.shape)

    # Load SRC positions
    cities = pd.read_csv("data/WUP2018-F22-Cities_Over_300K_Annual.csv")

    cities_lat = np.asarray(cities['Latitude']).reshape((-1, 1))
    cities_lon = np.asarray(cities['Longitude']).reshape((-1, 1))
    src_pos = np.hstack((cities_lat, cities_lon))
    print(f"The chosen city is {src_pos[CUR_CITY]}")

    # Make the terrestrial GST graph
    terrestrial_gst_graph = delaunay_pos_graph(gst_pos)

    # Randomly pick some GSTs to deactivate
    indexes = np.arange(gst_pos.shape[0])

    inactive = np.random.choice(indexes, size=NUM_DEACTIVATE, replace=False)

    src_dist_latency, na = optimize_end_to_end_latency_rerouting(
        sat_pos, H, gst_pos, src_pos, MIN_ANGLE, ORBITS, SAT_PER_ORBIT,
        terrestrial_gst_graph, inactive)

    print(f"The computation took {time.time() - start} s.")

    # Optional plot checks -----------------------------------------------------

    ax, _ = scatter_on_map(src_pos[:, 0], src_pos[:, 1],
                           c=src_dist_latency[CUR_CITY, :],
                           cmap="Spectral_r", annotate=False, figsize=(24, 18),
                           label="Cities")
    # all gsts
    scatter_on_map(gst_pos[:, 0], gst_pos[:, 1], ax=ax,
                   c='b', marker='o', s=2, label="IXPs")
    # closest active gst
    scatter_on_map(gst_pos[na, 0], gst_pos[na, 1], label="Closest active",
                   oceans=False, s=20, c='c', ax=ax, marker="^")
    # positions of inactive gsts
    scatter_on_map(gst_pos[inactive, 0], gst_pos[inactive, 1], label="Inactive",
                   c='m', marker='v', s=20, ax=ax)
    # current city
    scatter_on_map(src_pos[CUR_CITY, 0], src_pos[CUR_CITY, 1], c="b",
                   marker="X", s=40, ax=ax, label="Current source")
    ax.legend().set_zorder(1000)

    plt.figure(figsize=(12, 8))
    n, bins, patches = plt.hist(src_dist_latency.flatten(), density=True,
                                cumulative=True,
                                bins=1000, histtype='step')

    plt.show()
