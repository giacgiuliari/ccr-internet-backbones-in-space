"""Make a map of the latency achievable form one point to others.

Given a constellation with ISLs.
"""
import time

import cartopy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from geopy.distance import EARTH_RADIUS
from lib.sat_util import cart2sph
from lib.satellite_topology import SatelliteTopology
from lib.plot_util import gridvalues_to_matrix
from scipy.constants import speed_of_light
from shapely.geometry import Point
from shapely.prepared import prep
from sklearn.neighbors import BallTree, DistanceMetric

FIBER_PATH_STRETCH = 2.3  # Ref. Ankit's work
LIGHT_IN_VACUUM = speed_of_light / 1000  # km/h
LIGHT_IN_FIBER = LIGHT_IN_VACUUM * 2 / 3


def haversine_to_km(haversine, radius=EARTH_RADIUS):
    """Compute the distance in km given the haversine distances."""
    return radius * haversine


def haversine_to_km_altitude(haversine, altitude, radius=EARTH_RADIUS):
    # Using Carnot's rule
    a = EARTH_RADIUS + altitude
    b = EARTH_RADIUS
    C = haversine
    path_km = np.sqrt(np.square(a) + np.square(b) - 2 * a * b * np.cos(C))
    return path_km


def generator_to_matrix(gen, n_sat):
    matrix = np.zeros((n_sat, n_sat))
    for src in gen:
        for dst in gen[src]:
            matrix[src, dst] = gen[src][dst]

    # assert np.array_equal(matrix, matrix.T)
    return matrix


def optimize_end_to_end_latency(sat_pos, altitude, gst_pos, src_pos, nn_src_gst,
                                min_elev, orbits, sat_per_orbit):
    """Optimize the latency end-to-end.

    Optimize the latency over the satellite and terrestrial networks. Different
    degrees of connectivity and path control can be specified.

    Args:
        orbits: Number of orbits in the constellation.
        sat_per_orbit: Number of satellites per orbit.
        sat_pos: Position of the satellites.
        altitude: Altitude of the satellite orbits from the surface in km.
        gst_pos: Position of the GSTs.
        src_pos: Position of the traffic sources on earth. Sources are also
            destinations in the simulation.
        nn_gst_sat: Number of nearest satellites a GST can communicate with.
        nn_src_gst: Number of nearest GSTs a source (destination) can
            communicate with.
        min_elev: minimum elevation angle.

    TODO: Additional parameters can be path stretch ...
    """
    # Compute satellite graph
    st = time.time()
    sat_graph = create_baseline_graph(sat_pos, altitude, orbits, sat_per_orbit)
    sat_sat_dist = nx.all_pairs_dijkstra_path_length(sat_graph, weight='length')
    sat_sat_dist = dict(sat_sat_dist)
    sat_sat_dist = generator_to_matrix(sat_sat_dist, sat_pos.shape[0])

    # Compute the BallTree for the satellites. This gives nn to satellites.
    sat_tree = BallTree(np.deg2rad(sat_pos),
                        metric=DistanceMetric.get_metric("haversine"))

    print(f"time to generation of sat {time.time() - st}")
    # Compute maximum communication distances
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

    # Compute the distance matrix between any pair of GSTs.
    st = time.time()
    gst_gst_dist = gsts_optimization(sat_gst_ind, sat_gst_dist, sat_sat_dist,
                                     n_gsts=gst_pos.shape[0])
    gst_gst_time = gst_gst_dist / LIGHT_IN_VACUUM

    print(f"GST-GST optimization time {time.time() - st}")
    # Compute the BallTree with the distance from the GSTs
    gst_tree = BallTree(np.deg2rad(gst_pos),
                        metric=DistanceMetric.get_metric("haversine"))
    src_gst_dist, src_gst_ind = gst_tree.query(np.deg2rad(src_pos),
                                               k=nn_src_gst)

    print(f"shape of the src gst dist: {src_gst_dist.shape}")

    src_gst_dist = haversine_to_km(src_gst_dist)

    # Account for the path stretch
    src_gst_dist = src_gst_dist * FIBER_PATH_STRETCH
    src_gst_time = src_gst_dist / LIGHT_IN_FIBER

    # Optimize the gst-sat-sat-gst end to end
    st = time.time()
    src_dst_time = src_dst_optimization(src_gst_ind, src_gst_time, gst_gst_time,
                                        n_srcs=src_pos.shape[0])
    print(f"SRC-DST optimization time {time.time() - st}")
    return src_dst_time


def gsts_optimization(sat_gst_ind, sat_gst_dist, sat_sat_dist, n_gsts):
    """Optimize the paths from GST to GST.

    This reduces to finding the optimal hops GST-SAT given the internal
    satellite paths.

    Args:
        gst_pos:
        sat_gst_ind: Distances between the N satellites in the constellation
            following ISL paths. Represented in an (N, N) matrix.
        sat_gst_dist:
    """
    print(f"Running gsts optimization: {sat_gst_dist.shape} {n_gsts}")
    gst_distances = np.zeros((n_gsts, n_gsts))
    for src_gst in range(n_gsts):
        for dst_gst in range(src_gst + 1, n_gsts):
            cur_min_dist = _gst_to_gst_distance(sat_gst_ind, sat_gst_dist,
                                                sat_sat_dist, src_gst,
                                                dst_gst)
            gst_distances[src_gst, dst_gst] = cur_min_dist
    gst_distances += gst_distances.T
    return gst_distances


def _gst_to_gst_distance(sat_gst_ind, sat_gst_dist, sat_sat_dist, start_gst,
                         end_gst):
    """Compute the minimum distance between a pair of GSTs."""
    # Get the visible satellites and distancee for the current start and end
    start_sats = sat_gst_ind[start_gst]
    end_sats = sat_gst_ind[end_gst]
    start_sats_dist = sat_gst_dist[start_gst]
    end_sats_dist = sat_gst_dist[end_gst]

    start_sat_comb = np.repeat(start_sats, end_sats.shape[0])
    start_dist_comb = np.repeat(start_sats_dist, end_sats.shape[0])
    end_sat_comb = np.tile(end_sats, start_sats.shape[0])
    end_dist_comb = np.tile(end_sats_dist, start_sats.shape[0])

    cur_sat_sat_dist = sat_sat_dist[start_sat_comb, end_sat_comb]

    total_distances = start_dist_comb + end_dist_comb + cur_sat_sat_dist

    # print("MIN of the sat distances")
    # print(np.min(start_dist_comb))
    # print(np.min(end_dist_comb))
    # print(np.min(cur_sat_sat_dist))

    try:
        min_dist = np.min(total_distances)
    except ValueError:
        # print(f"no min dist for this {start_gst, end_gst}")
        # print(sat_gst_dist.shape, sat_sat_dist.shape)
        min_dist = np.inf
    return min_dist


def src_dst_optimization(src_gst_ind, src_gst_time, gst_gst_time, n_srcs):
    """Optimize the paths from SRC to DST."""
    src_distances = np.zeros((n_srcs, n_srcs))
    # print(f"Source optimization: {src_gst_ind.shape}, {n_srcs}")
    for src in range(n_srcs):
        for dst in range(src + 1, n_srcs):
            cur_min_dist = _src_to_dst_distance(src_gst_ind, src_gst_time,
                                                gst_gst_time, src, dst)
            src_distances[src, dst] = cur_min_dist
    src_distances += src_distances.T
    return src_distances


def _src_to_dst_distance(src_gst_ind, src_gst_dist, gst_gst_dist, src, dst):
    start_gsts = src_gst_ind[src]
    end_gsts = src_gst_ind[dst]
    start_gsts_dist = src_gst_dist[src]
    end_gsts_dist = src_gst_dist[dst]

    start_gst_comb = np.repeat(start_gsts, end_gsts.shape[0])
    start_dist_comb = np.repeat(start_gsts_dist, end_gsts.shape[0])
    end_gst_comb = np.tile(end_gsts, start_gsts.shape[0])
    end_dist_comb = np.tile(end_gsts_dist, start_gsts.shape[0])

    cur_gst_gst_dist = gst_gst_dist[start_gst_comb, end_gst_comb]

    total_distances = start_dist_comb + end_dist_comb + cur_gst_gst_dist
    min_dist = np.min(total_distances)
    return min_dist


def compute_threshold(h, min_angle):
    """Compute the maximum sat-ground distance using the law of sines.

    In the computation:
        alpha: angle at the GST, pointing SAT and CENTER.
        beta: angle at the SAT, pointing GST and CENTER.
        gamma: angle at GENTER, pointing at GST and SAT.
        (sides are relative).

    Returns:
        c: the maximum distance GST-SAT.
        arc: the great-circle distance from the nadir of a satellite
            to the farthest GST it can communicate to.
    """
    alpha = np.deg2rad(min_angle + 90)
    a = h + EARTH_RADIUS
    b = EARTH_RADIUS
    sin_beta = np.sin(alpha) / a * b
    beta = np.arcsin(sin_beta)
    gamma = np.pi - alpha - beta
    c = a * np.sin(gamma) / np.sin(alpha)
    arc = EARTH_RADIUS * gamma
    return c, arc


def make_test_constellation(n_orbits, sat_per_orbit, inclination, altitude,
                            time=0):
    sat_topology = []
    for cur in range(n_orbits):
        sat_topology.append(
            {'n_sat': sat_per_orbit,
             'inclination': inclination,
             'height': altitude,
             'initial_offset': 360 / n_orbits * cur,
             'phase_shift': (360 / sat_per_orbit) / n_orbits}
        )
    sat_topology = {'planes': sat_topology}
    topology = SatelliteTopology(sat_topology)
    sat_pos = topology.compute_topology(time_instant=time, spherical=False)
    sat_pos = cart2sph(sat_pos)
    x = np.asarray(sat_pos[:, 1]).reshape(-1)
    y = 90 - np.asarray(sat_pos[:, 2]).reshape(-1)
    constellation = np.vstack((y, x)).T
    return sat_pos, constellation


def load_locations(altitude, orbits, sat_per_orbit, inclination, gst_file,
                   src_file, time=0):
    # Get satellite positions
    _, sat_pos = make_test_constellation(orbits, sat_per_orbit,
                                         inclination, altitude,
                                         time=time)
    # Load GSTS
    ixps = pd.read_csv(gst_file)
    lats = np.asarray(ixps['lat']).reshape((-1, 1))
    lngs = np.asarray(ixps['lng']).reshape((-1, 1))

    # Remove IXPs that are too high in latitude
    higher = np.where(lats > 56)[0]
    lats = np.delete(lats, higher).reshape((-1, 1))
    lngs = np.delete(lngs, higher).reshape((-1, 1))
    # Remove duplicates
    ixp_pos = np.hstack((lats, lngs))
    gst_pos = np.unique(ixp_pos, axis=0)

    # Load SRC positions
    cities = pd.read_csv(src_file)
    cities_lat = np.asarray(cities['Latitude']).reshape((-1, 1))
    cities_lon = np.asarray(cities['Longitude']).reshape((-1, 1))
    src_pos = np.hstack((cities_lat, cities_lon))

    return sat_pos, gst_pos, src_pos


def min_dist_to_satellites(sat_pos, h, scaling, points=None):
    if points is None:
        la, lg = np.mgrid[-90:90:scaling, -180:180:scaling]

        points = np.c_[la.ravel(), lg.ravel()]

    tree = BallTree(np.deg2rad(sat_pos),
                    metric=DistanceMetric.get_metric("haversine"))
    dd, ii = tree.query(np.deg2rad(points))

    true_dist = haversine_to_km_altitude(dd, h)

    return true_dist, ii, tree, points


def orbit_and_pos_to_satid(orbit, position=None, orbits=None,
                           sats_per_orbit=None):
    if position is None:
        orbit, position = orbit
    satid = orbit * sats_per_orbit + position
    return satid


def compute_isl_length(sat1, sat2, h, sat_positions):
    """
        Compute ISL length between pairs of satellites
    """
    lat1 = np.deg2rad(sat_positions[sat1, 0])
    lng1 = np.deg2rad(sat_positions[sat1, 1])
    lat2 = np.deg2rad(sat_positions[sat2, 0])
    lng2 = np.deg2rad(sat_positions[sat2, 1])

    x1 = (EARTH_RADIUS + h) * np.cos(lat1) * np.sin(lng1)
    y1 = (EARTH_RADIUS + h) * np.sin(lat1)
    z1 = (EARTH_RADIUS + h) * np.cos(lat1) * np.cos(lng1)
    x2 = (EARTH_RADIUS + h) * np.cos(lat2) * np.sin(lng2)
    y2 = (EARTH_RADIUS + h) * np.sin(lat2)
    z2 = (EARTH_RADIUS + h) * np.cos(lat2) * np.cos(lng2)
    dist = np.sqrt(
        np.power((x2 - x1), 2) + np.power((y2 - y1), 2) + np.power((z2 - z1),
                                                                   2))
    return dist


def plot_links(G, sat_positions):
    plt.figure()
    for start, end in list(G.edges())[8:12]:
        slat, slng = sat_positions[start, 0], sat_positions[start, 1]
        elat, elng = sat_positions[end, 0], sat_positions[end, 1]
        plt.plot([slng, elng], [slat, elat], c='grey')


def create_baseline_graph(sat_positions, h, orbits, sat_orbit):
    """Creates the satellite constellation ISL connectivity graph.

    In this baseline graph, the ISL are simply connected in plane and cross
    plane.
    """
    G = nx.Graph()
    for satellite in range(sat_positions.shape[0]):
        G.add_node(satellite)

    for idx in range(sat_positions.shape[0]):
        orbit = np.floor(idx / sat_orbit).astype(int)
        pos = idx % sat_orbit

        # #Orthogonal grid computation
        SHIFT = 1
        prev = (orbit, (pos - SHIFT) % sat_orbit)
        next = (orbit, (pos + SHIFT) % sat_orbit)
        left = ((orbit - SHIFT) % orbits, pos)
        right = ((orbit + SHIFT) % orbits, pos)

        prev = orbit_and_pos_to_satid(prev, orbits=orbits,
                                      sats_per_orbit=sat_orbit)
        next = orbit_and_pos_to_satid(next, orbits=orbits,
                                      sats_per_orbit=sat_orbit)
        left = orbit_and_pos_to_satid(left, orbits=orbits,
                                      sats_per_orbit=sat_orbit)
        right = orbit_and_pos_to_satid(right, orbits=orbits,
                                       sats_per_orbit=sat_orbit)

        l_prev = compute_isl_length(idx, prev, h, sat_positions)
        l_next = compute_isl_length(idx, next, h, sat_positions)
        l_left = compute_isl_length(idx, left, h, sat_positions)
        l_right = compute_isl_length(idx, right, h, sat_positions)

        G.add_edge(idx, prev, length=l_prev)
        G.add_edge(idx, next, length=l_next)
        G.add_edge(idx, left, length=l_left)
        G.add_edge(idx, right, length=l_right)
    return G


def compute_land_mask(scaling, sat_coverage=False):
    land_10m = cartopy.feature.NaturalEarthFeature('physical', 'land',
                                                   '10m')
    land_polygons = list(land_10m.geometries())

    lats = np.arange(-90, 90, scaling)
    lons = np.arange(-180, 180, scaling)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    points = [Point(point) for point in
              zip(lon_grid.ravel(), lat_grid.ravel())]

    land_polygons_prep = [prep(land_polygon) for land_polygon in
                          land_polygons]

    land = []
    for land_polygon in land_polygons_prep:
        land.extend([tuple(point.coords)[0] for point in
                     filter(land_polygon.covers, points)])

    land = np.asarray(land)
    land = np.flip(land, axis=1)

    land_mask = gridvalues_to_matrix(land, values=np.ones(land.shape[0]),
                                     points_per_deg=scaling)
    if sat_coverage:
        land_mask[0:np.floor(34 / scaling).astype(int), :] = 0
        land_mask[np.floor(146 / scaling).astype(int):, :] = 0

    return land_mask


if __name__ == "__main__":
    pass
