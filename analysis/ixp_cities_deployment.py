"""Comparison of deploying at cities and at IXPs."""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from geopy.distance import EARTH_RADIUS
from sklearn.neighbors.ball_tree import BallTree
from sklearn.neighbors.dist_metrics import DistanceMetric

from lib.latency_lib import LIGHT_IN_VACUUM, \
    LIGHT_IN_FIBER, FIBER_PATH_STRETCH
# from eco_sim.ground_segment_cost.remake_gst_simulation import *
from lib.latency_lib import gsts_optimization
from lib.latency_lib import load_locations
from lib.rerouting_lib import compute_sat_sat_distance, \
    compute_gst_sat_distance, src_nearest_gst_distance, compute_src_dst_latency


def compute_distances():
    # Load IXP-GST positions
    altitude = 1150
    min_elev = 40
    orbits = 32
    sat_per_orbit = 50
    inclination = 53
    gst_file = "data/raw/ixp_geolocation.csv"
    src_file = "data/raw/WUP2018-F22-Cities_Over_300K_Annual.csv"

    # Load geo information
    sat_pos, gst_pos, src_pos = load_locations(altitude, orbits, sat_per_orbit,
                                               inclination, gst_file, src_file,
                                               time=15000)

    lon_sort_idx_src = np.argsort(src_pos[:, 1])
    src_pos = (src_pos[lon_sort_idx_src])

    # Remove SRCs that are too high in latitude
    higher = np.where(src_pos[:, 0] > 56)[0]
    src_pos = np.delete(src_pos, higher, axis=0)

    lon_sort_idx_gst = np.argsort(gst_pos[:, 1])
    gst_pos = (gst_pos[lon_sort_idx_gst])

    # %%
    sat_sat_dist = compute_sat_sat_distance(sat_pos, altitude, orbits,
                                            sat_per_orbit)
    # Compute the BallTree for the satellites. This gives nn to satellites.
    sat_tree = BallTree(np.deg2rad(sat_pos),
                        metric=DistanceMetric.get_metric("haversine"))

    # Get the satellites that are in reach for the ground stations
    #   and their distance.
    sat_gst_ind_city, sat_gst_dist_city = compute_gst_sat_distance(altitude,
                                                                   min_elev,
                                                                   src_pos,
                                                                   sat_tree)

    src_src_satellite = gsts_optimization(sat_gst_ind_city, sat_gst_dist_city,
                                          sat_sat_dist, n_gsts=src_pos.shape[0])

    src_src_latency = src_src_satellite / LIGHT_IN_VACUUM

    # %%
    sat_gst_ind_ixp, sat_gst_dist_ixp = compute_gst_sat_distance(altitude,
                                                                 min_elev,
                                                                 gst_pos,
                                                                 sat_tree)

    gst_gst_satellite = gsts_optimization(sat_gst_ind_ixp, sat_gst_dist_ixp,
                                          sat_sat_dist, n_gsts=gst_pos.shape[0])

    src_gst_ind, src_gst_dist = src_nearest_gst_distance(src_pos, gst_pos)

    n_src = src_pos.shape[0]
    src_gst_latency = compute_src_dst_latency(n_src, [], src_gst_ind,
                                              src_gst_dist, [],
                                              [],
                                              gst_gst_satellite)

    return src_gst_latency, src_src_latency, src_pos


def vector_map_statistics(reference, values, rounding, percentile=[95]):
    reference = np.around(reference / rounding) * rounding
    ranges = np.sort(np.unique(reference))

    average_val = []
    min_val = []
    max_val = []
    percent_val = {x: [] for x in percentile}
    for x in ranges:
        idx = np.where(reference == x)[0]
        cur = values[idx]
        if cur.shape[0] > 0:
            average_val.append(np.average(cur))
            min_val.append(np.min(cur))
            max_val.append(np.max(cur))
            for perc in percentile:
                percent_val[perc].append(np.percentile(cur, perc))
        else:
            average_val.append(0)
            min_val.append(0)
            max_val.append(0)
            for perc in percentile:
                percent_val[perc].append(np.percentile(cur, perc))

    return ranges, average_val, min_val, max_val, percent_val


def plot_absolute(src_gst_latency, src_src_latency, src_pos):
    triu = np.triu_indices(src_gst_latency.shape[0], 1)
    ixp_routed = np.around(src_gst_latency[triu], 6)
    city_gst = np.around(src_src_latency[triu], 6)

    SCALING = 1e3

    plt.figure(figsize=(8, 6))
    pairwise_src = DistanceMetric.pairwise(
        DistanceMetric.get_metric("haversine"),
        np.deg2rad(src_pos), np.deg2rad(src_pos))
    pairwise_src = pairwise_src * EARTH_RADIUS;

    pairwise = pairwise_src[triu]
    vals, avg_c, min_c, max_c, _ = vector_map_statistics(pairwise, city_gst, 10)

    avg_c = np.asarray(avg_c) * SCALING
    min_c = np.asarray(min_c) * SCALING
    max_c = np.asarray(max_c) * SCALING

    plt.plot(vals, avg_c, label="Average city-city", linewidth=3)
    plt.xlabel("SRC-DST distance (km)")
    plt.ylabel("Latency (s)")
    plt.legend(loc=2)

    pairwise = pairwise_src[triu]
    vals, avg_g, min_g, max_g, _ = vector_map_statistics(pairwise, ixp_routed,
                                                         10)

    avg_g = np.asarray(avg_g) * SCALING
    min_g = np.asarray(min_g) * SCALING
    max_g = np.asarray(max_g) * SCALING

    plt.plot(vals, avg_g, label="Average IXP-city", linewidth=3)

    plt.plot(vals, vals / LIGHT_IN_FIBER * SCALING, ':', linewidth=3,
             label="Great-circle in fiber")
    plt.plot(vals, vals / LIGHT_IN_VACUUM * SCALING, '--',
             label="Great-circle in vacuum", linewidth=3)
    plt.plot(vals, vals * FIBER_PATH_STRETCH / LIGHT_IN_FIBER * SCALING, '-.',
             label="Path-stretch in fiber", linewidth=3)

    plt.ylim(0, 150)
    plt.xlim(0, np.max(vals))

    plt.xlabel("SRC-DST great-circle distance (km)")
    plt.ylabel("One-way latency (s)")
    plt.legend(loc=9, ncol=2, mode="expand")

    # Save figures
    # plt.savefig("figures/latency-distance.pdf")
    plt.savefig("figures/latency-distance.png")


def plot_relative(src_gst_latency, src_src_latency, src_pos):
    triu = np.triu_indices(src_gst_latency.shape[0], 1)
    ixp_routed = np.around(src_gst_latency[triu], 6)
    city_gst = np.around(src_src_latency[triu], 6)

    pairwise_src = DistanceMetric.pairwise(
        DistanceMetric.get_metric("haversine"),
        np.deg2rad(src_pos), np.deg2rad(src_pos))
    pairwise_src = pairwise_src * EARTH_RADIUS;

    percent = (ixp_routed - city_gst) / city_gst * 100
    pairwise = pairwise_src[triu]

    vals, avg_c, min_c, max_c, percent = vector_map_statistics(pairwise,
                                                               percent,
                                                               100, [25, 75])
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cur_color = colors[0]

    plt.figure(figsize=(8, 6))
    gs1 = gridspec.GridSpec(2, 1)
    gs1.update(wspace=0.01, hspace=0.11)  # set the spacing between axes.

    axes = [plt.subplot(gs1[0]), plt.subplot(gs1[1])]

    axes[0].axhline(0, c='grey', linewidth=0.5)
    axes[0].semilogy(vals, avg_c, label="Average", c=cur_color, linewidth=3)
    axes[0].semilogy(vals, percent[25], "--", label="Quartiles", c=cur_color, linewidth=3)
    axes[0].semilogy(vals, percent[75], "--", c=cur_color, linewidth=3)
    axes[0].plot(vals, max_c, ":", linewidth=3, label="Min-max variability", c=cur_color)
    axes[0].set_ylim(90, 1000)
    axes[0].set_ylabel("log-scale")
    axes[0].set_xticks([])
    axes[0].legend()

    axes[1].axhline(0, c='grey', linewidth=0.5)
    axes[1].plot(vals, avg_c, label="Average city-city", c=cur_color, linewidth=3)
    axes[1].plot(vals, percent[25], "--", label="Quartiles", c=cur_color, linewidth=3)
    axes[1].plot(vals, percent[75], "--", c=cur_color, linewidth=3)
    axes[1].plot(vals, max_c, ":", linewidth=3, label="Min-max variability", c=cur_color)
    axes[1].plot(vals, min_c, ":", linewidth=3, label="Min-max variability", c=cur_color)
    axes[1].set_xlabel("SRC-DST great-circle distance (km)")
    axes[1].set_ylabel("Loss IXP deployment (%)")
    # axes[1].set_ylabel("Latency increase IXP deployment (%)")
    axes[1].set_ylim(-50, 90)

 
    plt.savefig("figures/percent-ixp-loss.png")

def main():
    print("Starting computation of latency under GST placement.")
    src_gst_latency, src_src_latency, src_pos = compute_distances()

    plot_absolute(src_gst_latency, src_src_latency, src_pos)
    plot_relative(src_gst_latency, src_src_latency, src_pos)
    plt.show()


if __name__ == "__main__":
    main()
