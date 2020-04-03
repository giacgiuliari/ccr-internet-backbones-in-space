"""Comparison of Path-Control and Re-routing schemes.

Analysis for inter-domain routing with satellite constellations.
"""
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import DistanceMetric

from lib.analysis_util import load_all_runs, plotmetric
from lib.latency_lib import LIGHT_IN_FIBER, LIGHT_IN_VACUUM, \
    FIBER_PATH_STRETCH, haversine_to_km, load_locations


def cdf_comparison(rerouting_experiment):
    # Load pre-processed data for path control
    data3, paco_hist3, rero_hist3, pc3 = load_all_runs(rerouting_experiment)

    # Load geo information
    altitude = 1300
    min_elev = 40
    orbits = 32
    sat_per_orbit = 50
    inclination = 53
    GST_FILE = "data/raw/ixp_geolocation.csv"
    SRC_FILE = "data/raw/WUP2018-F22-Cities_Over_300K_Annual.csv"

    sat_pos, gst_pos, src_pos = load_locations(altitude, orbits, sat_per_orbit,
                                               inclination, GST_FILE, SRC_FILE)

    # Distance between cities
    pairwise_src = DistanceMetric.pairwise(
        DistanceMetric.get_metric("haversine"),
        src_pos, src_pos)
    pairwise_src = haversine_to_km(pairwise_src)

    terr_only = pairwise_src * FIBER_PATH_STRETCH / LIGHT_IN_FIBER
    gclatency = pairwise_src / LIGHT_IN_FIBER
    gcbound = pairwise_src / LIGHT_IN_VACUUM

    gc_hist, buckets = np.histogram(gclatency * 1000, bins=100, range=(0, 120))
    gcb_hist, _ = np.histogram(gcbound * 1000, bins=100, range=(0, 120))
    terr_hist, _ = np.histogram(terr_only * 1000, bins=100, range=(0, 120))

    plt.figure(figsize=(8, 6))
    plt.plot(buckets[1:], np.cumsum(gc_hist) / np.sum(gcb_hist), ":",
             markersize=4, markevery=[50], label="Great-circle fiber-speed", linewidth=3)
    plt.plot(buckets[1:], np.cumsum(gcb_hist) / np.sum(gcb_hist), ':',
             markersize=4, markevery=[40], label="Great-circle c-speed", linewidth=3)
    plt.plot(buckets[1:], np.cumsum(terr_hist) / np.sum(gcb_hist), ':',
             markersize=4, markevery=[30], label="Great-circle  path", linewidth=3)
    plt.xlabel("One-way latency (ms)")
    plt.ylabel("CDF")

    x_axis = data3[10]['paco']['hist'][0][1][:-1] * 1000

    paco_avg = np.asarray(paco_hist3)
    paco_avg = np.average(paco_avg, axis=0)
    rero_avg = np.asarray(rero_hist3)
    rero_avg = np.average(rero_avg, axis=0)

    plt.plot(x_axis, np.cumsum(paco_avg) / np.sum(paco_avg), '-', markersize=4,
             markevery=[40], label="PaCo avg.", linewidth=3)

    plt.plot(x_axis, np.cumsum(rero_avg) / np.sum(rero_avg), '-.', markersize=4,
             markevery=[30], label="ReRo avg.", linewidth=3)

    plt.legend()
    plt.xlim(0, 120)
    plt.ylim(0, 1)

    # Save figure
    plt.savefig("figures/paco-rero-cdf.png")


def relative_comparison(rerouting_experiment):
    # Load the pre-computed datset
    data, all_hist_paco, all_hist_rero, all_percent_loss = load_all_runs(
        rerouting_experiment)

    # Compute the ranges
    all_maxima = [x['maxima'] for x in all_percent_loss]
    all_average = [x['average'] for x in all_percent_loss]
    all_percent95 = [x['percent95'] for x in all_percent_loss]

    # Plot the comparison
    plt.figure(figsize=(8, 6))

    plt.xlim(0, 120)
    plt.ylim(0, np.max(all_maxima))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    avg = np.average(all_maxima, axis=0)
    maxim = np.max(all_maxima, axis=0)
    plt.plot(np.arange(120), maxim, ":", label="Worst case", c=colors[0], linewidth=3, dash_capstyle='round')
    plt.plot(avg, "--", label=f"Avg. maximum", c=colors[2], linewidth=3, dash_capstyle='round')
    avg = np.average(all_average, axis=0)
    plt.plot(avg, label=f"Avg. median", c=colors[1], linewidth=3, solid_capstyle='round')

    plt.legend()

    plt.tight_layout()
    plt.xlabel("Latency PaCo (ms)")
    plt.ylabel("Loss without PaCo (%)")

    # Save figures, if necessaty
    # plt.savefig("figures/paco-rero-loss.pdf")
    plt.savefig("figures/paco-rero-loss.png")

    # Display the results


def main():

    parser = ArgumentParser()
    parser.add_argument("rerouting_experiment", type=str,
                       help="Directory containing the pre-processed rerouting "
                            "experiment data")

    args = parser.parse_args()

    cdf_comparison(args.rerouting_experiment)
    relative_comparison(args.rerouting_experiment)

    plt.show()


if __name__ == "__main__":
    main()
