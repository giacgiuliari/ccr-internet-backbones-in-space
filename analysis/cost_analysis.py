"""Cost of deployment of GSTs.

The simulations deploys GSTs randomly based on population. Then, the cost of
the fiber connection of the each GST to the closest IXP is computed.

Remake of the old 10000 GST simulation. See what the cost would be.
"""
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from lib.cost_util import ixps_ball_tree, sample_gst_coordinates, \
    nn_distance, load_gdp_distribution_data
from lib.latency_lib import FIBER_PATH_STRETCH

GDP_DISTR = "data/raw/spatialecon-gecon-v4-gis-ascii/" \
            "GIS data ASCII/MER2005/mer2005sum.asc"
BIGGEST_CITIES = "data/raw/WUP2018-F22-Cities_Over_300K_Annual.csv"
IXP_GEOLOCATION = "data/raw/ixp_geolocation.csv"

FIBER_COST = 10e3  # 10,000 $/km
BILLION = 1e9
COMM_FACILITY_UC = 3875  # Cost of communications facilities, per sqm.
GST_SQM = 1780  # Ground station size, sqm
ANTENNA_COST = 100e3  # Cost of the antenna T

ENGINEER_SALARY = 80000
PERSONNEL_SALARY_FACTOR = 1.6
GLOBAL_AVERAGE_WAGE = 18000  # Source https://www.bbc.com/news/magazine-17512040$$C


def repeated_gst_cost_analysis(rounds, num_gst, replace=True):
    """Repeat analysis_stations_price to get estimates of the variability.

    Args:
        rounds: Number of times to repeat the simulation
        max_stations: see analysis_stations_price
        step: see analysis_stations_price
    """
    IXP_tree, _ = ixps_ball_tree(IXP_GEOLOCATION)
    gdp_head, gdp_data = load_gdp_distribution_data(GDP_DISTR)

    all_analysis = []

    for cur_round in range(rounds):
        print(f"round {cur_round} / {rounds}\r", end="")
        lat, lng = sample_gst_coordinates(num_gst, gdp_data, gdp_head,
                                          replace=replace,
                                          filter_prob=True,
                                          randomize=True)

        lat = np.asarray(lat).reshape(-1, 1)
        lng = np.asarray(lng).reshape(-1, 1)
        gst_pos = np.hstack((lat, lng))

        dist_sum = analyze_gst_deployment(gst_pos, IXP_tree)
        all_analysis.append(np.sum(dist_sum))

    print(f"round {rounds} / {rounds}")
    cost = np.multiply(all_analysis, FIBER_COST)
    return np.average(cost)


def analyze_gst_deployment(gst_pos, tree, compute_cost=False):
    distances, _ = nn_distance(tree, gst_pos)
    # Apply the terrestrial path stretch
    distances = np.sum(distances[distances < 1000]) * FIBER_PATH_STRETCH
    if compute_cost:
        cost = distances * FIBER_COST
        return cost
    return distances


def cities_to_GSTs_distance():
    """Simulate the cost of the WAN in the particular case that the biggest
    cities are the GSTs locations."""
    # Load the positions of GSTs
    IXP_tree, _ = ixps_ball_tree(IXP_GEOLOCATION)
    cities = pd.read_csv(BIGGEST_CITIES)

    cities_lat = np.asarray(cities['Latitude']).reshape((-1, 1))
    cities_lon = np.asarray(cities['Longitude']).reshape((-1, 1))

    # Remove SRCs that are too high in latitude
    higher = np.where(cities_lat > 56)[0]
    cities_lat = np.delete(cities_lat, higher, axis=0)
    cities_lon = np.delete(cities_lon, higher, axis=0)

    cities_pos = np.hstack((cities_lat, cities_lon))

    deploy_cost = analyze_gst_deployment(cities_pos, IXP_tree,
                                         compute_cost=True)

    return deploy_cost, cities_lat.shape[0]


if __name__ == "__main__":

    np.random.seed(42)

    parser = ArgumentParser()

    parser.add_argument("-n", "--num_gst", type=int,
                        help="The number fo GSTs to sample")
    parser.add_argument("-r", "--rounds", type=int,
                        help="The number of re-sampling rounds to average.")

    args = parser.parse_args()

    wan_average_cost = repeated_gst_cost_analysis(args.rounds, args.num_gst)

    antennas_cost = args.num_gst * 3e6 / BILLION
    infrastructure_cost = args.num_gst * COMM_FACILITY_UC * GST_SQM / BILLION

    total_cost = wan_average_cost / BILLION + antennas_cost + infrastructure_cost

    print(f"BREAKDOWN of the costs of construction for {args.num_gst} "
          f"randomly-sampled GSTs")
    print(f"   - WAN: {wan_average_cost / BILLION} bln $ (average)")
    print(f"   - Antennas: {antennas_cost} bln $")
    print(f"   - Infrastructure: {infrastructure_cost} bln $")
    print(f"For a total of {total_cost} bln $")

    # Cost of the "Cities deployment"
    cities_wan_cost, n_cities = cities_to_GSTs_distance()

    antennas_cost = n_cities * 3e6 / BILLION
    infrastructure_cost = n_cities * COMM_FACILITY_UC * GST_SQM / BILLION

    total_cost = cities_wan_cost / BILLION + antennas_cost + infrastructure_cost

    print(f"BREAKDOWN of the costs of construction for {n_cities} "
          f"GSTs at mayor cities")
    print(f"   - WAN: {cities_wan_cost / BILLION} bln $")
    print(f"   - Antennas: {antennas_cost} bln $")
    print(f"   - Infrastructure: {infrastructure_cost} bln $")
    print(f"For a total of {total_cost} bln $")
