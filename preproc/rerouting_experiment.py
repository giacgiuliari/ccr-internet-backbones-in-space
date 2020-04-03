"""Compute the routing latencies using, using the sets of inactive GSTs.

Run the re-routing many times with different sets of inactive GSTs.
Compare with the optimal path control.
"""

import os
import pickle
import time
from argparse import ArgumentParser
from multiprocessing import Pool

import numpy as np
from sklearn.neighbors.ball_tree import BallTree
from sklearn.neighbors.dist_metrics import DistanceMetric

from lib.latency_lib import src_dst_optimization, LIGHT_IN_FIBER, \
    LIGHT_IN_VACUUM, gsts_optimization, load_locations
from lib.rerouting_lib import compute_src_dst_latency, \
    inactive_to_closest_active, gst_gst_terrestrial_distance, \
    src_nearest_gst_distance, delaunay_pos_graph, compute_gst_sat_distance, \
    compute_sat_sat_distance, add_rerouting_to_inactive


def run(path_control, deactivate_schedule, n_runs, cores, out_path,
        inactive_list=None):
    # Variables TODO: put as arguments
    altitude = 1300
    min_elev = 40
    orbits = 32
    sat_per_orbit = 50
    inclination = 53
    gst_file = "data/raw/ixp_geolocation.csv"
    src_file = "data/raw/WUP2018-F22-Cities_Over_300K_Annual.csv"

    # Load geo information
    sat_pos, gst_pos, src_pos = load_locations(altitude, orbits, sat_per_orbit,
                                               inclination, gst_file, src_file)
    n_src = src_pos.shape[0]
    terrestrial_gst_graph = delaunay_pos_graph(gst_pos)

    # Pre-compute intermediate results
    out = experiment_setup(sat_pos, altitude, src_pos, gst_pos, min_elev,
                           orbits, sat_per_orbit, terrestrial_gst_graph,
                           path_control)
    src_gst_ind, src_gst_dist, gst_gst_terrestrial, gst_gst_satellite = out

    # Run the comparison and save results
    indexes = np.arange(gst_pos.shape[0])
    fixed_params = {
        'indexes': indexes,
        'n_src': n_src,
        'gst_gst_satellite': gst_gst_satellite,
        'gst_gst_terrestrial': gst_gst_terrestrial,
        'src_gst_ind': src_gst_ind,
        'src_gst_dist': src_gst_dist,
        'out_path': out_path
    }

    if inactive_list is not None:
        print("========================================")
        print("Inactive list is present")
        n_runs = len(inactive_list)
        print(f"There are {n_runs} inactive lists")
        cores_split = np.array_split(np.arange(n_runs), cores)
        inactive_list_split = np.array_split(inactive_list, cores)
        args = []
        for cur_split, cur_inactive in zip(cores_split, inactive_list_split):
            args.append((cur_inactive, cur_split, fixed_params))

        with Pool(cores) as proc_pool:
            proc_pool.map(parallel_run_w_inactive, args)

    else:
        for num_deactivate in deactivate_schedule:
            print(f"=======================================")
            print(f"Running for #inactive: {num_deactivate}")
            startime = time.time()

            cores_split = np.array_split(np.arange(n_runs), cores)
            args = []
            for cur_split in cores_split:
                args.append((num_deactivate, cur_split, fixed_params))

            with Pool(cores) as proc_pool:
                proc_pool.map(parallel_run, args)

            print(f"Parallel run finished in {time.time() - startime} s.")


def parallel_run_w_inactive(args):
    inactive, cur_split, params = args
    for cur_num, cur_inactive in zip(cur_split, inactive):
        src_dst_lat_rerouting, src_dst_lat_pathcontrol = run_configuration(
            params['n_src'],
            cur_inactive,
            params['gst_gst_satellite'],
            params['gst_gst_terrestrial'],
            params['src_gst_ind'],
            params['src_gst_dist'])
        save_results(params['out_path'], cur_inactive, src_dst_lat_rerouting,
                     src_dst_lat_pathcontrol, run_n=cur_num)


def parallel_run(args):
    num_deactivate, cur_split, params = args
    for cur_num in cur_split:
        np.random.seed(cur_num * num_deactivate)
        cur_inactive = np.random.choice(params['indexes'],
                                        size=num_deactivate,
                                        replace=False)
        src_dst_lat_rerouting, src_dst_lat_pathcontrol = run_configuration(
            params['n_src'],
            cur_inactive,
            params['gst_gst_satellite'],
            params['gst_gst_terrestrial'],
            params['src_gst_ind'],
            params['src_gst_dist'])
        save_results(params['out_path'], cur_inactive, src_dst_lat_rerouting,
                     src_dst_lat_pathcontrol, run_n=cur_num)


def experiment_setup(sat_pos, altitude, src_pos, gst_pos, min_elev, orbits,
                     sat_per_orbit, terrestrial_gst_graph, path_control):
    sat_sat_dist = compute_sat_sat_distance(sat_pos, altitude, orbits,
                                            sat_per_orbit)

    # Compute the BallTree for the satellites. This gives nn to satellites.
    sat_tree = BallTree(np.deg2rad(sat_pos),
                        metric=DistanceMetric.get_metric("haversine"))

    # Get the satellites that are in reach for the ground stations
    #   and their distance.
    sat_gst_ind, sat_gst_dist = compute_gst_sat_distance(altitude, min_elev,
                                                         gst_pos, sat_tree)

    # Compute the terrestrial nearest neighbors to sources
    src_gst_ind, src_gst_dist = src_nearest_gst_distance(src_pos, gst_pos,
                                                         path_control)

    # Get the terrestrial GST -> GST distance
    gst_gst_terrestrial = gst_gst_terrestrial_distance(terrestrial_gst_graph,
                                                       gst_pos)

    # Get the satellite GST -> GST distance
    gst_gst_satellite = gsts_optimization(sat_gst_ind, sat_gst_dist,
                                          sat_sat_dist, n_gsts=gst_pos.shape[0])

    return src_gst_ind, src_gst_dist, gst_gst_terrestrial, gst_gst_satellite


def run_configuration(n_src, inactive, gst_gst_satellite, gst_gst_terrestrial,
                      src_gst_ind, src_gst_dist):
    """Run the experiment for a particular configuration of failed GSTs.

    Args:
        n_src: number of traffic sources
        inactive: inactive ground stations
        gst_gst_satellite: distance matrix gst-gst on satellites
        gst_gst_terrestrial: distance matrix gst-gst on the terrestrial graph.
        src_gst_ind:
        src_gst_dist:
    """
    # Get nearest active for every inactive gst
    nearest_active, nearest_active_dist = inactive_to_closest_active(
        inactive, gst_gst_terrestrial)

    # print(f"Nearest active: {len(nearest_active)}")

    # Latency matrix for the rerouting -----------------------------------------
    src_dst_lat_rerouting = compute_src_dst_latency(n_src, inactive,
                                                    src_gst_ind, src_gst_dist,
                                                    nearest_active,
                                                    nearest_active_dist,
                                                    gst_gst_satellite)

    # Latency matrix for path control ------------------------------------------
    src_dst_lat_pathcontrol = path_control_latency(n_src, inactive,
                                                   nearest_active,
                                                   nearest_active_dist,
                                                   gst_gst_satellite,
                                                   src_gst_ind, src_gst_dist)

    return src_dst_lat_rerouting, src_dst_lat_pathcontrol


def path_control_latency(n_src, inactive, nearest_active, nearest_active_dist,
                         gst_gst_satellite, src_gst_ind, src_gst_dist):
    """Compute the latency of the path with path control and inactive gsts.

    For the inactive gsts, the time to reroute via fiber to the closest active
    GST is added.
    """
    src_gst_time = src_gst_dist / LIGHT_IN_FIBER
    gst_gst_time = gst_gst_satellite / LIGHT_IN_VACUUM

    gst_gst_time_rerouted = add_rerouting_to_inactive(inactive, nearest_active,
                                                      nearest_active_dist,
                                                      gst_gst_time)

    src_dst_time = src_dst_optimization(src_gst_ind, src_gst_time,
                                        gst_gst_time_rerouted, n_src)

    return src_dst_time


def save_results(out_pat, inactive, src_dst_lat_rerouting,
                 src_dst_lat_pathcontrol, run_n):
    dirname = f"compare_{len(inactive)}_inactive_{run_n}"
    new_path = os.path.join(out_pat, dirname)
    os.mkdir(new_path)

    np.savetxt(os.path.join(new_path, "inactive.txt"), inactive)

    rerostat = get_statistics(src_dst_lat_rerouting)
    pacostat = get_statistics(src_dst_lat_pathcontrol)

    loss_hist = matrices_to_loss_histogram(src_dst_lat_pathcontrol,
                                           src_dst_lat_rerouting)

    rerostat.update({'percent_loss': loss_hist})

    with open(os.path.join(new_path, "rerouting_stat.pkl"), 'wb') as outfile:
        pickle.dump(rerostat, outfile)
    with open(os.path.join(new_path, "pathcontrol_stat.pkl"), 'wb') as outfile:
        pickle.dump(pacostat, outfile)

    # with open(os.path.join(new_path, "rerouting_stat.pkl"), 'wb') as outfile:
    #     pickle.dump(src_dst_lat_rerouting, outfile)
    # with open(os.path.join(new_path, "pathcontrol_stat.pkl"), 'wb') as outfile:
    #     pickle.dump(src_dst_lat_pathcontrol, outfile)


def matrices_to_loss_histogram(paco_latency, rero_latency):
    uptidx = np.triu_indices(paco_latency.shape[0], 1)
    x_axis = paco_latency[uptidx] * 1000
    idx_axis = np.around(x_axis)
    y_axis = (rero_latency[uptidx] * 1000)
    percent = np.around(y_axis - x_axis, 9)

    maxima = []
    average = []
    percent95 = []
    for cur_x in range(120):
        idx = np.where(idx_axis == cur_x)[0]
        cur_data = percent[idx]
        if cur_data.size != 0:
            maxima.append(np.max(cur_data))
            average.append(np.average(cur_data))
            percent95.append(np.percentile(cur_data, 95))
        else:
            maxima.append(0)
            average.append(0)
            percent95.append(0)

    data = {
        'maxima': maxima,
        'average': average,
        'percent95': percent95,
    }

    return data


def get_statistics(matrix):
    up_tri_idx = np.triu_indices(matrix.shape[0], 1)
    flat = matrix[up_tri_idx].flatten()
    stats = {
        'cmax': np.max(flat),
        'cmin': np.min(flat),
        'avg': np.average(flat),
        'q1': np.percentile(flat, 25),
        'q2': np.percentile(flat, 50),
        'q3': np.percentile(flat, 75),
        'hist': np.histogram(flat, bins=100, range=(0, 0.120))
    }
    return stats


if __name__ == "__main__":
    # Inactive list
    parser = ArgumentParser()
    parser.add_argument("path_control", type=int, default=3,
                       help="The degree of path control to be used in the "
                            "experiment. Default is 3.")
    parser.add_argument("inactive_stations", type=str,
                       help="Pickle file containing the lists of inactive GSTs"
                            "due to rainfall. Produced by "
                            "`rainfall_to_inactive_gst.py`")
    parser.add_argument("out", type=str,
                       help="Folder in which to save the outcome of the "
                            "experiment.")
    parser.add_argument("--cores", type=int, default=1,
                        help="Number of cores to use in the computation.")

    args = parser.parse_args()

    with open(args.inactive_stations, 'rb') as infile:
        inactive_list = pickle.load(infile)

    print(len(inactive_list))
    run(args.path_control, [-1], -1, args.cores, args.out, inactive_list)
