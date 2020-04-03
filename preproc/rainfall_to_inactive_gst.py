"""Use historical weather data from NOAA to compute sets of inactive GSTs."""


import datetime
import os
import pickle
import time
from argparse import ArgumentParser

import cfgrib
import numpy as np
import pandas as pd
from termcolor import colored


def load_rainfall_values(file):
    datasets = cfgrib.open_datasets(file)
    print("executed open datasets")
    i = 0
    for cur in datasets:
        try:
            rainfall_values = cur['tp'].values
            print(f"Found at iteration {i}")
            break
        except:
            i += 1
    # rainfall_values = dataset[26]['tp'].values
    return rainfall_values


def latlon_to_matrix_pos(latlon, scale=2):
    tmp = latlon.copy()
    tmp = np.floor(tmp)
    matrix_idx = np.zeros(tmp.shape)
    matrix_idx[:, 0] = - tmp[:, 0] + 90
    lons = tmp[:, 1]
    lons[lons < 0] = lons[lons < 0] + 360
    matrix_idx[:, 1] = lons
    matrix_idx = np.floor(matrix_idx * 2).astype(int)
    return matrix_idx


def rainfall_to_inactive(gst_matrix_pos, rainfall_values, thresh):
    rainvalues = rainfall_values[gst_matrix_pos[:, 0], gst_matrix_pos[:, 1]]
    inactive_idx = np.where(rainvalues > thresh)[0]
    return inactive_idx


def load_gst_ixp_positions(datafile):
    ixps = pd.read_csv(datafile)
    lats = np.asarray(ixps['lat']).reshape((-1, 1))
    lngs = np.asarray(ixps['lng']).reshape((-1, 1))

    # Remove IXPs that are too high in latitude
    higher = np.where(lats > 56)[0]
    lats = np.delete(lats, higher).reshape((-1, 1))
    lngs = np.delete(lngs, higher).reshape((-1, 1))

    gst_pos = np.hstack((lats, lngs))
    gst_pos = np.unique(gst_pos, axis=0)
    gst_matrix_pos = latlon_to_matrix_pos(gst_pos)
    return gst_matrix_pos


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("dataset_dir", type=str,
                       help="Directory containing the weather data")
    parser.add_argument("out", type=str,
                       help="Pickle file in which to save the outcome of the "
                            "experiment.")

    args = parser.parse_args()

    # Load the positions of the ground stations
    ixps = pd.read_csv("data/raw/ixp_geolocation.csv")
    lats = np.asarray(ixps['lat']).reshape((-1, 1))
    lngs = np.asarray(ixps['lng']).reshape((-1, 1))

    # Remove IXPs that are too high in latitude
    higher = np.where(lats > 56)[0]
    lats = np.delete(lats, higher).reshape((-1, 1))
    lngs = np.delete(lngs, higher).reshape((-1, 1))

    gst_pos = np.hstack((lats, lngs))
    gst_pos = np.unique(gst_pos, axis=0)
    gst_matrix_pos = latlon_to_matrix_pos(gst_pos)

    # Threshold levels of rain intensity
    intensity_thrs = [0.5, 0.75, 1, 2, 3, 5, 7, 10, 13, 15, 20]

    # Cycle through the files and check the inactive files


    all_inactive = []

    start_t = time.time()

    files_list = list(filter(lambda x: x.endswith(".grb2"),
                              os.listdir(args.dataset_dir)))
    n_files = len(files_list)
    print(f"There are {n_files} valid grib files")
    times = [start_t]

    for idx, cur_file in enumerate(files_list):
        if not cur_file.endswith(".grb2"):
            continue
        print(f"Running {cur_file}")
        rainfall_values = load_rainfall_values(
            os.path.join(args.dataset_dir, cur_file))
        for cur_thrs in intensity_thrs:
            print(f"   - Running {cur_thrs}")
            cur_inactive = rainfall_to_inactive(gst_matrix_pos, rainfall_values,
                                                cur_thrs)
            all_inactive.append(cur_inactive)
        times.append(time.time())
        deltas = np.ediff1d(times)
        avg_time = np.average(deltas)
        remaining = str(
            datetime.timedelta(seconds=avg_time * (n_files - (idx + 1))))
        print(f">>> Finished {idx + 1} / {n_files}; "
              f"AVERAGE TIME: {avg_time} s;")
        print(colored(f"WILL TERMINATE IN {remaining}", 'yellow'))

    print(f"Total running time is {time.time() - start_t}")

    with open(args.out, 'wb') as outfile:
        pickle.dump(all_inactive, outfile)

    # TODO: Test
