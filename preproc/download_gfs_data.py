"""Download the GFS historical weather data from NOAA."""

import datetime
import os
from argparse import ArgumentParser

import urllib3

URL = "https://nomads.ncdc.noaa.gov/data/gfsanl/"


def save_data(data, date, out, time):
    filename = date.strftime(f'gfsanl_4_%Y%m%d_{time}_006.grb2')
    with open(os.path.join(out, filename), 'wb') as writer:
        writer.write(data)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--start_date", type=str, default="2018.01.01",
                        help="Starting date for the download of the weather "
                             "data. The fromat must be YYYY.MM.DD. "
                             "Default: 2018.01.01")
    parser.add_argument("--end_date", type=str, default="2019.01.01",
                        help="Ending date for the download of the weather "
                             "data. The fromat must be YYYY.MM.DD. "
                             "Default: 2019.01.01")
    parser.add_argument("--time_delta", type=int, default=1,
                        help="Interval in days at which to sample the weather "
                             "data. Default: 1.")
    parser.add_argument("--out", type=str,
                        help="Folder in which to save the downloaded files.")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    start_arg = args.start_date.split(".")
    end_arg = args.end_date.split(".")
    start_arg = [int(x) for x in start_arg]
    end_arg = [int(x) for x in end_arg]

    start = datetime.date(start_arg[0], start_arg[1], start_arg[2])
    end = datetime.date(end_arg[0], end_arg[1], end_arg[2])
    delta = datetime.timedelta(days=args.time_delta)

    http = urllib3.PoolManager()

    cur = start
    while cur < end:
        print(cur.strftime('%Y-%m-%d'))
        # Morning
        url_locator = cur.strftime('%Y%m/%Y%m%d/gfsanl_4_%Y%m%d_0600_006.grb2')
        r = http.request('GET', URL + url_locator)
        while r.status != 200 and r.status != 404:
            continue
        if r.status == 200:
            save_data(r.data, cur, args.out, "0600")
        else:
            print(f"request for {url_locator} failed")
        # Afternoon
        url_locator = cur.strftime('%Y%m/%Y%m%d/gfsanl_4_%Y%m%d_1800_006.grb2')
        r = http.request('GET', URL + url_locator)
        while r.status != 200 and r.status != 404:
            continue
        if r.status == 200:
            save_data(r.data, cur, args.out, "1800")
        else:
            print(f"request for {url_locator} failed")
        # Next day
        cur = cur + delta


if __name__ == "__main__":
    main()
