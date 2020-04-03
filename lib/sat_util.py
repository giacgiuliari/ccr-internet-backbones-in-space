# The MIT License (MIT)
#
# Copyright (c) 2020 Giacomo Giuliari
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import matplotlib as mpl
import os
if "DISPLAY" not in os.environ:
    mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.constants import G
from scipy.spatial.distance import euclidean
# from modules.sumatra_helper import write_smt_txt

# Average great-circle radius in kilometers.
EARTH_RADIUS = 6371.230

# Average duration of a day in seconds, from Wikipedia.
DAY_DURATION = 86400

# Earth mass
EARTH_MASS = 5.9722e24

# Atmosphere
ATMOSPHERE_HEIGHT = 100

# Standard Gravitational Parameter for earth
MU = G * EARTH_MASS

# Speed of light in km/s
LIGHTSPEED = 299792


def compute_orbit_period(orbit_height):
    """Compute the period of an orbit.

    Args:
        orbit_height: The height of the orbit from the surface of the earth,
            in kilometers.

    Returns:
        period: The period of the orbit, in seconds.
    """
    radius = (orbit_height + EARTH_RADIUS) * 1000
    period = 2 * np.pi * np.sqrt(np.power(radius, 3) / MU)
    return period


def sph2cart(r, theta, phi):
    """Converts spherical coordinates to cartesian.

    Args:
        r: Radius.
        theta: Orizontal angle. In degrees.
        phi: Vertical angle (from z axis). In degrees

    Returns:
        np.ndarray([x, y, z]): Numpy array containg the cartesian coordinates.
    """
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.asarray([x, y, z])


def sph2cart_matrix(sph):
    r = sph[:, 0]
    theta = np.deg2rad(sph[:, 1])
    phi = np.deg2rad(sph[:, 2])
    x = np.multiply(r, np.multiply(np.sin(phi), np.cos(theta)))
    y = np.multiply(r, np.multiply(np.sin(phi), np.sin(theta)))
    z = np.multiply(r, np.cos(phi))
    sph_matrix = np.hstack((x, y, z))
    return sph_matrix


def cart2sph(cart):
    """Cartesian to spherical coordinates.

    Args:
        cart: Matrix (N, 3) with the spherical coordinates,
        [radius, theta, phi]. Angles in degrees.
    """
    cart = np.matrix(cart)
    radius = np.sqrt(np.sum(np.square(cart), axis=1))
    theta = np.arctan2(cart[:, 1], cart[:, 0])
    phi = np.arctan2((np.sqrt(np.sum(np.square(cart[:, 0:2]), axis=1))),
                     cart[:, 2])
    spherical = np.hstack((radius, theta, phi))
    spherical[:, 1:3] = np.rad2deg(spherical[:, 1:3])
    return spherical


def pointwise_distance_sph(point1, point2):
    """Euclidean distance between two points in spherical coordinates.

    See:
    https://math.stackexchange.com/questions/833002/distance-between-two-points-in-spherical-coordinates

    Args:
        point1: First point in spherical coordinates. Angles in degrees.
        point2: First point in spherical coordinates. Angles in degrees.
    """
    cart1 = sph2cart(point1[0], point1[1], point1[2])
    cart2 = sph2cart(point2[0], point2[1], point2[2])
    return euclidean(cart1, cart2)


def latlon2cart(point):
    """Converts a (lat, lon) pair on earth to a (x, y, z) coordinate."""
    lat, lon = point
    cart = sph2cart(EARTH_RADIUS, lon, 90 - lat)
    return cart


def format_print(name, value):
    print("{:.<40}{:.>40}".format(name, str(value)))


