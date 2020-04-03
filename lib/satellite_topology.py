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

import lib.sat_util as sutil


class Orbit:

    def __init__(self, arg_dict):
        """TODO: Docstring"""
        # Load parameters
        self.n_satellites = arg_dict['n_sat']
        self.inclination = float(arg_dict['inclination'])
        self.orbit_height = float(arg_dict['height'])
        self.initial_offset = float(arg_dict['initial_offset'])

        if "phase_shift" in arg_dict.keys():
            self.phase_shift = float(arg_dict['phase_shift'])
        else:
            self.phase_shift = 0
        if "failing_sats" in arg_dict.keys():
            self.failing_sats = arg_dict['failing_sats']
        else:
            self.failing_sats = None

        # Other instance variables
        self.radius = self.orbit_height + sutil.EARTH_RADIUS
        # Degrees of distance from one sat to another
        self.satellite_spacing = 360 / arg_dict['n_sat']
        self.period = sutil.compute_orbit_period(self.orbit_height)
        # Fin the vectors for orbit projection
        self.normal_vec, self.a_vec, self.b_vec = \
            self._compute_orbit_projection_vectors()

    def _compute_orbit_projection_vectors(self):
        """Compute the vectors needed for the projection of the 2D circular
        orbit in the 3D space.

        Returns:
        """
        normal_vec = sutil.sph2cart(1,
                                    90 + self.initial_offset,
                                    self.inclination)
        # Find the first position for which normal_vec is nonzero and use it to
        # find and orthogonal vector
        a = sutil.sph2cart(1, self.initial_offset, 90)
        b = np.cross(a, normal_vec)
        return normal_vec, a, b

    def compute_satellite_positions(self, time_instant, spherical=False):
        """Compute the positions of the satellites, given the time offset from
        instant 0.

        Args:
            time_instant: Time offset from instant 0.0 . In seconds.
            spherical: Boolean. If True, the positions of the satellites are
                returned in spherical coordinates.

        Returns:
            sat_pos: A numpy matrix of shape (n_satellites, 3), that represents
                the spacial position of each
                satellite in cartesian coordinates if spherical=False, in
                spherical coordinates otherwise.
        """
        # Find the positions on the 2D orbit circle
        time_angle_shift = time_instant / self.period * 360
        sat1_angle = time_angle_shift % 360
        if self.failing_sats:
            sat_idxs = np.delete(np.arange(self.n_satellites),
                                 self.failing_sats)
        else:
            sat_idxs = np.arange(self.n_satellites)
        sat_shifts = sat_idxs * self.satellite_spacing
        sat_angles = np.ones(sat_idxs.shape[0]) * sat1_angle + sat_shifts \
                     + self.phase_shift
        sat_angles = np.deg2rad(sat_angles)
        # Now in the 3D space
        x_pos = self.radius * (np.cos(sat_angles) * self.a_vec[0]
                               + np.sin(sat_angles) * self.b_vec[0])
        y_pos = self.radius * (np.cos(sat_angles) * self.a_vec[1]
                               + np.sin(sat_angles) * self.b_vec[1])
        z_pos = self.radius * (np.cos(sat_angles) * self.a_vec[2]
                               + np.sin(sat_angles) * self.b_vec[2])
        # Transpose to have the (3, n_sat) shape
        sat_pos = np.vstack((x_pos, y_pos, z_pos)).T

        if spherical:
            sat_pos = sutil.cart2sph(sat_pos)
        return sat_pos


class SatelliteTopology:

    def __init__(self, orbits_parameters):
        """ Loads the parameters into the orbits.

        Args:
            orbits_parameters: A dictionary containing the list of inactive
                orbital plains, called 'inactive_planes', and a list of
                dictionaries called 'planes', containing the description of
                each orbit in the constellation.
        """
        if orbits_parameters == {}:
            return
        self.planes_params = orbits_parameters['planes']
        if 'inactive_planes' in orbits_parameters:
            self.inactive_planes = orbits_parameters['inactive_planes']
        else:
            self.inactive_planes = []
        self.orbits = []
        self.num_sat = 0
        # Load orbits
        for plane_idx, cur in enumerate(self.planes_params):
            if plane_idx not in self.inactive_planes:
                # Load the plane only if it's not in the list of inactive planes
                self.orbits.append(Orbit(cur))
                self.num_sat += cur['n_sat']

    def compute_topology(self, time_instant, spherical=False):
        """Computes the position of all the satellites in the topology at the
        given time instant.

        Args:
            time_instant: Time from instant 0.0 in which to compute the
                topology (seconds).

        Returns:
            TODO
        """
        sat_pos = self.orbits[0].compute_satellite_positions(time_instant,
                                                             spherical=spherical)
        for cur_orb in self.orbits[1:]:
            sat_pos = np.vstack((
                sat_pos,
                cur_orb.compute_satellite_positions(time_instant,
                                                    spherical=spherical)))
        sat_pos = self.add_earth_rotation_shift(sat_pos, time_instant)
        return sat_pos

    def add_earth_rotation_shift(self, sat_position, time_instant):
        """Modify the position of the satellites according to earth's rotation.

        NOTE: The minus is b/c the relative motion is east-west
        """
        sph_pos = sutil.cart2sph(sat_position)
        deg_per_sec = 360 / sutil.DAY_DURATION
        deg_shift = time_instant * deg_per_sec
        # Shift the theta
        sph_pos[:, 1] = sph_pos[:, 1] - deg_shift
        cart_shifted = sutil.sph2cart_matrix(sph_pos)
        return cart_shifted
