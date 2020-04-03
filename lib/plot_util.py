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

import cartopy as cart
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from cartopy.feature import COLORS
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def gridvalues_to_matrix(gridpoints, values, points_per_deg, val_min=None,
                         min_thrs=None, max_thrs=None, val_max=None):
    """Transform arrais of [lat, lon] -> val to a matrix with values."""
    values = np.asarray(values)
    grid = gridpoints.copy()
    grid[:, 0] = (grid[:, 0] + 90) * (1 / points_per_deg)
    grid[:, 1] = (grid[:, 1] + 180) * (1 / points_per_deg)
    grid = np.round(grid).astype(int)
    mat = np.zeros((int(180 / points_per_deg), int(360 / points_per_deg)))
    mat[grid[:, 0], grid[:, 1]] = values.reshape(-1)
    if min_thrs:
        mat[mat < min_thrs] = val_min
    if max_thrs:
        mat[mat > max_thrs] = val_max
    return mat


def plot_gridded_data(data, oceans=True, extent=(-180, 180, -90, 90),
                      origin='lower'):
    plt.figure(figsize=(24, 16))
    data_crs = ccrs.PlateCarree()
    projection = ccrs.PlateCarree()
    ax = plt.axes(projection=projection)
    ax.set_global()
    im = ax.imshow(data, extent=extent, origin=origin,
               transform=data_crs, cmap='Spectral_r')
    if oceans:
        OCEAN = cart.feature.NaturalEarthFeature('physical', 'ocean', '50m',
                                                 edgecolor='face',
                                                 facecolor=COLORS['water'],
                                                 zorder=-1)

        ax.add_feature(OCEAN, zorder=100, edgecolor='k')
    else:
        ax.coastlines()
    ax.set_extent(extent)

    gl = ax.gridlines(crs=data_crs, draw_labels=True, zorder=250,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.ylabels_left = False
    gl.xlines = False
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    plt.colorbar(im)
    return ax


def scatter_on_map(lats, lons, oceans=True, ax=None, annotate=False,
                   figsize=None, extent=(-180, 180, -89.9, 89.9), **kwargs):
    # Plotting

    data_crs = ccrs.PlateCarree()
    projection = ccrs.PlateCarree()
    if not ax:
        if figsize is None:
            figsize = plt.rcParams['figure.figsize']
        plt.figure(figsize=figsize)
        ax = plt.axes(projection=projection)
        ax.set_global()
        if oceans:
            ax.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k')
        else:
            ax.coastlines()
        ax.set_extent(extent)

    sc = ax.scatter(lons, lats, **kwargs, transform=data_crs, zorder=200)
    if 'c' in kwargs and not isinstance(kwargs['c'], str):
        plt.colorbar(sc, ax=ax)

    if annotate:
        for i in range(len(lons)):
            ax.annotate(i, (lons[i], lats[i]), zorder=200)
    return ax, data_crs


def plot_links(start_lat, start_lon, end_lat, end_lon, ax=None, transform=None,
               oceans=None, **kwargs):
    if not ax:
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        transform = ccrs.PlateCarree()
        ax.set_global()
        if oceans:
            ax.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k')
        ax.set_extent((-180, 180, -89.9, 89.9))

    for sla, slo, ela, elo in zip(start_lat, start_lon, end_lat, end_lon):
        if np.abs(slo - elo) > 180:
            if elo < 0:
                elo += +360
            elif elo >= 0:
                elo += -360
        x = [slo, elo]
        y = [sla, ela]
        ax.plot(x, y, transform=transform, zorder=150, **kwargs)


def plot_graph(graph, lats, lons, ax, **kwargs):
    n_nodes = graph.shape[0]
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if graph[i, j] < np.inf:
                ax.plot([lons[i], lons[j]], [lats[i], lats[j]], **kwargs)
    return ax


def plot_networkx(G, lats, lons, ax, transform, code=None, **kwargs):
    if code:
        max_weight = max(dict(G.edges).items(),
                         key=lambda x: x[1][code])
        max_weight = max_weight[1][code]
        cmap = mpl.cm.get_cmap('Spectral_r')

    for start, end in G.edges():
        slo = lons[start]
        elo = lons[end]
        if np.abs(slo - elo) > 180:
            if elo < 0:
                elo += +360
            elif elo >= 0:
                elo += -360
        if code:
            cur_weight = G[start][end][code]
            ax.plot([slo, elo], [lats[start], lats[end]], transform=transform,
                    color=cmap(cur_weight / max_weight))
        else:
            ax.plot([slo, elo], [lats[start], lats[end]], transform=transform,
                    **kwargs)
    return ax
