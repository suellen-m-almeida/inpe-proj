#
# This file is part of Python Client Library for WTSS.
# Copyright (C) 2022 INPE.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.
#

"""Synthetic pixel centres laid on a real native grid, for offline grid tests.

A WTSS server answers an area query with the centres of the native raster cells
that fall inside the polygon, reprojected to lon/lat. These helpers do the same
without a server: they lay a grid in the coverage CRS (the BDC Albers or the
MODIS sinusoidal), keep the cells whose centre falls inside a lon/lat box, and
return those centres in lon/lat. Away from the projection's central meridian the
grid is rotated (Albers) or sheared (sinusoidal) in lon/lat, which is exactly
what broke the old latitude-based reconstruction.
"""

import numpy
from pyproj import Transformer

#: The BDC Albers equal-area grid of S2-16D-2, LANDSAT-16D-1 and CBERS4-WFI-16D-2.
ALBERS_BDC = ('+proj=aea +lat_0=-12 +lon_0=-54 +lat_1=-2 +lat_2=-22 +x_0=5000000 '
              '+y_0=10000000 +ellps=GRS80 +units=m +no_defs ')

#: The MODIS sinusoidal grid of mod13q1-6.1, with its pixel size and grid origin.
SINUSOIDAL_MODIS = ('+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 '
                    '+b=6371007.181 +units=m +no_defs')
MODIS_RES = 231.65635826395825
MODIS_ORIGIN = (-20015109.354, 10007554.677)


def native_cells_in_box(crs, res, origin, box):
    """Return the native cells whose centre falls inside a lon/lat box.

    Args:
        crs (str): The native CRS.
        res (float): The pixel size, in CRS units.
        origin (tuple): ``(x, y)`` of the grid's upper-left corner; cell centres
            sit at ``origin + (col + 0.5) * res`` east and ``(row + 0.5) * res``
            south of it.
        box (tuple): ``(west, south, east, north)`` in degrees.

    Returns:
        list: ``(row, col, lon, lat)`` for each cell, with ``row``/``col`` counted
        from the north-western cell of the returned set.
    """
    to_native = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    to_lonlat = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    west, south, east, north = box

    corner_x, corner_y = to_native.transform([west, east, west, east],
                                             [south, south, north, north])
    col_lo = int(numpy.floor((min(corner_x) - origin[0]) / res)) - 1
    col_hi = int(numpy.ceil((max(corner_x) - origin[0]) / res)) + 1
    row_lo = int(numpy.floor((origin[1] - max(corner_y)) / res)) - 1
    row_hi = int(numpy.ceil((origin[1] - min(corner_y)) / res)) + 1

    cols, rows = numpy.meshgrid(numpy.arange(col_lo, col_hi + 1),
                                numpy.arange(row_lo, row_hi + 1))
    x = origin[0] + (cols + 0.5) * res
    y = origin[1] - (rows + 0.5) * res
    lon, lat = to_lonlat.transform(x, y)
    inside = (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)

    rows, cols, lon, lat = rows[inside], cols[inside], lon[inside], lat[inside]
    return list(zip((rows - rows.min()).tolist(), (cols - cols.min()).tolist(),
                    lon.tolist(), lat.tolist()))


def grid_block(crs, res, origin, lon, lat, nrows, ncols):
    """Return an ``nrows`` x ``ncols`` block of native cells, the first holding a point.

    Returns:
        list: ``(row, col, lon, lat)`` for each cell, row by row from the north-west,
        with ``row``/``col`` counted within the block.
    """
    to_native = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    to_lonlat = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    x, y = to_native.transform(lon, lat)
    col0 = int(numpy.floor((x - origin[0]) / res))
    row0 = int(numpy.floor((origin[1] - y) / res))

    block = []
    for row in range(nrows):
        for col in range(ncols):
            cx = origin[0] + (col0 + col + 0.5) * res
            cy = origin[1] - (row0 + row + 0.5) * res
            clon, clat = to_lonlat.transform(cx, cy)
            block.append((row, col, float(clon), float(clat)))
    return block


def box_around(lon, lat, side_deg):
    """A ``(west, south, east, north)`` box of ``side_deg`` degrees centred on a point."""
    half = side_deg / 2
    return (lon - half, lat - half, lon + half, lat + half)


def pixel(lon, lat, timeline, values, res):
    """One WTSS area result: a pixel centre and its series over ``timeline``."""
    return {
        'pixel_center': {'type': 'Point', 'coordinates': [lon, lat]},
        'pixel_size': [res, -res],
        'time_series': {'timeline': list(timeline), 'values': {'NDVI': list(values)}},
    }


def coverage_metadata(name, crs, res, timeline):
    """The describe-coverage payload of a one-band (NDVI) coverage on ``crs``."""
    return {
        'fullname': name, 'description': 'synthetic',
        'bands': [{'name': 'NDVI', 'nodata': -9999, 'data_type': 'int16',
                   'resolution_x': res, 'resolution_y': res}],
        'bdc:crs': crs, 'raster_size': {'x': 1, 'y': 1},
        'extent': {'type': 'Polygon',
                   'coordinates': [[[-180, -90], [180, -90], [180, 90],
                                    [-180, 90], [-180, -90]]]},
        'timeline': list(timeline),
    }
