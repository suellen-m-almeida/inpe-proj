#
# This file is part of Python Client Library for WTSS.
# Copyright (C) 2024 INPE.
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
"""A class that represents a Time Series in WTSS."""

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy
import shapely.geometry

from .summarize import Summarize
from .utils import render_html

Series = Dict[str, List[Union[float, str]]]
"""Represent the time series context attributes.

The keys available are:
- ``timeline``: List of given location series.
- ``values``: The map attributes and time series."""

#: Two pixel centres closer than this (in degrees, about 11 cm) belong to the same
#: grid line. Reprojecting a raster to lon/lat leaves each centre with a slightly
#: different value, so a raster row arrives as hundreds of near-identical
#: coordinates; the threshold sits far above that noise and far below the spacing
#: of any Earth-observation raster, the finest of which are around half a metre.
_SAME_CELL_DEGREES = 1e-6


@dataclass
class Location:
    """Represent the time series location.

    Once time series request is made in :class:`wtss.coverage.Coverage`, the location is created
    and related with time series attributes and values found for location.
    These values may be dynamically set using pagination.
    """

    geom: shapely.geometry.Point
    series: Series
    pixel_size: Tuple[float, float]

    @classmethod
    def from_dict(cls, pixel_center: Any, pixel_size: Any, time_series: Series):
        """Create a WTSS Location from dict keys found in Time Series result."""
        geom = shapely.geometry.shape(pixel_center)
        return cls(geom, time_series, pixel_size=pixel_size)

    @property
    def x(self):
        """Retrieve location longitude."""
        return self.geom.x

    @property
    def y(self):
        """Retrieve location latitude."""
        return self.geom.y

    @property
    def timeline(self) -> List[str]:
        """Retrieve the location time series associated."""
        return self.series['timeline']

    def extend(self, other: 'Location'):
        """Extend time line values into current context."""
        other_timeline = other.series['timeline']
        other_series = other.series['values']

        self.series['timeline'].extend(other_timeline)
        for attribute, series in other_series.items():
            self.series['values'][attribute].extend(series)


class TimeSeries:
    """A class that represents a time series in WTSS.

    .. note::

        For more information about time series definition, please, refer to
        `WTSS specification <https://github.com/brazil-data-cube/wtss-spec>`_.
    """

    def __init__(self, coverage: 'Coverage', data, **options):
        """Create a TimeSeries object associated to a coverage.

        Args:
            coverage (Coverage): The coverage that this time series belongs to.
        """
        #: Coverage: The associated coverage.
        self._coverage = coverage
        self._options = options
        self._data = data
        self._pagination = None
        if data.get('pagination'):
            self._pagination = data['pagination']

        self._locations = {}
        for location_ts in data['results']:
            location = Location.from_dict(**location_ts)
            self._locations[(location.x, location.y)] = location

    @property
    def total_locations(self):
        """Return the computed locations in timeseries."""
        return len(self._data['results'])

    @property
    def timeline(self):
        """Return the timeline associated to the time series."""
        return self._data['results'][0]['time_series']['timeline'] if len(self._data["results"]) > 0 else []

    @property
    def attributes(self):
        """Return a list with attribute names selected by user."""
        return [attr for attr in self._data['query']['attributes']]

    def values(self, attr_name) -> List[List[float]]:
        """Return the time series for the given attribute."""
        entries = [
            location.series['values'][attr_name] for location in self._locations.values()
        ]

        return entries

    def _values_for(self, location, attr_name, apply_scale, mask_nodata):
        """Return a location's samples for an attribute, applying band metadata."""
        raw = location.series['values'][attr_name]
        if not (apply_scale or mask_nodata) or self._coverage is None:
            return raw
        band_meta = self._coverage.band(attr_name)
        return self._coverage._apply_metadata(raw, band_meta, apply_scale, mask_nodata)

    def series(self, attr_name: str, apply_scale: bool = False, mask_nodata: bool = False):
        """Return the time series of an attribute as a time-aligned pandas Series.

        Unlike :meth:`values`, the result carries the timeline (as a
        ``DatetimeIndex``) and the location, so values are correlated without
        digging into private structures (addresses B9).

        - **Single location**: a :class:`pandas.Series` indexed by datetime.
        - **Multiple locations**: a :class:`pandas.Series` with a MultiIndex
          ``(datetime, location)``, where ``location`` is the pixel-center
          ``(longitude, latitude)`` tuple. Use ``.unstack('location')`` to get a
          wide :class:`pandas.DataFrame` (one column per location).

        Args:
            apply_scale (bool): Apply the band scale/offset client-side (Ciclo iv).
            mask_nodata (bool): Replace nodata samples with ``NaN`` (Ciclo iv).

        Raises:
            KeyError: If ``attr_name`` is not present in the time series.
            ImportError: If pandas could not be imported.
        """
        try:
            import pandas
        except ImportError:
            raise ImportError('You should install pandas!')

        locations = list(self._locations.values())

        if not locations:
            return pandas.Series([], dtype='float64', name=attr_name)

        if attr_name not in locations[0].series['values']:
            raise KeyError(
                f"Attribute '{attr_name}' not found. Available: "
                f"{list(locations[0].series['values'])}"
            )

        if len(locations) == 1:
            location = locations[0]
            index = pandas.to_datetime(location.timeline)
            return pandas.Series(self._values_for(location, attr_name, apply_scale, mask_nodata),
                                 index=index, name=attr_name)

        # Multiple locations: build a (datetime, location) MultiIndex Series.
        times, locs, vals = [], [], []
        for location in locations:
            label = (location.x, location.y)
            values = self._values_for(location, attr_name, apply_scale, mask_nodata)
            for moment, value in zip(location.timeline, values):
                times.append(moment)
                locs.append(label)
                vals.append(value)

        index = pandas.MultiIndex.from_arrays(
            [pandas.to_datetime(times), locs], names=['datetime', 'location'])

        return pandas.Series(vals, index=index, name=attr_name).sort_index()

    def __getitem__(self, attr_name: str):
        """Return ``series(attr_name)`` with raw values (see :meth:`series`)."""
        return self.series(attr_name)

    def df(self, format: str = 'long', apply_scale: bool = False, mask_nodata: bool = False):
        """Return the time series as a pandas DataFrame.

        Builds on :meth:`series`, so values are always aligned with their
        timeline and location (addresses the same gap as B9, Ciclo iii).

        Args:
            format (str): Either ``'long'`` (default) or ``'wide'``.

                - ``'long'``: one row per (datetime, attribute, location), with a
                  single ``value`` column. Index is ``(datetime, attribute,
                  location)``.
                - ``'wide'``: index is the ``datetime``. Columns are the
                  attributes for a single location, or a ``(attribute, location)``
                  MultiIndex for several locations.
            apply_scale (bool): Apply the band scale/offset client-side (Ciclo iv).
            mask_nodata (bool): Replace nodata samples with ``NaN`` (Ciclo iv).

        Raises:
            ValueError: If ``format`` is not ``'long'`` or ``'wide'``.
            ImportError: If pandas could not be imported.
        """
        try:
            import pandas
        except ImportError:
            raise ImportError('You should install pandas!')

        if format not in ('long', 'wide'):
            raise ValueError(f"format must be 'long' or 'wide', got {format!r}")

        attributes = self.attributes
        locations = list(self._locations.values())

        if not locations:
            return pandas.DataFrame()

        if format == 'long':
            times, attrs, locs, vals = [], [], [], []
            for attr in attributes:
                for location in locations:
                    label = (location.x, location.y)
                    values = self._values_for(location, attr, apply_scale, mask_nodata)
                    for moment, value in zip(location.timeline, values):
                        times.append(moment)
                        attrs.append(attr)
                        locs.append(label)
                        vals.append(value)
            frame = pandas.DataFrame({
                'datetime': pandas.to_datetime(times),
                'attribute': attrs,
                'location': locs,
                'value': vals,
            })
            return frame.set_index(['datetime', 'attribute', 'location'])

        # wide
        columns = {attr: self.series(attr, apply_scale, mask_nodata) for attr in attributes}

        if len(locations) == 1:
            frame = pandas.DataFrame(columns)
            frame.index.name = 'datetime'
            frame.columns.name = 'attribute'
            return frame

        parts = {attr: series.unstack('location') for attr, series in columns.items()}
        return pandas.concat(parts, axis=1, names=['attribute', 'location'])

    def _grid_cube(self, attr_name, apply_scale: bool = False, mask_nodata: bool = False):
        """Rebuild the native raster of one attribute as a ``(time, y, x)`` cube.

        Pixel centres reprojected to lon/lat are usually **not** a regular lon/lat
        grid: e.g. MODIS in a sinusoidal projection has the column longitudes drift
        from row to row. So we group pixels by the axis with fewer distinct values
        into **rows**, order the other axis into **columns**, and return the actual
        longitudes/latitudes as 2D arrays. ``y``/``x`` are therefore row/column
        indices; cells with no returned pixel stay ``NaN``.

        Returns:
            tuple: ``(lon2d, lat2d, cube)`` — 2D longitude and latitude arrays of
            shape ``(ny, nx)`` and a cube of shape ``(len(timeline), ny, nx)``.
        """
        import numpy
        from collections import defaultdict

        locations = list(self._locations.values())
        ntime = len(self.timeline)

        if not locations:
            empty = numpy.zeros((0, 0), dtype='float64')
            return empty, empty, numpy.zeros((ntime, 0, 0), dtype='float64')

        lats = numpy.array([loc.y for loc in locations], dtype='float64')
        lons = numpy.array([loc.x for loc in locations], dtype='float64')
        lat_slot, lat_centres = self._grid_slots(lats)
        lon_slot, lon_centres = self._grid_slots(lons)

        # Rows along the "cleaner" axis (fewer distinct slots); columns along the
        # other. Rows run north -> south when latitude drives them.
        by_lat = lat_centres.size <= lon_centres.size
        row_slot = lat_slot if by_lat else lon_slot
        col_value = lons if by_lat else lats

        buckets = defaultdict(list)
        for i, loc in enumerate(locations):
            buckets[int(row_slot[i])].append((float(col_value[i]), i))
        row_keys = sorted(buckets, reverse=by_lat)  # north (max lat) first
        rows = [sorted(buckets[key]) for key in row_keys]

        # Column spacing, taken from the gaps actually observed inside the rows.
        spacings = [numpy.diff([value for value, _ in row])
                    for row in rows if len(row) > 1]
        step = float(numpy.median(numpy.concatenate(spacings))) if spacings else 0.0

        # Column slots come from where each row *starts*, not from the position of
        # the pixel inside it. A lon/lat box cuts the reprojected grid unevenly, so
        # neighbouring rows may hold a different number of pixels; packing them all
        # from column zero would shift a whole row by one cell against its
        # neighbours. Offsets accumulate row by row because the reprojected grid is
        # curvilinear: consecutive rows drift by a fraction of a cell, which rounds
        # to zero, while a genuinely missing edge pixel rounds to one.
        offsets = [0]
        for previous, current in zip(rows, rows[1:]):
            drift = 0
            if step:
                drift = int(round((current[0][0] - previous[0][0]) / step))
            offsets.append(offsets[-1] + drift)
        base = min(offsets)
        offsets = [offset - base for offset in offsets]

        nrows = len(rows)
        ncols = max((offset + len(row) for offset, row in zip(offsets, rows)),
                    default=0)

        cube = numpy.full((ntime, nrows, ncols), numpy.nan, dtype='float64')
        lon2d = numpy.full((nrows, ncols), numpy.nan, dtype='float64')
        lat2d = numpy.full((nrows, ncols), numpy.nan, dtype='float64')
        for ri, (offset, row) in enumerate(zip(offsets, rows)):
            for position, (_, index) in enumerate(row):
                ci = offset + position
                loc = locations[index]
                values = self._values_for(loc, attr_name, apply_scale, mask_nodata)
                n = min(len(values), ntime)
                cube[:n, ri, ci] = values[:n]
                lon2d[ri, ci] = loc.x
                lat2d[ri, ci] = loc.y

        # A cell without a pixel still has a position: the grid geometry defines it,
        # and only its *value* is missing. Leaving the coordinates as NaN would make
        # the cube unplottable (pcolormesh rejects non-finite coordinates), so each
        # row extrapolates them from the cells it did receive -- rows are evenly
        # spaced, so a straight line through the known ones is exact.
        self._fill_row_coordinates(lon2d)
        self._fill_row_coordinates(lat2d)
        return lon2d, lat2d, cube

    @staticmethod
    def _fill_row_coordinates(grid):
        """Complete each row of a coordinate grid by linear extrapolation, in place."""
        import numpy

        columns = numpy.arange(grid.shape[1], dtype='float64')
        for ri, row in enumerate(grid):
            known = numpy.isfinite(row)
            if known.all() or known.sum() < 2:
                continue
            slope, intercept = numpy.polyfit(columns[known], row[known], 1)
            grid[ri, ~known] = slope * columns[~known] + intercept

    @staticmethod
    def _grid_slots(values):
        """Cluster nearly-equal coordinates into grid slots, tolerating float noise.

        Reprojected pixel centres repeat a coordinate with tiny floating-point
        differences: a Sentinel-2 area of 56 raster rows comes back with a distinct
        latitude for every one of its pixels. Grouping by exact equality would turn
        each pixel into its own row, so slots are cut only where consecutive values
        jump by more than the noise floor.

        The split uses a physical threshold rather than one inferred from the data:
        two centres less than :data:`_SAME_CELL_DEGREES` apart are the same grid
        line, because no Earth-observation raster has pixels that fine. Reading the
        threshold off the data was tried first and proved brittle -- the observed
        noise floor ranges over four orders of magnitude between collections, and a
        grid with rows missing produces the same large jump as noise does.

        Args:
            values (numpy.ndarray): The coordinate of every location, in order.

        Returns:
            tuple: ``(slot, centres)`` -- the slot index of each input value, and
            the representative coordinate of each slot.
        """
        import numpy

        values = numpy.asarray(values, dtype='float64')
        unique = numpy.unique(values)
        if unique.size < 2:
            return numpy.zeros(values.shape, dtype=int), unique

        edges = numpy.flatnonzero(numpy.diff(unique) > _SAME_CELL_DEGREES) + 1

        centres = numpy.array([group.mean() for group in numpy.split(unique, edges)])
        slot = numpy.abs(values[:, None] - centres[None, :]).argmin(axis=1)
        return slot, centres

    def to_xarray(self, apply_scale: bool = False, mask_nodata: bool = False,
                  grid: bool = False):
        """Return the time series as a labelled :class:`xarray.Dataset` (Ciclo v).

        Each attribute becomes a data variable. Dimensions are:

        - **Single location**: ``(time,)``; ``longitude``/``latitude`` are scalar
          coordinates.
        - **Multiple locations**: ``(time, location)``; ``longitude``/``latitude``
          are coordinates along ``location``.
        - **Grid (polygon/area)**, with ``grid=True``: ``(time, y, x)``, rebuilding
          the 2D raster from the sparse pixel centres. ``x``/``y`` are the distinct
          longitudes/latitudes; grid cells with no returned pixel are ``NaN``.

        Args:
            apply_scale (bool): Apply the band scale/offset client-side (Ciclo iv).
            mask_nodata (bool): Replace nodata samples with ``NaN`` (Ciclo iv).
            grid (bool): Rebuild the 2D ``(time, y, x)`` raster from the pixels of
                a polygon/area query, instead of the flat ``(time, location)``.
                Default is False.

        Raises:
            ImportError: If xarray (or pandas/numpy) could not be imported.
        """
        try:
            import numpy
            import pandas
            import xarray
        except ImportError:
            raise ImportError('You should install xarray (and numpy, pandas)!')

        attributes = self.attributes
        locations = list(self._locations.values())

        if not locations:
            return xarray.Dataset()

        time = pandas.to_datetime(self.timeline)

        if grid:
            # Polygon/area case (Ciclo v): rebuild the (time, y, x) raster. y/x are
            # row/column indices; the real coordinates are 2D longitude/latitude
            # arrays because the reprojected grid is curvilinear (see _grid_cube).
            data_vars = {}
            lon2d = lat2d = None
            for attr in attributes:
                lon2d, lat2d, cube = self._grid_cube(attr, apply_scale, mask_nodata)
                data_vars[attr] = (('time', 'y', 'x'), cube)
            ny, nx = lon2d.shape
            return xarray.Dataset(
                data_vars,
                coords={
                    'time': time,
                    'y': numpy.arange(ny),
                    'x': numpy.arange(nx),
                    'longitude': (('y', 'x'), lon2d),
                    'latitude': (('y', 'x'), lat2d),
                },
            )

        if len(locations) == 1:
            location = locations[0]
            data_vars = {
                attr: ('time', self._values_for(location, attr, apply_scale, mask_nodata))
                for attr in attributes
            }
            return xarray.Dataset(
                data_vars,
                coords={'time': time, 'longitude': location.x, 'latitude': location.y},
            )

        # Multiple locations: dims (time, location).
        data_vars = {}
        for attr in attributes:
            # rows = locations, columns = time; transpose to (time, location).
            matrix = numpy.array(
                [self._values_for(location, attr, apply_scale, mask_nodata)
                 for location in locations],
                dtype='float64',
            ).T
            data_vars[attr] = (('time', 'location'), matrix)

        return xarray.Dataset(
            data_vars,
            coords={
                'time': time,
                'location': numpy.arange(len(locations)),
                'longitude': ('location', [location.x for location in locations]),
                'latitude': ('location', [location.y for location in locations]),
            },
        )

    def to_netcdf(self, path=None, apply_scale: bool = False, mask_nodata: bool = False, **kwargs):
        """Export the time series to NetCDF, via the labelled :meth:`to_xarray`.

        Args:
            path (str, optional): Destination file. When ``None`` (default), the
                NetCDF document is returned as ``bytes`` instead of being written.
            apply_scale (bool): Apply the band scale/offset client-side (Ciclo iv).
            mask_nodata (bool): Replace nodata samples with ``NaN`` (Ciclo iv).
            **kwargs: Forwarded to :meth:`xarray.Dataset.to_netcdf`
                (e.g. ``engine``, ``encoding``).

        Raises:
            ImportError: If xarray or a NetCDF backend is unavailable.
        """
        dataset = self.to_xarray(apply_scale=apply_scale, mask_nodata=mask_nodata)
        return dataset.to_netcdf(path, **kwargs)

    def to_zarr(self, path, apply_scale: bool = False, mask_nodata: bool = False,
                grid: bool = False, **kwargs):
        """Export the time series to a Zarr store, via :meth:`to_xarray` (semana 10).

        Args:
            path (str): Destination Zarr store (a directory).
            apply_scale (bool): Apply the band scale/offset client-side (Ciclo iv).
            mask_nodata (bool): Replace nodata samples with ``NaN`` (Ciclo iv).
            grid (bool): Export the ``(time, y, x)`` raster cube instead of the flat
                ``(time, location)`` layout (see :meth:`to_xarray`). Default False.
            **kwargs: Forwarded to :meth:`xarray.Dataset.to_zarr`.

        Raises:
            ImportError: If xarray or the ``zarr`` backend is unavailable.
        """
        try:
            import zarr  # noqa: F401  (backend used by xarray)
        except ImportError:
            raise ImportError('Install zarr for Zarr export: pip install zarr')
        dataset = self.to_xarray(apply_scale=apply_scale, mask_nodata=mask_nodata, grid=grid)
        return dataset.to_zarr(path, **kwargs)

    def to_parquet(self, path, apply_scale: bool = False, mask_nodata: bool = False,
                   format: str = 'long', **kwargs):
        """Export the time series to Parquet, via :meth:`df` (semana 10).

        Args:
            path (str): Destination ``.parquet`` file.
            apply_scale (bool): Apply the band scale/offset client-side (Ciclo iv).
            mask_nodata (bool): Replace nodata samples with ``NaN`` (Ciclo iv).
            format (str): ``'long'`` (default) or ``'wide'`` — see :meth:`df`.
            **kwargs: Forwarded to :meth:`pandas.DataFrame.to_parquet`.

        Raises:
            ImportError: If pandas or a Parquet engine (pyarrow) is unavailable.
        """
        try:
            import pyarrow  # noqa: F401  (default parquet engine)
        except ImportError:
            raise ImportError('Install pyarrow for Parquet export: pip install pyarrow')

        frame = self.df(format=format, apply_scale=apply_scale, mask_nodata=mask_nodata)

        if format == 'wide':
            # Column labels may be an (attribute, location) tuple; Parquet needs
            # plain string names. The datetime index is kept.
            frame = frame.copy()
            frame.columns = ['_'.join(map(str, c)) if isinstance(c, tuple) else str(c)
                             for c in frame.columns]
            return frame.to_parquet(path, **kwargs)

        # long: flatten to a fully columnar table. The 'location' level is a
        # (lon, lat) tuple, which Parquet cannot store, so split it into columns.
        frame = frame.reset_index()
        if 'location' in frame.columns:
            frame['longitude'] = [loc[0] for loc in frame['location']]
            frame['latitude'] = [loc[1] for loc in frame['location']]
            frame = frame.drop(columns='location')
        return frame.to_parquet(path, index=False, **kwargs)

    def to_geotiff(self, path, attribute: str, apply_scale: bool = False,
                   mask_nodata: bool = False, crs: str = 'EPSG:4326', **kwargs):
        """Export a polygon/area band as a multi-band GeoTIFF (semana 10).

        Rebuilds the ``(time, y, x)`` raster of ``attribute`` (see :meth:`to_xarray`
        with ``grid=True``) and writes it as a north-up GeoTIFF whose **bands are
        the timestamps**. Cells with no pixel are written as ``NaN`` (the file's
        nodata). The affine transform comes from the pixel-centre spacing.

        Args:
            path (str): Destination ``.tif`` file.
            attribute (str): The band to export.
            apply_scale (bool): Apply the band scale/offset client-side (Ciclo iv).
            mask_nodata (bool): Replace nodata samples with ``NaN`` (Ciclo iv).
            crs (str): Coordinate reference system. Defaults to ``'EPSG:4326'``.
            **kwargs: Extra fields merged into the rasterio profile.

        Raises:
            ValueError: If the area is smaller than 2x2 pixels.
            KeyError: If ``attribute`` is not present in the time series.
            ImportError: If rasterio is unavailable.
        """
        try:
            import numpy
            import rasterio
            from rasterio.transform import from_origin
        except ImportError:
            raise ImportError('Install rasterio for GeoTIFF export: pip install rasterio')

        locations = list(self._locations.values())
        if locations and attribute not in locations[0].series['values']:
            raise KeyError(
                f"Attribute '{attribute}' not found. Available: "
                f"{list(locations[0].series['values'])}"
            )

        lon2d, lat2d, cube = self._grid_cube(attribute, apply_scale, mask_nodata)
        ny, nx = cube.shape[1], cube.shape[2]
        if nx < 2 or ny < 2:
            raise ValueError('to_geotiff needs an area (a grid of at least 2x2 pixels).')

        # North-up: row 0 must be the northernmost latitude.
        if numpy.nanmean(lat2d[0]) < numpy.nanmean(lat2d[-1]):
            cube = cube[:, ::-1, :]
            lon2d, lat2d = lon2d[::-1], lat2d[::-1]

        # Affine transform from the mean pixel spacing. The reprojected grid is
        # slightly curvilinear, so this is a nearest-regular-grid approximation.
        dx = abs(float(numpy.nanmean(numpy.diff(lon2d, axis=1))))
        dy = abs(float(numpy.nanmean(numpy.diff(lat2d, axis=0))))
        transform = from_origin(float(numpy.nanmin(lon2d)) - dx / 2,
                                float(numpy.nanmax(lat2d)) + dy / 2, dx, dy)
        data = cube.astype('float64')
        ntime = data.shape[0]

        profile = {
            'driver': 'GTiff', 'height': ny, 'width': nx, 'count': ntime,
            'dtype': 'float64', 'crs': crs, 'transform': transform,
            'nodata': float('nan'),
        }
        profile.update(kwargs)

        with rasterio.open(path, 'w', **profile) as dst:
            for i in range(ntime):
                dst.write(data[i], i + 1)
                dst.set_band_description(i + 1, str(self.timeline[i])[:10])
        return path

    @property
    def locations(self) -> dict:
        """Retrieve the time series locations matched as dict.

        Each location is a shapely.geometry.Point representing the pixel center.
        """
        return self._locations

    def summarize(self,
                  operations: Optional[List[str]] = None,
                  masked: Optional[bool] = False,
                  mask: Any = None,
                  start_datetime: Optional[str] = None,
                  end_datetime: Optional[str] = None) -> Summarize:
        """Summarize the current Time Series object."""
        if not start_datetime and not end_datetime:
            start_datetime = self._data['query']['start_datetime']
            end_datetime = self._data['query']['end_datetime']
            if self._pagination:
                start_datetime = self._pagination['start_datetime']
                end_datetime = self._pagination['end_datetime']

        return self._coverage.summarize(operations=operations,
                                        masked=masked,
                                        mask=mask,
                                        geom=self._data['query']['geom'],
                                        start_datetime=start_datetime,
                                        end_datetime=end_datetime,
                                        attributes=self._data['query']['attributes'])

    def _prepared_values(self, attr_def, raw, apply_scale: bool, mask_nodata: bool):
        """Turn a raw sample list into plot-ready values.

        Honors the ``apply_scale``/``mask_nodata`` flags. Without them, keeps the
        legacy behavior (mask nodata to ``None`` so it draws as a gap, raw scale).
        When scaling, nodata is always masked so the sentinel is never scaled.

        Shared by :meth:`plot` and :meth:`_plot_interactive` so both renderings
        show exactly the same numbers.
        """
        if apply_scale or mask_nodata:
            return self._coverage._apply_metadata(
                list(raw), attr_def, apply_scale, mask_nodata or apply_scale)

        nodata = attr_def['nodata']
        return [value if value != nodata else None for value in raw]

    def plot(self, stats: bool = True, limit: Optional[int] = None,
             pixels: bool = True, apply_scale: bool = False,
             mask_nodata: bool = False, interactive: bool = False, **options):
        """Plot the time series on a chart.

        Args:
            stats (bool): Flag to display time series statistics. Default is True.
                (Only applied on Time Series per Area)
            limit (Optiona[int]): Limit the number of time series to plot. Default is None.
                When None, display all. You may have performance issues.
            pixels (bool): Draw one faint line per pixel/location. Default is True.
                Set to False to show only the statistics (median and quartiles),
                which is cleaner when ``stats`` is on and there are many pixels.
            apply_scale (bool): Apply the band ``scale``/``offset`` client-side, so
                the Y axis is in the physical unit (e.g. NDVI in -1..1) instead of
                raw integers. Default is False. When on, nodata is always masked so
                the sentinel is never scaled.
            mask_nodata (bool): Replace nodata samples with ``NaN`` (drawn as gaps).
                Default is False; note the plot already hides nodata visually, so
                this mainly matters combined with ``apply_scale``.
            interactive (bool): Render with plotly instead of matplotlib, so the
                chart can be zoomed and panned and each sample shows its date,
                value and pixel on hover. Default is False. Unlike the matplotlib
                rendering, which draws and returns ``None``, this returns the
                :class:`plotly.graph_objects.Figure` for further customization.
                Requires ``plotly`` (``pip install wtss[interactive]``).

        Keyword Args:
            attributes (sequence): A sequence like ('red', 'nir') or ['red', 'nir'] .
            line_styles (sequence): Not implemented yet.
            markers (sequence): Not implemented yet.
            line_width (numeric): Not implemented yet.
            line_widths (sequence): Not implemented yet,
            labels (sequence): Not implemented yet.
        Raises:
            ImportError: If datetime, matplotlib or numpy or datetime could not be imported.
        """
        if limit is not None and limit < 0:
            raise ValueError('Limit cannot be negative')

        if interactive:
            return self._plot_interactive(stats=stats, limit=limit, pixels=pixels,
                                          apply_scale=apply_scale,
                                          mask_nodata=mask_nodata, **options)

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('You should install Matplotlib and Numpy!')

        # Get attribute value if user defined, otherwise use the first
        attributes = options.get('attributes') or self.attributes

        if not attributes:
            raise ValueError('No attributes available to plot.')

        axes = options.get("axes")
        fig = options.get("fig")

        if fig is None or axes is None:
            fig, axes = plt.subplots(len(attributes), figsize=(16, 5*len(attributes)))
            # fig, axes = plt.subplots(len(attributes), figsize=plt.figaspect(0.5))

            if len(attributes) == 1:
                axes = [axes]

        x = [dt.datetime.fromisoformat(d.replace('Z', '+00:00')) for d in self.timeline]

        attribute_map = {
            attr['name']: attr
            for attr in self._coverage.attributes
        }

        locations = list(self._locations.values())

        summarize_kwargs = {}
        if locations:
            summarize_kwargs["start_datetime"] = locations[0].timeline[0]
            summarize_kwargs["end_datetime"] = locations[0].timeline[-1]
        summarize = self.summarize(**summarize_kwargs)

        # Compute the number of locations to draw once, before the loop, so it
        # is always defined even when the per-attribute loop does not run.
        if limit is None:
            _limit = len(self._locations)
        else:
            _limit = min(limit, len(self._locations))

        alpha = 0.2 if _limit > 100 else 0.6

        for idx, axis in enumerate(axes):
            band_name = attributes[idx]
            attr_def = attribute_map[band_name]

            def _prep(raw, attr_def=attr_def):
                return self._prepared_values(attr_def, raw, apply_scale, mask_nodata)

            if pixels:
                for location in locations[:_limit]:
                    values = _prep(location.series['values'][band_name])
                    # The pixel band is self-explanatory, so it is kept out of the legend.
                    axis.plot(x, values, ls='-', linewidth=1, color='#7F9BB1', alpha=alpha,
                              label='_nolegend_')

            if stats:
                for i, quantile_name in enumerate(['q1', 'q3']):
                    quantile = _prep(summarize.values(band_name).values(quantile_name))
                    # Label only the first quantile so q1 and q3 share one entry.
                    axis.plot(x, quantile[:len(x)], color='#b19541', linewidth=1.5,
                              label='quartis (q1, q3)' if i == 0 else '_nolegend_')

                median = _prep(summarize.values(band_name).values('median'))
                axis.plot(x, median[:len(x)], label='mediana',
                          color='#B16240', linewidth=2.5)

            # Legend for the median and quartiles (only drawn when stats are on).
            if stats:
                axis.legend(loc='upper right', fontsize=12)

            axis.text(-0.05, 0.5, f"{band_name}", transform=axis.transAxes, rotation=90, va='center', ha='left', fontsize=20) #Adjust subplot title position. Displacement ranges from 0 to 1. Negative to left and bottom, positive to right and up.
            fig.canvas.draw()

        title = 'Time Series'
        if _limit < len(self._locations):
            title += f' (Showing {_limit} of {len(self._locations)} points)'

        fig.suptitle(title, fontsize=22, y=0.99)

        fig.subplots_adjust(top=0.90) #Adjust distance the position of subplot border to figure border.

        fig.autofmt_xdate()

        # Show only after every series has been drawn.
        fig.show()

    def _plot_interactive(self, stats: bool = True, limit: Optional[int] = None,
                          pixels: bool = True, apply_scale: bool = False,
                          mask_nodata: bool = False, **options):
        """Render the time series with plotly (see ``interactive`` in :meth:`plot`).

        Mirrors the matplotlib rendering — same series, same colors, same
        statistics — but returns the figure instead of drawing it, so the caller
        can adjust the layout before displaying.

        Returns:
            plotly.graph_objects.Figure: One subplot per attribute.

        Raises:
            ImportError: If plotly could not be imported.
            ValueError: If there is no attribute to plot.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError('Install plotly for interactive plots: '
                              'pip install wtss[interactive]')

        attributes = options.get('attributes') or self.attributes

        if not attributes:
            raise ValueError('No attributes available to plot.')

        x = [dt.datetime.fromisoformat(d.replace('Z', '+00:00')) for d in self.timeline]

        attribute_map = {attr['name']: attr for attr in self._coverage.attributes}
        locations = list(self._locations.values())

        summarize = None
        if stats:
            summarize_kwargs = {}
            if locations:
                summarize_kwargs['start_datetime'] = locations[0].timeline[0]
                summarize_kwargs['end_datetime'] = locations[0].timeline[-1]
            summarize = self.summarize(**summarize_kwargs)

        total = len(self._locations)
        _limit = total if limit is None else min(limit, total)

        figure = make_subplots(rows=len(attributes), cols=1, shared_xaxes=True,
                               vertical_spacing=0.06)

        for row, band_name in enumerate(attributes, start=1):
            attr_def = attribute_map[band_name]

            if pixels and locations:
                # Every pixel series goes into a single trace, the runs separated
                # by None. One trace instead of thousands keeps panning and
                # zooming responsive, and the whole band toggles as one legend
                # entry -- which is what the matplotlib rendering does visually.
                xs, ys, labels = [], [], []
                for location in locations[:_limit]:
                    values = self._prepared_values(
                        attr_def, location.series['values'][band_name],
                        apply_scale, mask_nodata)
                    label = f'{location.x:.5f}, {location.y:.5f}'
                    xs.extend(x)
                    ys.extend(values)
                    labels.extend([label] * len(x))
                    xs.append(None)
                    ys.append(None)
                    labels.append(None)

                figure.add_trace(
                    go.Scattergl(
                        x=xs, y=ys, mode='lines',
                        name=f'pixels ({_limit})',
                        legendgroup='pixels', showlegend=row == 1,
                        line={'color': '#7F9BB1', 'width': 1},
                        opacity=0.2 if _limit > 100 else 0.6,
                        customdata=labels,
                        hovertemplate=f'%{{x|%d/%m/%Y}}<br>{band_name}: %{{y:.4f}}'
                                      '<br>pixel: %{customdata}<extra></extra>',
                    ), row=row, col=1)

            if stats:
                aggregations = summarize.values(band_name)
                for name, color, width, group, rotulo in (
                        ('q1', '#b19541', 1.5, 'quartis', 'quartis (q1, q3)'),
                        ('q3', '#b19541', 1.5, 'quartis', 'quartis (q1, q3)'),
                        ('median', '#B16240', 2.5, 'mediana', 'mediana')):
                    values = self._prepared_values(
                        attr_def, aggregations.values(name), apply_scale, mask_nodata)
                    # Scattergl too, and never Scatter: plotly draws WebGL traces
                    # on a canvas layer above the SVG one, so a Scatter here would
                    # end up *under* the pixel band no matter the trace order.
                    figure.add_trace(
                        go.Scattergl(
                            x=x, y=values[:len(x)], mode='lines', name=rotulo,
                            legendgroup=group,
                            # q1 and q3 share a single legend entry, as in plot().
                            showlegend=row == 1 and name != 'q3',
                            line={'color': color, 'width': width},
                            hovertemplate=f'%{{x|%d/%m/%Y}}<br>{rotulo}: '
                                          '%{y:.4f}<extra></extra>',
                        ), row=row, col=1)

            figure.update_yaxes(title_text=band_name, row=row, col=1)

        title = 'Time Series'
        if _limit < total:
            title += f' (Showing {_limit} of {total} points)'

        figure.update_layout(title={'text': title, 'x': 0.5, 'xanchor': 'center'},
                             height=320 * len(attributes) + 90,
                             hovermode='closest',
                             legend={'orientation': 'h', 'yanchor': 'bottom',
                                     'y': 1.02, 'xanchor': 'right', 'x': 1})

        return figure

    def _map_values(self, attribute, datetime, reduce, apply_scale, mask_nodata):
        """Collapse each location's series into the single value that colours it.

        Shared by :meth:`plot_map` and :meth:`_plot_map_interactive` so the static
        and the clickable map always paint the same numbers.

        Returns:
            tuple: ``(locations, values)`` in matching order.

        Raises:
            ValueError: If there are no locations, ``reduce`` is unknown, or
                ``datetime`` is not in the timeline.
            KeyError: If ``attribute`` is not present in the time series.
        """
        import numpy

        locations = list(self._locations.values())
        if not locations:
            raise ValueError('No locations to plot.')

        if attribute not in locations[0].series['values']:
            raise KeyError(
                f"Attribute '{attribute}' not found. Available: "
                f"{list(locations[0].series['values'])}"
            )

        reducers = {
            'mean': numpy.nanmean, 'median': numpy.nanmedian,
            'min': numpy.nanmin, 'max': numpy.nanmax,
            'std': numpy.nanstd,
        }
        if datetime is None and reduce not in reducers:
            raise ValueError(f"reduce must be one of {list(reducers)}, got {reduce!r}")

        values = []
        for location in locations:
            samples = self._values_for(location, attribute, apply_scale, mask_nodata)
            if datetime is None:
                value = float(reducers[reduce](numpy.array(samples, dtype='float64')))
            else:
                timeline = location.timeline
                if datetime not in timeline:
                    raise ValueError(f'{datetime!r} is not in the timeline.')
                value = samples[timeline.index(datetime)]
            values.append(value)

        return locations, values

    def plot_map(self, attribute: str, datetime: Optional[str] = None,
                 reduce: str = 'mean', apply_scale: bool = False,
                 mask_nodata: bool = False, ax=None, cmap: str = 'viridis',
                 markersize: int = 40, legend: bool = True,
                 basemap: bool = False, source=None, interactive: bool = False,
                 limit: Optional[int] = None, **kwargs):
        """Plot the queried locations on a map, colored by a band value.

        Complements :meth:`plot` (which shows *time*) by showing *space*: each
        queried location (a point, or a pixel center for multipoint/polygon
        queries) is drawn at its geographic position and colored by the value of
        ``attribute``. Builds a :class:`geopandas.GeoDataFrame` in ``EPSG:4326``
        and delegates to its ``.plot``.

        Args:
            attribute (str): The band whose value colors each location.
            datetime (str, optional): A specific date from the timeline. When
                ``None`` (default), the value is reduced over the whole timeline
                using ``reduce``.
            reduce (str): How to collapse the time axis when ``datetime`` is
                ``None``: one of ``'mean'``, ``'median'``, ``'min'``, ``'max'``,
                ``'std'``. Use ``'std'`` to map how much each pixel varied over
                the period, which highlights the heterogeneity within the area
                instead of its overall level.
            apply_scale (bool): Apply the band scale/offset client-side (Ciclo iv).
            mask_nodata (bool): Replace nodata samples with ``NaN`` before
                reducing, so the sentinel does not skew the color (Ciclo iv).
            ax (matplotlib.axes.Axes, optional): Existing axes to draw on.
            cmap (str): Matplotlib colormap name. Defaults to ``'viridis'``.
            markersize (int): Point size. Defaults to 40.
            legend (bool): Draw the colorbar. Defaults to True.
            basemap (bool): Add an OpenStreetMap basemap via ``contextily``
                (needs network and the optional ``contextily`` package).
                Defaults to False so the map stays offline-friendly.
            source: Tile provider for the basemap (e.g.
                ``contextily.providers.OpenStreetMap.Mapnik``). When ``None``
                (default) and ``basemap`` is True, OpenStreetMap Mapnik is used.
            interactive (bool): Render a pannable, zoomable map with folium
                instead of a static matplotlib figure. Default is False. Each
                marker carries the series of its own pixel: **clicking a dot
                opens that pixel's chart**, so the map answers "what happened
                here over time", not only "what is the reduced value here".
                Returns the :class:`folium.Map` rather than drawing it, and
                requires ``folium`` (``pip install wtss[interactive]``).
                ``markersize`` is not forwarded: matplotlib sizes markers by
                area in points and folium by radius in pixels, so the
                interactive map keeps its own default and is resized with
                ``radius=`` instead.
            limit (int, optional): Draw at most this many locations, for the
                interactive map only. One popup per pixel gets heavy on large
                areas; the legend then announces how many of how many are shown.
            **kwargs: Forwarded to :meth:`geopandas.GeoDataFrame.plot`, or to
                :class:`folium.Map` when ``interactive`` is True.

        Returns:
            matplotlib.axes.Axes: The axes with the map.

        Raises:
            ValueError: If there are no locations, ``reduce`` is unknown, or
                ``datetime`` is not in the timeline.
            KeyError: If ``attribute`` is not present in the time series.
            ImportError: If geopandas/matplotlib (or contextily) are missing.
        """
        if interactive:
            return self._plot_map_interactive(
                attribute, datetime=datetime, reduce=reduce,
                apply_scale=apply_scale, mask_nodata=mask_nodata, cmap=cmap,
                legend=legend, limit=limit, **kwargs)

        try:
            import geopandas
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('You should install geopandas and matplotlib!')

        locations, values = self._map_values(attribute, datetime, reduce,
                                             apply_scale, mask_nodata)
        geoms = [location.geom for location in locations]

        gdf = geopandas.GeoDataFrame({attribute: values}, geometry=geoms, crs='EPSG:4326')

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))

        gdf.plot(column=attribute, ax=ax, cmap=cmap, markersize=markersize,
                 legend=legend, **kwargs)

        if basemap:
            try:
                import contextily
            except ImportError:
                raise ImportError('Install contextily for basemaps: pip install contextily')
            if source is None:
                source = contextily.providers.OpenStreetMap.Mapnik
            # Suppress contextily's built-in credit (bottom-left, on top of the
            # data) and re-add it just below the axes so it never covers points.
            contextily.add_basemap(ax, crs=gdf.crs, source=source, attribution=False)
            attribution = getattr(source, 'attribution', '') or ''
            if attribution:
                # Draw the credit vertically just outside the right edge of the
                # map, in the gap before the colorbar, so it never covers data.
                ax.annotate(attribution, xy=(1.02, 0.5), xycoords='axes fraction',
                            rotation=90, rotation_mode='anchor',
                            ha='left', va='center', fontsize=6, color='gray',
                            annotation_clip=False)

        when = datetime if datetime is not None else f'{reduce} over timeline'
        ax.set_title(f'{self._coverage.name} — {attribute} ({when})')
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')

        return ax

    def _series_svg(self, location, attribute, apply_scale, mask_nodata,
                    width=300, height=120, pad=26):
        """Draw one location's series as a standalone SVG, for a map popup.

        The chart is plain markup with no script and no external asset, so it
        survives inside a Leaflet popup and in an exported notebook alike.
        """
        import html
        import numpy

        samples = numpy.array(
            self._values_for(location, attribute, apply_scale, mask_nodata),
            dtype='float64')
        timeline = [str(moment)[:10] for moment in self.timeline]
        known = numpy.isfinite(samples)

        if not known.any():
            return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                    f'height="40"><text x="6" y="24" font-size="11" '
                    f'font-family="sans-serif">sem dado em {html.escape(attribute)}'
                    '</text></svg>')

        low = float(numpy.nanmin(samples))
        high = float(numpy.nanmax(samples))
        span = (high - low) or 1.0
        xs = numpy.linspace(pad, width - 6, len(samples))
        ys = height - pad - (samples - low) / span * (height - 2 * pad)
        points = ' '.join(f'{x:.1f},{y:.1f}'
                          for x, y, ok in zip(xs, ys, known) if ok)

        title = html.escape(f'{attribute}  ({location.x:.5f}, {location.y:.5f})')
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<text x="4" y="12" font-size="10" font-family="sans-serif">{title}</text>'
            f'<line x1="{pad}" y1="{height - pad}" x2="{width - 6}" y2="{height - pad}" '
            'stroke="#999" stroke-width="1"/>'
            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" '
            'stroke="#999" stroke-width="1"/>'
            f'<polyline fill="none" stroke="#2a7f3f" stroke-width="1.6" points="{points}"/>'
            f'<text x="2" y="{pad + 4}" font-size="8" font-family="sans-serif">{high:.2f}</text>'
            f'<text x="2" y="{height - pad}" font-size="8" font-family="sans-serif">{low:.2f}</text>'
            f'<text x="{pad}" y="{height - 8}" font-size="8" '
            f'font-family="sans-serif">{timeline[0]}</text>'
            f'<text x="{width - 62}" y="{height - 8}" font-size="8" '
            f'font-family="sans-serif">{timeline[-1]}</text>'
            '</svg>')

    def _plot_map_interactive(self, attribute, datetime=None, reduce='mean',
                              apply_scale=False, mask_nodata=False, cmap='viridis',
                              markersize=6, legend=True, limit=None,
                              tiles='OpenStreetMap', **kwargs):
        """Render the locations on a slippy map (see ``interactive`` in :meth:`plot_map`).

        Returns:
            folium.Map: One circle marker per location, coloured like
            :meth:`plot_map` and carrying its own series in the popup.

        Raises:
            ImportError: If folium could not be imported.
            ValueError: If ``limit`` is negative, or as raised by :meth:`_map_values`.
        """
        try:
            import folium
            import numpy
            from matplotlib import colors, pyplot
        except ImportError:
            raise ImportError('Install folium for interactive maps: '
                              'pip install wtss[interactive]')

        if limit is not None and limit < 0:
            raise ValueError('Limit cannot be negative')

        # The marker radius is in pixels here, unlike the point *area* matplotlib
        # takes, so it gets its own name and never reaches the folium.Map kwargs.
        markersize = kwargs.pop('radius', markersize)

        locations, values = self._map_values(attribute, datetime, reduce,
                                             apply_scale, mask_nodata)
        total = len(locations)
        drawn = total if limit is None else min(limit, total)

        finite = [value for value in values if value == value]
        norm = colors.Normalize(vmin=min(finite) if finite else 0.0,
                                vmax=max(finite) if finite else 1.0)
        colormap = pyplot.get_cmap(cmap)

        centre = [float(numpy.mean([loc.y for loc in locations])),
                  float(numpy.mean([loc.x for loc in locations]))]
        figure = folium.Map(location=centre, tiles=tiles, **kwargs)

        for location, value in list(zip(locations, values))[:drawn]:
            colour = ('#808080' if value != value
                      else colors.to_hex(colormap(norm(value))))
            folium.CircleMarker(
                location=[location.y, location.x],
                radius=markersize,
                color=None, fill=True, fill_opacity=0.85, fill_color=colour,
                # The popup is what the reviewer asked for: click a dot, get the
                # series of that very pixel instead of only its reduced value.
                popup=folium.Popup(self._series_svg(location, attribute,
                                                    apply_scale, mask_nodata),
                                   max_width=340),
                tooltip=f'{attribute} {reduce}: {value:.4f}',
            ).add_to(figure)

        # Frame the data rather than trusting a fixed zoom: areas range from a few
        # pixels to a whole municipality.
        lats = [loc.y for loc in locations[:drawn]]
        lons = [loc.x for loc in locations[:drawn]]
        if lats and lons:
            figure.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

        if legend:
            rotulo = f'{attribute} ({reduce})' if datetime is None else f'{attribute} {datetime}'
            if drawn < total:
                rotulo += f' — {drawn} de {total} pixels'
            figure.get_root().html.add_child(folium.Element(
                '<div style="position:fixed;top:10px;left:60px;z-index:9999;'
                'background:white;padding:4px 8px;border:1px solid #999;'
                'border-radius:3px;font:12px sans-serif">'
                f'{rotulo} — clique num pixel para ver a sua série</div>'))

        return figure

    def plot_cube(self, attribute: str, apply_scale: bool = False,
                  mask_nodata: bool = False, ax=None, cmap: str = 'viridis',
                  title: Optional[str] = None, **kwargs):
        """Render the area as a 3D **spatiotemporal data cube** ``(time, y, x)``.

        Extends :meth:`plot_map` (a single date, in 2D) into the time axis: each
        timestamp becomes one horizontal raster layer of ``attribute``, stacked
        along the vertical (time) axis to form the cube that :meth:`to_xarray`
        (with ``grid=True``) produces. Meant for polygon/area queries.

        Args:
            attribute (str): The band that colours each cell.
            apply_scale (bool): Apply the band scale/offset client-side (Ciclo iv).
            mask_nodata (bool): Replace nodata samples with ``NaN`` (Ciclo iv).
            ax (mpl_toolkits.mplot3d.axes3d.Axes3D, optional): Existing 3D axes to
                draw on. When ``None`` (default) a new 3D figure is created.
            cmap (str): Matplotlib colormap name. Defaults to ``'viridis'``.
            title (str, optional): Figure title. Defaults to a generated one.
            **kwargs: Forwarded to ``Axes3D.plot_surface``.

        Returns:
            mpl_toolkits.mplot3d.axes3d.Axes3D: The axes with the cube.

        Raises:
            ValueError: If there are no locations or the area is smaller than 2x2.
            KeyError: If ``attribute`` is not present in the time series.
            ImportError: If matplotlib/numpy are missing.
        """
        try:
            import numpy
            import matplotlib.pyplot as plt
            from matplotlib import cm, colors
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d')
        except ImportError:
            raise ImportError('You should install matplotlib and numpy!')

        locations = list(self._locations.values())
        if not locations:
            raise ValueError('No locations to plot.')
        if attribute not in locations[0].series['values']:
            raise KeyError(
                f"Attribute '{attribute}' not found. Available: "
                f"{list(locations[0].series['values'])}"
            )

        lon2d, _lat2d, cube = self._grid_cube(attribute, apply_scale, mask_nodata)

        # Crop to the rectangle every row shares, so each layer renders as a plain
        # matrix. A lon/lat box cuts the reprojected grid along a diagonal, so the
        # rows genuinely cover different column ranges and the full extent is a
        # parallelogram. Drawn as-is, its staircase edge reads as structure in the
        # data rather than as the shape of the query, so the stack keeps only the
        # columns present in every row -- and then the rows complete within them.
        # Coverage comes from the samples, not from the coordinates: those are
        # filled for every cell of the grid, including the ones with no pixel.
        covered = numpy.isfinite(cube).any(axis=0)
        full_cols = numpy.flatnonzero(covered.all(axis=0))
        if full_cols.size:
            cube = cube[:, :, full_cols[0]:full_cols[-1] + 1]
            covered = covered[:, full_cols[0]:full_cols[-1] + 1]
            full_rows = numpy.flatnonzero(covered.all(axis=1))
            if full_rows.size:
                cube = cube[:, full_rows[0]:full_rows[-1] + 1, :]

        ny, nx = cube.shape[1], cube.shape[2]
        if nx < 2 or ny < 2:
            raise ValueError('plot_cube needs an area (a grid of at least 2x2 pixels).')

        grid_x, grid_y = numpy.meshgrid(numpy.arange(nx), numpy.arange(ny))
        finite = cube[numpy.isfinite(cube)]
        vmin = float(finite.min()) if finite.size else 0.0
        vmax = float(finite.max()) if finite.size else 1.0
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        colormap = plt.get_cmap(cmap)

        created = ax is None
        if created:
            fig = plt.figure(figsize=(9, 8))
            ax = fig.add_subplot(projection='3d')
        else:
            fig = ax.figure

        # One raster layer per timestamp, stacked along z = time index.
        for t in range(cube.shape[0]):
            z = numpy.full_like(grid_x, t, dtype='float64')
            ax.plot_surface(grid_x, grid_y, z, facecolors=colormap(norm(cube[t])),
                            rstride=1, cstride=1, shade=False, **kwargs)

        if created:
            ax.view_init(elev=22, azim=-60)

        # Axis titles: pull 'longitude'/'latitude' close to their numbers and push
        # 'time' clear of the (wider) date labels, aiming for a similar gap on all
        # three. The z tick pad also runs higher because the dates are long.
        ax.set_xlabel('x (longitude)', labelpad=-2)
        ax.set_ylabel('y (latitude)', labelpad=-2)
        ax.set_zlabel('time', labelpad=23)
        ax.set_zticks(range(cube.shape[0]))
        ax.set_zticklabels([str(t)[:10] for t in self.timeline], fontsize=8)
        ax.tick_params(axis='z', pad=12)
        ax.tick_params(axis='x', pad=-3)
        ax.tick_params(axis='y', pad=-3)
        ax.set_title(title or f'{self._coverage.name} — {attribute} (data cube)')

        mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
        mappable.set_array([])
        # Extra pad keeps the colorbar off the z (time) axis label.
        fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.16, label=attribute)

        return ax

    def _repr_pretty_(self, p, cycle):
        """Customize how the REPL pretty-prints a time series."""
        return self._repr_html_()

    def _repr_html_(self):
        """Display the time series as a HTML.

        This integrates a rich display in IPython.
        """
        html = render_html('timeseries.html', timeseries=self)

        return html
