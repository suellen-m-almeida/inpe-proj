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

    def to_xarray(self, apply_scale: bool = False, mask_nodata: bool = False):
        """Return the time series as a labelled :class:`xarray.Dataset` (Ciclo v).

        Each attribute becomes a data variable. Dimensions are:

        - **Single location**: ``(time,)``; ``longitude``/``latitude`` are scalar
          coordinates.
        - **Multiple locations**: ``(time, location)``; ``longitude``/``latitude``
          are coordinates along ``location``.

        The polygon case (rebuilding a 2D ``(time, y, x)`` grid from sparse
        pixels) is not handled here yet.

        Args:
            apply_scale (bool): Apply the band scale/offset client-side (Ciclo iv).
            mask_nodata (bool): Replace nodata samples with ``NaN`` (Ciclo iv).

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

    def plot(self, stats: bool = True, limit: Optional[int] = None,
             pixels: bool = True, apply_scale: bool = False,
             mask_nodata: bool = False, **options):
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
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('You should install Matplotlib and Numpy!')

        if limit is not None and limit < 0:
            raise ValueError('Limit cannot be negative')

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
            nodata = attr_def['nodata']

            # Turn a raw sample list into plot-ready values, honoring the
            # apply_scale/mask_nodata flags. Without them, keep the legacy
            # behavior (mask nodata to None so it draws as a gap, raw scale).
            # When scaling, always mask nodata so the sentinel is never scaled.
            def _prep(raw):
                if apply_scale or mask_nodata:
                    return self._coverage._apply_metadata(
                        list(raw), attr_def, apply_scale, mask_nodata or apply_scale)
                return [value if value != nodata else None for value in raw]

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

    def plot_map(self, attribute: str, datetime: Optional[str] = None,
                 reduce: str = 'mean', apply_scale: bool = False,
                 mask_nodata: bool = False, ax=None, cmap: str = 'viridis',
                 markersize: int = 40, legend: bool = True,
                 basemap: bool = False, source=None, **kwargs):
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
                ``None``: one of ``'mean'``, ``'median'``, ``'min'``, ``'max'``.
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
            **kwargs: Forwarded to :meth:`geopandas.GeoDataFrame.plot`.

        Returns:
            matplotlib.axes.Axes: The axes with the map.

        Raises:
            ValueError: If there are no locations, ``reduce`` is unknown, or
                ``datetime`` is not in the timeline.
            KeyError: If ``attribute`` is not present in the time series.
            ImportError: If geopandas/matplotlib (or contextily) are missing.
        """
        try:
            import geopandas
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('You should install geopandas and matplotlib!')

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
        }
        if datetime is None and reduce not in reducers:
            raise ValueError(f"reduce must be one of {list(reducers)}, got {reduce!r}")

        geoms, values = [], []
        for location in locations:
            samples = self._values_for(location, attribute, apply_scale, mask_nodata)
            if datetime is None:
                value = float(reducers[reduce](numpy.array(samples, dtype='float64')))
            else:
                timeline = location.timeline
                if datetime not in timeline:
                    raise ValueError(f'{datetime!r} is not in the timeline.')
                value = samples[timeline.index(datetime)]
            geoms.append(location.geom)
            values.append(value)

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

    def _repr_pretty_(self, p, cycle):
        """Customize how the REPL pretty-prints a time series."""
        return self._repr_html_()

    def _repr_html_(self):
        """Display the time series as a HTML.

        This integrates a rich display in IPython.
        """
        html = render_html('timeseries.html', timeseries=self)

        return html
