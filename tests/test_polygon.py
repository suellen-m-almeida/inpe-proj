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

"""Offline tests for **polygon (area) queries** end to end.

Unlike ``test_pagination.py`` (which paginates a *single* pixel) and ``test_plot.py``
(which hand-builds a multi-location :class:`~wtss.timeseries.TimeSeries`), here a real
polygon query is driven through the public ``coverage.ts(geom=<Polygon>)`` API against a
mocked server that answers with a **single page holding several pixels**. We then check
that the whole area path — materialisation, the spatial ``plot_map``, ``to_xarray`` and
``summarize`` — works without a live server.

A second class pins down the server-side rejection we observed against the live v4 service:
an oversized/unsupported polygon makes the server answer ``400 Bad Request``. The client is
not at fault (it sends the same well-formed request that succeeds for a small box), so the
contract is simply that the ``HTTPError`` surfaces to the caller.
"""

import matplotlib

matplotlib.use('Agg')  # headless: render in memory, never open a window
import matplotlib.pyplot as plt  # noqa: E402

import pytest  # noqa: E402
import requests  # noqa: E402
import responses  # noqa: E402

from wtss import WTSS  # noqa: E402

MOCK_URL = 'http://wtss.test'

TIMELINE = ['2017-01-01', '2017-01-17', '2017-02-02']

ROOT_RESPONSE = {
    'wtss_version': '2.0',
    'links': [
        {'rel': 'self', 'title': 'WTSS', 'href': f'{MOCK_URL}/'},
        {'rel': 'data', 'title': 'Coverage MOD13Q1-6', 'href': f'{MOCK_URL}/MOD13Q1-6'},
    ],
}

COVERAGE_RESPONSE = {
    'fullname': 'MOD13Q1-6',
    'description': 'MODIS Vegetation Indices 16-day 250m (MOD13Q1) v6.',
    'bands': [{'name': 'NDVI', 'nodata': -3000, 'data_type': 'int16'}],
    'bdc:crs': '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181',
    'raster_size': {'xsize': 172800, 'ysize': 86400},
    'extent': {
        'type': 'Polygon',
        'coordinates': [[[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]]],
    },
    'timeline': TIMELINE,
}

#: A small box (a closed Polygon) covering a 2x2 grid of pixels.
POLYGON_GEOM = {
    'type': 'Polygon',
    'coordinates': [[[-54, -12], [-53.99, -12], [-53.99, -11.99], [-54, -11.99], [-54, -12]]],
}

_QUERY = {
    'attributes': ['NDVI'],
    'start_datetime': '2017-01-01',
    'end_datetime': '2017-02-28',
    'geom': POLYGON_GEOM,
}


def _pixel(lon, lat, values):
    """One pixel result: its centre and NDVI series over the shared timeline."""
    return {
        'pixel_center': {'type': 'Point', 'coordinates': [lon, lat]},
        'pixel_size': [231.65, 231.65],
        'time_series': {'timeline': list(TIMELINE), 'values': {'NDVI': list(values)}},
    }


#: Four pixels of the 2x2 grid, in increasing (lon, lat). The last one holds a nodata
#: sample (-3000) so masking can be exercised through the area path too.
PIXELS = [
    _pixel(-54.00, -12.00, [1000, 2000, 3000]),
    _pixel(-53.99, -12.00, [1100, 2100, 3100]),
    _pixel(-54.00, -11.99, [1200, 2200, 3200]),
    _pixel(-53.99, -11.99, [1300, -3000, 3300]),
]

#: A single response carrying every pixel. WTSS pagination splits the *timeline*
#: (not the pixels), so all pixels already arrive here with the full timeline;
#: omitting the ``pagination`` block signals "single page" and the client stops.
POLYGON_PAGE = {
    'results': PIXELS,
    'query': dict(_QUERY),
}

#: Summarize payload for the same area, with every aggregation the plots consume.
SUMMARIZE_RESPONSE = {
    'query': {'attributes': ['NDVI'], 'geom': POLYGON_GEOM,
              'aggregations': ['q1', 'q3', 'median', 'mean', 'std']},
    'results': {
        'timeline': TIMELINE,
        'values': {
            'NDVI': {
                'q1': [1050, 2050, 3050], 'q3': [1250, 2250, 3250],
                'median': [1150, 2150, 3150], 'mean': [1150, 2150, 3150],
                'std': [110, 120, 130],
            }
        },
    },
}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


@pytest.fixture
def service():
    """WTSS client whose area endpoints (timeseries + summarize) are mocked."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        rsps.add(responses.GET, f'{MOCK_URL}/', json=ROOT_RESPONSE, status=200)
        rsps.add(responses.GET, f'{MOCK_URL}/MOD13Q1-6', json=COVERAGE_RESPONSE, status=200)
        rsps.add(responses.POST, f'{MOCK_URL}/MOD13Q1-6/timeseries',
                 json=POLYGON_PAGE, status=200)
        rsps.add(responses.POST, f'{MOCK_URL}/MOD13Q1-6/summarize',
                 json=SUMMARIZE_RESPONSE, status=200)
        yield WTSS(MOCK_URL, access_token='fake-token')


def _area_timeseries(service):
    return service['MOD13Q1-6'].ts(**_QUERY).ts


class TestPolygonQuery:
    """A polygon query must materialise every pixel and feed the area views."""

    def test_polygon_materialises_all_pixels(self, service):
        ts = _area_timeseries(service)
        # One NDVI series per pixel of the 2x2 grid.
        assert len(ts.values('NDVI')) == 4
        assert ts.timeline == TIMELINE

    def test_polygon_plot_map_draws_every_pixel(self, service):
        ts = _area_timeseries(service)
        ax = ts.plot_map('NDVI', reduce='mean', legend=False)

        offsets = ax.collections[0].get_offsets()
        assert offsets.shape[0] == 4  # four pixel centres drawn
        assert [round(x, 2) for x, _ in offsets] == [-54.0, -53.99, -54.0, -53.99]

    def test_polygon_to_xarray_has_location_dim(self, service):
        ts = _area_timeseries(service)
        ds = ts.to_xarray()
        assert ds['NDVI'].dims == ('time', 'location')
        assert ds.sizes == {'time': 3, 'location': 4}

    def test_polygon_to_xarray_grid_rebuilds_raster(self, service):
        """Ciclo v / semana 9: grid=True rebuilds the (time, y, x) cube."""
        ts = _area_timeseries(service)
        ds = ts.to_xarray(grid=True)

        assert ds['NDVI'].dims == ('time', 'y', 'x')
        assert ds.sizes == {'time': 3, 'y': 2, 'x': 2}
        # Geographic coordinates come as 2D arrays (curvilinear once reprojected).
        assert ds['longitude'].dims == ('y', 'x')
        assert ds['latitude'].dims == ('y', 'x')
        assert sorted(set(ds['longitude'].values.ravel().tolist())) == [-54.0, -53.99]
        assert sorted(set(ds['latitude'].values.ravel().tolist())) == [-12.0, -11.99]
        # Every pixel's series is present (order-independent).
        assert sorted(ds['NDVI'].isel(time=0).values.ravel().tolist()) == [1000, 1100, 1200, 1300]

    def test_grid_leaves_missing_pixels_as_nan(self, service):
        """A pixel absent from the response leaves its grid cell as NaN."""
        import numpy as np
        from wtss.timeseries import TimeSeries

        coverage = service['MOD13Q1-6']
        # Only 3 of the 2x2 grid cells; one pixel is missing.
        partial = TimeSeries(coverage, {'results': PIXELS[:3], 'query': dict(_QUERY)})
        ds = partial.to_xarray(grid=True)

        assert ds.sizes == {'time': 3, 'y': 2, 'x': 2}
        # Exactly one (y, x) cell is fully NaN across time (the missing pixel).
        fully_nan = np.isnan(ds['NDVI'].values).all(axis=0)
        assert int(fully_nan.sum()) == 1

    def test_polygon_plot_stats_runs_headless(self, service):
        """plot(pixels=False) draws median + quartiles from the summarize endpoint."""
        ts = _area_timeseries(service)
        ts.plot(attributes=['NDVI'], pixels=False)

        labels = [line.get_label() for line in plt.gcf().axes[0].lines]
        assert 'mediana' in labels
        assert 'quartis (q1, q3)' in labels

    def test_polygon_summarize_mean_std_runs_headless(self, service):
        resumo = service['MOD13Q1-6'].summarize(**_QUERY)
        resumo.plot_mean_std(attribute='NDVI')

        labels = [line.get_label() for line in plt.gcf().axes[0].lines]
        assert 'mean' in labels

    def test_polygon_plot_cube_stacks_one_layer_per_date(self, service):
        """plot_cube renders a 3D cube with one raster layer per timestamp."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        ts = _area_timeseries(service)
        ax = ts.plot_cube('NDVI')

        assert ax.name == '3d'
        surfaces = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(surfaces) == 3  # one surface per date in TIMELINE

    def test_plot_cube_unknown_attribute_raises(self, service):
        ts = _area_timeseries(service)
        with pytest.raises(KeyError, match='EVI'):
            ts.plot_cube('EVI')

    def test_plot_cube_needs_a_2d_area(self, service):
        """A single pixel is not an area: plot_cube must refuse it clearly."""
        from wtss.timeseries import TimeSeries

        coverage = service['MOD13Q1-6']
        one = TimeSeries(coverage, {'results': PIXELS[:1], 'query': dict(_QUERY)})
        with pytest.raises(ValueError, match='at least 2x2'):
            one.plot_cube('NDVI')

    def test_polygon_masks_nodata_on_map(self, service):
        """The -3000 sample in the last pixel must not skew its mean on the map."""
        ts = _area_timeseries(service)
        ax = ts.plot_map('NDVI', reduce='mean', mask_nodata=True, legend=False)

        # Last pixel: nanmean(1300, nan, 3300) == 2300, not (1300-3000+3300)/3.
        assert list(ax.collections[0].get_array())[-1] == 2300.0


class TestPolygonServerError:
    """An oversized/unsupported polygon is rejected by the server with 400."""

    def test_polygon_400_raises_httperror(self):
        with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
            rsps.add(responses.GET, f'{MOCK_URL}/', json=ROOT_RESPONSE, status=200)
            rsps.add(responses.GET, f'{MOCK_URL}/MOD13Q1-6', json=COVERAGE_RESPONSE, status=200)
            rsps.add(responses.POST, f'{MOCK_URL}/MOD13Q1-6/timeseries',
                     json={'code': 400, 'description': 'Invalid geometry'}, status=400)
            service = WTSS(MOCK_URL, access_token='fake-token')

            # The client sends a well-formed request; the 400 comes from the server
            # and must surface to the caller when the deferred query materialises.
            with pytest.raises(requests.exceptions.HTTPError):
                service['MOD13Q1-6'].ts(**_QUERY).ts
