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

from grade_sintetica import (MODIS_ORIGIN, MODIS_RES,  # noqa: E402
                             SINUSOIDAL_MODIS, grid_block)
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
    'bands': [{'name': 'NDVI', 'nodata': -3000, 'data_type': 'int16',
               'resolution_x': MODIS_RES, 'resolution_y': MODIS_RES}],
    'bdc:crs': SINUSOIDAL_MODIS,
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
        'pixel_size': [MODIS_RES, -MODIS_RES],
        'time_series': {'timeline': list(TIMELINE), 'values': {'NDVI': list(values)}},
    }


def _block(nrows, ncols):
    """``(row, col) -> (lon, lat)`` of a block of real MODIS sinusoidal cells."""
    return {(row, col): (lon, lat) for row, col, lon, lat
            in grid_block(SINUSOIDAL_MODIS, MODIS_RES, MODIS_ORIGIN, -54.0, -12.0,
                          nrows, ncols)}


#: The centres of a 2x2 block of the MODIS grid, by native (row, col); row 0 is north.
CELLS = _block(2, 2)

#: Four pixels of the 2x2 grid: south-west, south-east, north-west, north-east. The
#: last one holds a nodata sample (-3000) so masking can be exercised too.
PIXELS = [
    _pixel(*CELLS[1, 0], [1000, 2000, 3000]),
    _pixel(*CELLS[1, 1], [1100, 2100, 3100]),
    _pixel(*CELLS[0, 0], [1200, 2200, 3200]),
    _pixel(*CELLS[0, 1], [1300, -3000, 3300]),
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
        expected = [p['pixel_center']['coordinates'][0] for p in PIXELS]
        assert [x for x, _ in offsets] == pytest.approx(expected)

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
        # Geographic coordinates come as 2D arrays: the sinusoidal grid is sheared
        # in lon/lat, so each row has its own longitudes.
        assert ds['longitude'].dims == ('y', 'x')
        assert ds['latitude'].dims == ('y', 'x')
        for (row, col), (lon, lat) in CELLS.items():
            assert ds['longitude'].values[row, col] == pytest.approx(lon)
            assert ds['latitude'].values[row, col] == pytest.approx(lat)
        # Each pixel in its own cell: north-west 1200, north-east 1300, south-west
        # 1000, south-east 1100.
        assert ds['NDVI'].isel(time=0).values.tolist() == [[1200, 1300], [1000, 1100]]

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


class TestGridReconstruction:
    """``_grid_cube`` must rebuild the native raster out of reprojected pixel centres."""

    def _timeseries(self, service, centres):
        """Build an area :class:`TimeSeries` from ``(lon, lat)`` pixel centres."""
        from wtss.timeseries import TimeSeries

        pixels = [_pixel(lon, lat, [1000 + i, 2000 + i, 3000 + i])
                  for i, (lon, lat) in enumerate(centres)]
        return TimeSeries(service['MOD13Q1-6'],
                          {'results': pixels, 'query': dict(_QUERY)})

    def test_float_noise_does_not_split_rows(self, service):
        """Reprojected centres carry float noise; it must not create new rows.

        Grouping by exact equality once turned a 3x3 area into 9 rows of one pixel.
        """
        block = _block(3, 3)
        centres = [(lon + i * 1e-9, lat + i * 1e-9)
                   for i, (lon, lat) in enumerate(block.values())]
        ts = self._timeseries(service, centres)

        ds = ts.to_xarray(grid=True)

        assert ds.sizes == {'time': 3, 'y': 3, 'x': 3}

    def test_short_rows_keep_their_columns_aligned(self, service):
        """A row missing its westernmost pixel must not slide a whole row left.

        Packing every row from column zero put pixels of different longitudes in
        the same column, misregistering the raster by one cell.
        """
        import numpy as np

        block = _block(3, 3)
        centres = [centre for cell, centre in block.items() if cell != (1, 0)]
        ts = self._timeseries(service, centres)

        lon2d, lat2d, cube = ts._grid_cube('NDVI')

        assert lon2d.shape == (3, 3)
        # The hole is the missing cell, and only it.
        holes = np.argwhere(np.isnan(cube).all(axis=0)).tolist()
        assert holes == [[1, 0]]
        # Its position is still known: the grid defines it, only the value is missing.
        assert (lon2d[1, 0], lat2d[1, 0]) == pytest.approx(block[1, 0])
        assert np.isfinite(lon2d).all()

    def test_plot_cube_crops_to_the_shared_rectangle(self, service, monkeypatch):
        """The cube draws only the columns every row shares.

        A lon/lat box cuts the native grid along a diagonal, so the full extent is
        a parallelogram; drawn whole, its staircase edge reads as structure in the
        data instead of the shape of the query.
        """
        from mpl_toolkits.mplot3d import Axes3D

        block = _block(3, 5)
        cells = ([(0, c) for c in range(4)] + [(1, c) for c in range(4)]
                 + [(2, c) for c in range(1, 5)])
        ts = self._timeseries(service, [block[cell] for cell in cells])

        # A união das linhas tem 5 colunas, com buracos nas pontas.
        lon2d, _lat2d, _cube = ts._grid_cube('NDVI')
        assert lon2d.shape == (3, 5)

        desenhadas = []
        original = Axes3D.plot_surface

        def espiao(self, grid_x, grid_y, z, **kwargs):
            desenhadas.append(grid_x.shape)
            return original(self, grid_x, grid_y, z, **kwargs)

        monkeypatch.setattr(Axes3D, 'plot_surface', espiao)
        ts.plot_cube('NDVI')

        # Uma camada por data, todas no retângulo de 3 colunas comuns.
        assert desenhadas == [(3, 3)] * len(TIMELINE)
