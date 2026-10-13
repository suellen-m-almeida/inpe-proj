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

"""Offline tests for the native-grid raster, the GeoTIFF, server errors and MultiPolygon.

The grid tests lay synthetic pixels on the real BDC Albers and MODIS sinusoidal
grids (see ``grade_sintetica``) around Nova Mutum (56 W), where the Albers grid is
rotated about 0.4 degrees in lon/lat and the sinusoidal one is sheared. The old
latitude-based reconstruction transposed the 440 m box of case study 1 and
collapsed a 2 km box into a single row; every test here would have failed on it.
"""

import json
import math

import numpy
import pytest
import requests
import responses
import shapely.geometry
from grade_sintetica import (ALBERS_BDC, MODIS_ORIGIN, MODIS_RES,
                             SINUSOIDAL_MODIS, box_around, coverage_metadata,
                             native_cells_in_box, pixel)
from pyproj import Transformer

from wtss import WTSS
from wtss.timeseries import TimeSeries

MOCK_URL = 'http://wtss.test'
TIMELINE = ['2025-04-07', '2025-04-23']
LON, LAT = -56.00, -13.55

#: (name, CRS, pixel size, grid origin, box side in degrees). The Albers boxes are
#: the 440 m box of case study 1 (transposed before the fix) and a 2 km box
#: (collapsed into one row before the fix); the sinusoidal one is a 5 km box.
GRIDS = {
    'albers-caso1': ('S2-16D-2', ALBERS_BDC, 10.0, (0.0, 0.0), 0.004),
    'albers-2km': ('S2-16D-2', ALBERS_BDC, 10.0, (0.0, 0.0), 0.018),
    'sinusoidal-5km': ('mod13q1-6.1', SINUSOIDAL_MODIS, MODIS_RES, MODIS_ORIGIN, 0.045),
}


def _service(name, crs, res, extra=None):
    """A mocked WTSS client exposing one synthetic coverage."""
    root = {'wtss_version': '2.0',
            'links': [{'rel': 'data', 'title': f'Coverage {name}',
                       'href': f'{MOCK_URL}/{name}'}]}
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        rsps.add(responses.GET, f'{MOCK_URL}/', json=root, status=200)
        rsps.add(responses.GET, f'{MOCK_URL}/{name}',
                 json=coverage_metadata(name, crs, res, TIMELINE), status=200)
        service = WTSS(MOCK_URL)
        coverage = service[name]
    return service, coverage


def _area(key):
    """A TimeSeries over a synthetic box, and the native ``(row, col)`` of each pixel.

    Every pixel carries its own cell in its values (``row * 1000 + col`` on the
    first date), so a misplaced pixel shows up as a wrong number in the wrong cell.
    """
    name, crs, res, origin, side = GRIDS[key]
    cells = native_cells_in_box(crs, res, origin, box_around(LON, LAT, side))
    _service_, coverage = _service(name, crs, res)
    results = [pixel(lon, lat, TIMELINE, [row * 1000 + col, -(row * 1000 + col)], res)
               for row, col, lon, lat in cells]
    ts = TimeSeries(coverage, {'results': results,
                               'query': {'attributes': ['NDVI'], 'geom': {'type': 'Polygon'}}})
    return ts, cells, crs, res


@pytest.mark.parametrize('key', list(GRIDS))
class TestNativeGrid:
    """The raster is rebuilt on the native grid, whatever it looks like in lon/lat."""

    def test_shape_is_the_native_extent(self, key):
        ts, cells, _crs, _res = _area(key)
        ds = ts.to_xarray(grid=True)

        nrows = max(row for row, *_ in cells) + 1
        ncols = max(col for _row, col, *_ in cells) + 1
        assert ds.sizes == {'time': len(TIMELINE), 'y': nrows, 'x': ncols}

    def test_every_pixel_once_in_its_own_cell(self, key):
        ts, cells, _crs, _res = _area(key)
        cube = ts.to_xarray(grid=True)['NDVI'].values

        # No pixel lost or duplicated: one finite cell per returned pixel ...
        assert int(numpy.isfinite(cube[0]).sum()) == len(cells)
        # ... and each pixel's value sits exactly at its native (row, col).
        for row, col, _lon, _lat in cells:
            assert cube[0, row, col] == row * 1000 + col

    def test_north_up_east_right(self, key):
        ts, _cells, crs, res = _area(key)
        ds = ts.to_xarray(grid=True)

        # Native coordinates: x grows eastwards, y falls southwards, one pixel apart.
        assert numpy.allclose(numpy.diff(ds['x'].values), res)
        assert numpy.allclose(numpy.diff(ds['y'].values), -res)
        assert ds.attrs['crs'] == crs
        # Geographic sense: the first row is north of the last, the first column
        # west of the last.
        lon2d, lat2d = ds['longitude'].values, ds['latitude'].values
        assert lat2d[0].mean() > lat2d[-1].mean()
        assert lon2d[:, 0].mean() < lon2d[:, -1].mean()

    def test_lonlat_matches_the_pixel_centres(self, key):
        ts, cells, _crs, _res = _area(key)
        ds = ts.to_xarray(grid=True)

        for row, col, lon, lat in cells:
            assert ds['longitude'].values[row, col] == pytest.approx(lon, abs=1e-9)
            assert ds['latitude'].values[row, col] == pytest.approx(lat, abs=1e-9)


class TestGridIsRefusedWhenWrong:
    """A raster that cannot be rebuilt exactly raises instead of coming back wrong."""

    def test_centres_off_the_grid_raise(self):
        ts, _cells, _crs, _res = _area('albers-caso1')
        # Move one centre 3 m (0.3 of a pixel) east in the native CRS.
        location = next(iter(ts.locations.values()))
        to_native = Transformer.from_crs('EPSG:4326', ALBERS_BDC, always_xy=True)
        to_lonlat = Transformer.from_crs(ALBERS_BDC, 'EPSG:4326', always_xy=True)
        x, y = to_native.transform(location.x, location.y)
        location.geom = shapely.geometry.Point(*to_lonlat.transform(x + 3.0, y))

        with pytest.raises(ValueError, match='do not fall on the native grid'):
            ts.to_xarray(grid=True)

    def test_two_pixels_on_one_cell_raise(self):
        ts, cells, _crs, res = _area('albers-caso1')
        _row, _col, lon, lat = cells[0]
        duplicate = TimeSeries(ts._coverage, {
            'results': [pixel(lon, lat, TIMELINE, [1, 2], res),
                        pixel(lon + 1e-10, lat, TIMELINE, [3, 4], res),
                        pixel(*cells[1][2:], TIMELINE, [5, 6], res)],
            'query': {'attributes': ['NDVI'], 'geom': {'type': 'Polygon'}}})

        with pytest.raises(ValueError, match='same cell'):
            duplicate.to_xarray(grid=True)

    def test_coverage_without_crs_raises(self):
        ts, _cells, _crs, _res = _area('albers-caso1')
        ts._coverage.pop('bdc:crs')

        with pytest.raises(ValueError, match='no CRS'):
            ts.to_xarray(grid=True)


class TestGeotiffRoundTrip:
    """The GeoTIFF is the native raster, verified by reading it back."""

    @pytest.mark.parametrize('key', list(GRIDS))
    def test_reread_matches_the_cube(self, key, tmp_path):
        rasterio = pytest.importorskip('rasterio')
        from rasterio.crs import CRS
        from rasterio.transform import from_origin

        ts, cells, crs, res = _area(key)
        ds = ts.to_xarray(grid=True)
        cube = ds['NDVI'].values
        path = str(tmp_path / 'area.tif')
        ts.to_geotiff(path, 'NDVI')

        with rasterio.open(path) as src:
            assert (src.count, src.height, src.width) == cube.shape
            assert src.crs == CRS.from_user_input(crs)
            assert src.transform == from_origin(float(ds['x'][0]) - res / 2,
                                                float(ds['y'][0]) + res / 2, res, res)
            assert src.dtypes == ('float64',) * len(TIMELINE)
            assert math.isnan(src.nodata)
            assert list(src.descriptions) == TIMELINE
            back = src.read()

            # Same values and same holes as the cube in memory.
            assert numpy.array_equal(back, cube, equal_nan=True)

            # Every pixel centre falls inside the file, on the cell holding its value.
            to_native = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
            for row, col, lon, lat in cells:
                x, y = to_native.transform(lon, lat)
                assert src.index(x, y) == (row, col)
                assert back[0, row, col] == row * 1000 + col

    def test_other_crs_is_refused(self, tmp_path):
        pytest.importorskip('rasterio')
        ts, _cells, _crs, _res = _area('albers-caso1')

        with pytest.raises(ValueError, match='resample'):
            ts.to_geotiff(str(tmp_path / 'x.tif'), 'NDVI', crs='EPSG:4326')

    def test_native_crs_may_be_named_explicitly(self, tmp_path):
        rasterio = pytest.importorskip('rasterio')
        ts, _cells, crs, _res = _area('albers-caso1')
        path = str(tmp_path / 'x.tif')
        ts.to_geotiff(path, 'NDVI', crs=crs)

        with rasterio.open(path) as src:
            assert src.count == len(TIMELINE)


def _mocked_query(status, body=None, text=None, geom=None):
    """Run one area query against a mocked server answering ``status``."""
    name = 'S2-16D-2'
    root = {'wtss_version': '2.0',
            'links': [{'rel': 'data', 'title': f'Coverage {name}',
                       'href': f'{MOCK_URL}/{name}'}]}
    geom = geom if geom is not None else shapely.geometry.box(-56.01, -13.56, -55.99, -13.54)
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        rsps.add(responses.GET, f'{MOCK_URL}/', json=root, status=200)
        rsps.add(responses.GET, f'{MOCK_URL}/{name}',
                 json=coverage_metadata(name, ALBERS_BDC, 10.0, TIMELINE), status=200)
        if text is not None:
            rsps.add(responses.POST, f'{MOCK_URL}/{name}/timeseries', body=text, status=status)
        else:
            rsps.add(responses.POST, f'{MOCK_URL}/{name}/timeseries', json=body, status=status)
        ts = WTSS(MOCK_URL)[name].ts(attributes=['NDVI'], geom=geom,
                                     start_datetime='2025-04-01',
                                     end_datetime='2025-04-30').ts
        sent = json.loads(rsps.calls[-1].request.body)
    return ts, sent


class TestServerErrorMessage:
    """The reason the server gives for a rejection reaches the caller."""

    def test_json_description_is_in_the_exception(self):
        reason = 'The polygon area 3006.9316461982726 must be less than 3000ha.'
        with pytest.raises(requests.exceptions.HTTPError) as caught:
            _mocked_query(400, body={'code': 400, 'description': reason})

        error = caught.value
        assert reason in str(error)
        assert '400 Client Error' in str(error)
        assert error.response.status_code == 400
        # The original requests error is kept as the cause, with its traceback.
        assert isinstance(error.__cause__, requests.exceptions.HTTPError)
        assert error.__cause__.__traceback__ is not None

    def test_plain_text_body_is_in_the_exception(self):
        # 500, not 502-504: those are retried by the session before surfacing.
        with pytest.raises(requests.exceptions.HTTPError, match='database unavailable'):
            _mocked_query(500, text='database unavailable')

    def test_empty_body_keeps_the_original_error(self):
        with pytest.raises(requests.exceptions.HTTPError) as caught:
            _mocked_query(400, text='')

        assert str(caught.value).startswith('400 Client Error')
        assert caught.value.__cause__ is None


class TestMultiPolygon:
    """MultiPolygon is accepted by every path that accepts a Polygon."""

    PARTS = shapely.geometry.MultiPolygon([
        shapely.geometry.box(-56.003, -13.553, -56.001, -13.547),
        shapely.geometry.box(-55.999, -13.553, -55.997, -13.547),
    ])

    def _page(self):
        cells = native_cells_in_box(ALBERS_BDC, 10.0, (0.0, 0.0),
                                    box_around(LON, LAT, 0.002))
        return {'results': [pixel(lon, lat, TIMELINE, [0.5, 0.6], 10.0)
                            for _r, _c, lon, lat in cells],
                'query': {'attributes': ['NDVI'], 'geom': {'type': 'MultiPolygon'}}}

    @pytest.mark.parametrize('form', ['shapely', 'geojson-dict', 'geojson-str'])
    def test_multipolygon_is_sent_and_answered(self, form):
        geom = {'shapely': self.PARTS,
                'geojson-dict': shapely.geometry.mapping(self.PARTS),
                'geojson-str': json.dumps(shapely.geometry.mapping(self.PARTS))}[form]
        page = self._page()

        ts, sent = _mocked_query(200, body=page, geom=geom)

        assert sent['geom']['type'] == 'MultiPolygon'
        assert len(sent['geom']['coordinates']) == 2
        assert len(ts.locations) == len(page['results'])
