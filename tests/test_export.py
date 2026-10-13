#
# This file is part of Python Client Library for WTSS.
# Copyright (C) 2022 INPE.
#
# Python Client Library for WTSS is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Offline tests for the direct exporters (semana 10).

Each exporter is written to a temporary path and **read back** with the matching
reader to prove the round trip: ``to_zarr`` (xarray), ``to_parquet`` (pandas) and
``to_geotiff`` (rasterio). ``to_netcdf`` already has coverage elsewhere. The data
comes from the same mocked polygon page used by ``test_polygon.py``.
"""

import pytest
import responses
from grade_sintetica import (MODIS_ORIGIN, MODIS_RES, SINUSOIDAL_MODIS,
                             grid_block)

from wtss import WTSS
from wtss.timeseries import TimeSeries

MOCK_URL = 'http://wtss.test'
TIMELINE = ['2017-01-01', '2017-01-17', '2017-02-02']

ROOT_RESPONSE = {
    'wtss_version': '2.0',
    'links': [
        {'rel': 'self', 'href': f'{MOCK_URL}/'},
        {'rel': 'data', 'title': 'Coverage MOD13Q1-6', 'href': f'{MOCK_URL}/MOD13Q1-6'},
    ],
}
COVERAGE_RESPONSE = {
    'fullname': 'MOD13Q1-6', 'description': 'demo',
    'bands': [{'name': 'NDVI', 'nodata': -3000, 'data_type': 'int16',
               'resolution_x': MODIS_RES, 'resolution_y': MODIS_RES}],
    'bdc:crs': SINUSOIDAL_MODIS, 'raster_size': {'xsize': 1, 'ysize': 1},
    'extent': {'type': 'Polygon',
               'coordinates': [[[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]]]},
    'timeline': TIMELINE,
}


def _pixel(lon, lat, values):
    return {
        'pixel_center': {'type': 'Point', 'coordinates': [lon, lat]},
        'pixel_size': [MODIS_RES, -MODIS_RES],
        'time_series': {'timeline': list(TIMELINE), 'values': {'NDVI': list(values)}},
    }


#: A 2x2 block of real MODIS sinusoidal cells, by native (row, col); row 0 is north.
CELLS = {(row, col): (lon, lat) for row, col, lon, lat
         in grid_block(SINUSOIDAL_MODIS, MODIS_RES, MODIS_ORIGIN, -54.0, -12.0, 2, 2)}

#: A full 2x2 grid so the raster exporters have a proper area: south-west,
#: south-east, north-west, north-east.
PIXELS = [
    _pixel(*CELLS[1, 0], [1000, 2000, 3000]),
    _pixel(*CELLS[1, 1], [1100, 2100, 3100]),
    _pixel(*CELLS[0, 0], [1200, 2200, 3200]),
    _pixel(*CELLS[0, 1], [1300, 2300, 3300]),
]

_QUERY = {'attributes': ['NDVI'], 'start_datetime': '2017-01-01',
          'end_datetime': '2017-02-28', 'geom': {'type': 'Polygon'}}


@pytest.fixture
def area_ts():
    """A materialised 2x2-pixel area TimeSeries, built without hitting the network."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        rsps.add(responses.GET, f'{MOCK_URL}/', json=ROOT_RESPONSE, status=200)
        rsps.add(responses.GET, f'{MOCK_URL}/MOD13Q1-6', json=COVERAGE_RESPONSE, status=200)
        coverage = WTSS(MOCK_URL, access_token='x')['MOD13Q1-6']
    return TimeSeries(coverage, {'results': PIXELS, 'query': dict(_QUERY)})


class TestToParquet:
    def test_round_trip_long(self, area_ts, tmp_path):
        pytest.importorskip('pyarrow')
        import pandas as pd

        path = tmp_path / 'ts.parquet'
        area_ts.to_parquet(str(path))
        assert path.exists()

        back = pd.read_parquet(path)
        # long format: one row per (datetime, attribute, location); 3 dates x 4 pixels.
        assert len(back) == 12
        assert 'value' in back.columns


class TestToZarr:
    def test_round_trip_grid(self, area_ts, tmp_path):
        pytest.importorskip('zarr')
        import xarray as xr

        path = tmp_path / 'ts.zarr'
        area_ts.to_zarr(str(path), grid=True)

        ds = xr.open_zarr(str(path))
        assert ds['NDVI'].dims == ('time', 'y', 'x')
        assert ds.sizes == {'time': 3, 'y': 2, 'x': 2}
        # Order-independent: every pixel's first sample survived the round trip.
        assert sorted(ds['NDVI'].isel(time=0).values.ravel().tolist()) == [1000, 1100, 1200, 1300]


class TestToGeotiff:
    def test_round_trip_bands_are_dates(self, area_ts, tmp_path):
        pytest.importorskip('rasterio')
        import numpy as np
        import rasterio

        path = tmp_path / 'ts.tif'
        area_ts.to_geotiff(str(path), 'NDVI')

        with rasterio.open(str(path)) as src:
            assert src.count == 3          # one band per timestamp
            assert (src.width, src.height) == (2, 2)
            assert src.descriptions[0] == '2017-01-01'
            band1 = src.read(1)            # north-up: row 0 is the northern row
            # The south-west pixel held 1000 at t=0.
            assert band1[1, 0] == 1000.0
            # Written on the native sinusoidal grid, not resampled to lon/lat.
            assert src.crs == rasterio.crs.CRS.from_user_input(SINUSOIDAL_MODIS)
            assert src.res == (MODIS_RES, MODIS_RES)

    def test_geotiff_needs_area(self, area_ts, tmp_path):
        pytest.importorskip('rasterio')
        from wtss.timeseries import TimeSeries

        one = TimeSeries(area_ts._coverage, {'results': PIXELS[:1], 'query': dict(_QUERY)})
        with pytest.raises(ValueError, match='at least 2x2'):
            one.to_geotiff(str(tmp_path / 'x.tif'), 'NDVI')

    def test_geotiff_unknown_attribute_raises(self, area_ts, tmp_path):
        pytest.importorskip('rasterio')
        with pytest.raises(KeyError, match='EVI'):
            area_ts.to_geotiff(str(tmp_path / 'x.tif'), 'EVI')
