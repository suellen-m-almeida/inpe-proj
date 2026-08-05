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

"""Legacy parity tests, modernised for the reformulated client API.

Each test here preserves the **intent** of an original test from the pre-TG
baseline, now expressed against the new interface (Ciclos ii-v). The module is
DUAL-MODE, so the same assertions run **with and without a server**:

- Offline (default): a ``responses`` mock stands in for the BDC server, so the
  whole file runs with no network and no token.
- Live (``WTSS_LIVE=1``): the exact same tests run against the real server
  (``WTSS_SERVER_URL``, default v4), on coverage ``mod13q1-6.1``.

Wherever a test compares the client against the raw HTTP response, the raw
request hits the same backend as the client (mock or live), so the assertions
are backend-agnostic.

Mapping (old test -> new equivalent):
  TestSucessRequests.test_timeseries_success_MOD13Q1 -> TestTimeSeriesParity
  TestSucessRequests.test_summarize_success_MOD13Q1  -> TestSummarizeParity
  TestSucessRequests.test_coverage_success_MOD13Q1   -> TestCoverageMetadata
  TestDifferentGeometries.test_GeoJSON_valid_*       -> TestGeometryValidation.test_supported_*
  TestDifferentGeometries.test_GeoJSON_<invalid>     -> TestGeometryValidation.test_unsupported_*

Note on the geometry checks: the baseline expected the *server* to reject
non-Point/Polygon geometries with HTTP 400. The reformulated client validates
geometries **client-side** (fail-fast, no network) and now also accepts
MultiPoint, so the modern equivalent asserts a local ``ValueError`` for
unsupported types and no error for the supported ones.
"""

import os

import pytest
import requests
import responses

from wtss import WTSS

# --- Backend selection --------------------------------------------------------

LIVE = os.getenv('WTSS_LIVE') == '1'
LIVE_URL = os.getenv('WTSS_SERVER_URL', 'https://data.inpe.br/bdc/wtss/v4/')
LIVE_TOKEN = os.getenv('BDC_AUTH_CLIENT_SECRET')  # may be None for public coverages
LIVE_COVERAGE = os.getenv('WTSS_COVERAGE', 'mod13q1-6.1')

MOCK_URL = 'http://wtss.test'
MOCK_COVERAGE = 'MOD13Q1-6'

# --- Shared query parameters --------------------------------------------------

ATTRS = ['NDVI']
START = '2017-01-01T00:00:00Z'
END = '2017-02-28T00:00:00Z'
AGGREGATIONS = ['mean', 'std']

POINT = {'type': 'Point', 'coordinates': [-54.0, -12.0]}
POLYGON = {'type': 'Polygon', 'coordinates': [[[-54, -12], [-53.99, -12],
                                               [-53.99, -11.99], [-54, -11.99],
                                               [-54, -12]]]}
MULTIPOINT = {'type': 'MultiPoint', 'coordinates': [[-54.0, -12.0], [-53.9, -11.9]]}

LINESTRING = {'type': 'LineString', 'coordinates': [[-54, -12], [-53.99, -11.99]]}
MULTILINESTRING = {'type': 'MultiLineString',
                   'coordinates': [[[-10, -75], [-10, 75]], [[10, -75], [10, 75]]]}
MULTIPOLYGON = {'type': 'MultiPolygon',
                'coordinates': [[[[-50, 60], [-30, 60], [-30, 80], [-50, 80], [-50, 60]]]]}
GEOMETRYCOLLECTION = {'type': 'GeometryCollection',
                      'geometries': [{'type': 'Point', 'coordinates': [40, -50]}]}

# --- Canonical mock payloads (used only in offline mode) ----------------------

ROOT_RESPONSE = {
    'wtss_version': '2.0',
    'links': [
        {'rel': 'self', 'title': 'WTSS', 'href': f'{MOCK_URL}/'},
        {'rel': 'data', 'title': f'Coverage {MOCK_COVERAGE}',
         'href': f'{MOCK_URL}/{MOCK_COVERAGE}'},
    ],
}

COVERAGE_RESPONSE = {
    'fullname': MOCK_COVERAGE,
    'description': 'MODIS Vegetation Indices 16-day 250m (MOD13Q1) v6.',
    'bands': [
        {'name': 'NDVI', 'nodata': -3000, 'data_type': 'int16'},
        {'name': 'EVI', 'nodata': -3000, 'data_type': 'int16'},
    ],
    'bdc:crs': '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181',
    'raster_size': {'xsize': 172800, 'ysize': 86400},
    'extent': {
        'type': 'Polygon',
        'coordinates': [[[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]]],
    },
    # deliberately unsorted, to exercise Coverage.timeline sorting (B7).
    'timeline': ['2017-01-17', '2017-01-01', '2017-02-02'],
}

TIMESERIES_RESPONSE = {
    'results': [
        {
            'pixel_center': {'type': 'Point', 'coordinates': [-54.0, -12.0]},
            'pixel_size': [231.65, 231.65],
            'time_series': {
                'timeline': ['2017-01-01', '2017-01-17', '2017-02-02'],
                'values': {'NDVI': [1000, 2000, 3000]},
            },
        }
    ],
    'query': {
        'attributes': ATTRS,
        'start_datetime': START,
        'end_datetime': END,
        'geom': POINT,
    },
}

SUMMARIZE_RESPONSE = {
    'query': {
        'attributes': ATTRS,
        'start_datetime': START,
        'end_datetime': END,
        'geom': POLYGON,
        'aggregations': AGGREGATIONS,
    },
    'results': {
        'timeline': ['2017-01-01', '2017-01-17', '2017-02-02'],
        'values': {
            'NDVI': {'mean': [0.5, 0.6, 0.7], 'std': [0.1, 0.1, 0.1]},
        },
    },
}


class Backend:
    """Thin adapter that lets each test talk to either the mock or the live server.

    Exposes the WTSS client plus raw HTTP helpers pointed at the same base URL,
    so ``client output == raw response`` comparisons hold in both modes.
    """

    def __init__(self, base_url, coverage, token):
        self.base = base_url.rstrip('/') + '/'
        self.coverage = coverage
        self.token = token
        self.service = WTSS(base_url, access_token=token)

    def _headers(self):
        return {'x-api-key': self.token} if self.token else {}

    def raw_timeseries(self, geom):
        resp = requests.post(
            f'{self.base}{self.coverage}/timeseries',
            json={'attributes': ATTRS, 'start_datetime': START,
                  'end_datetime': END, 'geom': geom},
            headers=self._headers(),
        )
        return resp.json()

    def raw_summarize(self, geom):
        resp = requests.post(
            f'{self.base}{self.coverage}/summarize',
            json={'attributes': ATTRS, 'start_datetime': START,
                  'end_datetime': END, 'geom': geom,
                  'aggregations': AGGREGATIONS},
            headers=self._headers(),
        )
        return resp.json()

    def search(self, geom):
        return self.service[self.coverage].ts(
            attributes=ATTRS, geom=geom, start_datetime=START, end_datetime=END)

    def summarize(self, geom):
        return self.service[self.coverage].summarize(
            attributes=ATTRS, geom=geom, start_datetime=START, end_datetime=END,
            aggregations=AGGREGATIONS)

    def cov(self):
        return self.service[self.coverage]


@pytest.fixture
def backend():
    """Yield a Backend wired to the live server or to a responses mock."""
    if LIVE:
        yield Backend(LIVE_URL, LIVE_COVERAGE, LIVE_TOKEN)
    else:
        with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
            rsps.add(responses.GET, f'{MOCK_URL}/', json=ROOT_RESPONSE, status=200)
            rsps.add(responses.GET, f'{MOCK_URL}/{MOCK_COVERAGE}',
                     json=COVERAGE_RESPONSE, status=200)
            rsps.add(responses.POST, f'{MOCK_URL}/{MOCK_COVERAGE}/timeseries',
                     json=TIMESERIES_RESPONSE, status=200)
            rsps.add(responses.POST, f'{MOCK_URL}/{MOCK_COVERAGE}/summarize',
                     json=SUMMARIZE_RESPONSE, status=200)
            yield Backend(MOCK_URL, MOCK_COVERAGE, 'fake-token')


class TestTimeSeriesParity:
    """Parity with the old ``test_timeseries_success_MOD13Q1``.

    The client's parsed time series must reflect exactly what the server sent.
    """

    def test_number_of_pixels_matches_server(self, backend):
        raw = backend.raw_timeseries(POINT)
        assert backend.search(POINT).total_locations() == len(raw['results'])

    def test_attributes_match_server(self, backend):
        raw = backend.raw_timeseries(POINT)
        ts = backend.search(POINT).ts
        expected = list(raw['results'][0]['time_series']['values'].keys())
        assert ts.attributes == expected

    def test_timeline_matches_server(self, backend):
        raw = backend.raw_timeseries(POINT)
        ts = backend.search(POINT).ts
        assert ts.timeline == raw['results'][0]['time_series']['timeline']

    def test_values_match_server(self, backend):
        raw = backend.raw_timeseries(POINT)
        ts = backend.search(POINT).ts
        assert ts.values('NDVI')[0] == raw['results'][0]['time_series']['values']['NDVI']

    def test_dataframe_values_match_server(self, backend):
        raw = backend.raw_timeseries(POINT)
        df = backend.search(POINT).df()
        got = df[df['attribute'] == 'NDVI']['value'].tolist()
        assert got == raw['results'][0]['time_series']['values']['NDVI']


class TestSummarizeParity:
    """Parity with the old ``test_summarize_success_MOD13Q1``."""

    def test_attributes_match_server(self, backend):
        raw = backend.raw_summarize(POLYGON)
        summ = backend.summarize(POLYGON)
        assert summ.attributes == list(raw['results']['values'].keys())

    def test_timeline_matches_server(self, backend):
        raw = backend.raw_summarize(POLYGON)
        assert backend.summarize(POLYGON).timeline == raw['results']['timeline']

    def test_mean_values_match_server(self, backend):
        raw = backend.raw_summarize(POLYGON)
        summ = backend.summarize(POLYGON)
        assert summ.values('NDVI').values('mean') == raw['results']['values']['NDVI']['mean']

    def test_aggregations_match_request(self, backend):
        assert backend.summarize(POLYGON).aggregations == AGGREGATIONS

    def test_geometry_echoed(self, backend):
        assert backend.summarize(POLYGON).geometry['type'] == 'Polygon'

    def test_dataframe_mean_matches_server(self, backend):
        raw = backend.raw_summarize(POLYGON)
        df = backend.summarize(POLYGON).df()
        got = df[(df['attribute'] == 'NDVI') & (df['aggregation'] == 'mean')]['value'].tolist()
        assert got == raw['results']['values']['NDVI']['mean']


class TestCoverageMetadata:
    """Parity with the old ``test_coverage_success_MOD13Q1``.

    The baseline cross-checked the Coverage against the STAC collection. The
    modern equivalent asserts the same metadata is parsed and well-formed,
    without coupling the test to a second (STAC) service.
    """

    def test_name(self, backend):
        assert backend.cov().name == backend.coverage

    def test_attributes_include_ndvi(self, backend):
        names = [b['name'] if isinstance(b, dict) else b for b in backend.cov().attributes]
        assert 'NDVI' in names

    def test_description_present(self, backend):
        assert backend.cov().description

    def test_crs_present(self, backend):
        assert backend.cov().crs

    def test_dimensions_present(self, backend):
        assert backend.cov().dimensions is not None

    def test_timeline_sorted_and_nonempty(self, backend):
        timeline = backend.cov().timeline
        assert len(timeline) > 0
        assert timeline == sorted(timeline)


class TestGeometryValidation:
    """Parity with the old ``TestDifferentGeometries``.

    Unsupported geometries are now rejected client-side with ValueError; the
    supported ones (Point, Polygon and, newly, MultiPoint) are accepted.
    """

    @pytest.mark.parametrize('geom', [POINT, POLYGON, MULTIPOINT],
                             ids=['point', 'polygon', 'multipoint'])
    def test_supported_geometry_accepted(self, backend, geom):
        # Building the search validates the geometry client-side; must not raise.
        assert backend.search(geom) is not None

    @pytest.mark.parametrize('geom',
                             [LINESTRING, MULTILINESTRING, MULTIPOLYGON, GEOMETRYCOLLECTION],
                             ids=['linestring', 'multilinestring',
                                  'multipolygon', 'geometrycollection'])
    def test_unsupported_geometry_rejected(self, backend, geom):
        with pytest.raises(ValueError):
            backend.search(geom)
