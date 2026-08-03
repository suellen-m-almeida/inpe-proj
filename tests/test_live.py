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

"""End-to-end tests exercising the whole client against a real WTSS server.

These hit the network, so they are **skipped by default**. Enable them with:

    WTSS_LIVE=1 pytest tests/test_live.py -v         (bash)
    $env:WTSS_LIVE=1; pytest tests\\test_live.py -v    (PowerShell)

Configuration via environment variables (all optional):
    WTSS_SERVER_URL          default https://data.inpe.br/bdc/wtss/v4/
    WTSS_COVERAGE            default mod13q1-6.1
    WTSS_ATTRIBUTE           default NDVI
    BDC_AUTH_CLIENT_SECRET   access token, if the server requires one

Assertions check stable properties (types, shapes, non-emptiness) rather than
exact values, since the server data changes over time. Optional-dependency
paths (xarray/NetCDF, matplotlib, geopandas) skip cleanly when a library is
absent.
"""

import os
import warnings

import pandas
import pytest
import shapely.geometry

LIVE = os.getenv('WTSS_LIVE') == '1'

pytestmark = pytest.mark.skipif(
    not LIVE, reason='set WTSS_LIVE=1 to run live tests against the real server')

URL = os.getenv('WTSS_SERVER_URL', 'https://data.inpe.br/bdc/wtss/v4/')
COVERAGE = os.getenv('WTSS_COVERAGE', 'mod13q1-6.1')
ATTR = os.getenv('WTSS_ATTRIBUTE', 'NDVI')
TOKEN = os.getenv('BDC_AUTH_CLIENT_SECRET')  # may be None for public coverages

LON, LAT = -54.0, -12.0
START, END = '2021-01-01', '2021-06-30'

#: Three well-separated points, so the server returns several locations.
MULTIPOINT = shapely.geometry.MultiPoint([
    shapely.geometry.Point(-54.0, -12.0),
    shapely.geometry.Point(-53.5, -12.0),
    shapely.geometry.Point(-53.0, -11.5),
])


# --------------------------------------------------------------------------- #
# Fixtures (module-scoped to keep the number of HTTP round-trips low).
# --------------------------------------------------------------------------- #
@pytest.fixture(scope='module')
def service():
    from wtss import WTSS
    return WTSS(URL, access_token=TOKEN)


@pytest.fixture(scope='module')
def coverage(service):
    return service[COVERAGE]


@pytest.fixture(scope='module')
def point_ts(coverage):
    """A materialised single-point time series (built the way users do)."""
    return coverage.ts(attributes=[ATTR], latitude=LAT, longitude=LON,
                       start_datetime=START, end_datetime=END).ts


@pytest.fixture(scope='module')
def multipoint_ts(coverage):
    """A materialised multi-location time series."""
    return coverage.ts(attributes=[ATTR], geom=MULTIPOINT,
                       start_datetime=START, end_datetime=END).ts


# --------------------------------------------------------------------------- #
# Service + coverage metadata.
# --------------------------------------------------------------------------- #
def test_coverages_listed(service):
    assert len(service.coverages) > 0
    assert COVERAGE in service.coverages


def test_coverage_timeline(coverage):
    assert len(coverage.timeline) > 0


def test_coverage_has_bands(coverage):
    band_names = [band['name'] for band in coverage.attributes]
    assert ATTR in band_names


# --------------------------------------------------------------------------- #
# TimeSeries: labelled outputs (Ciclos ii/iii).
# --------------------------------------------------------------------------- #
def test_series_is_time_indexed(point_ts):
    series = point_ts[ATTR]
    assert isinstance(series, pandas.Series)
    assert isinstance(series.index, pandas.DatetimeIndex)
    assert len(series) > 0


def test_dataframe_long(point_ts):
    df = point_ts.df(format='long')
    assert list(df.index.names) == ['datetime', 'attribute', 'location']
    assert 'value' in df.columns
    assert len(df) > 0


def test_dataframe_wide(point_ts):
    df = point_ts.df(format='wide')
    assert ATTR in df.columns
    assert isinstance(df.index, pandas.DatetimeIndex)
    assert len(df) > 0


# --------------------------------------------------------------------------- #
# Multipoint + pagination.
# --------------------------------------------------------------------------- #
def test_multipoint_returns_several_locations(multipoint_ts):
    assert multipoint_ts.total_locations >= 2


def test_multipoint_series_has_multiindex(multipoint_ts):
    series = multipoint_ts[ATTR]
    assert isinstance(series.index, pandas.MultiIndex)
    assert list(series.index.names) == ['datetime', 'location']
    # Unstacking by location yields one column per point.
    wide = series.unstack('location')
    assert wide.shape[1] == multipoint_ts.total_locations


# --------------------------------------------------------------------------- #
# Summarize (aggregations).
# --------------------------------------------------------------------------- #
def test_summarize_basic(coverage):
    summarize = coverage.summarize(
        attributes=[ATTR], geom=shapely.geometry.Point(LON, LAT),
        start_datetime=START, end_datetime=END)
    assert ATTR in summarize.attributes
    assert len(summarize.timeline) > 0
    mean = summarize.values(ATTR).values('mean')
    assert len(mean) == len(summarize.timeline)


def test_summarize_df(coverage):
    summarize = coverage.summarize(
        attributes=[ATTR], geom=shapely.geometry.Point(LON, LAT),
        start_datetime=START, end_datetime=END)
    df = summarize.df()
    for column in ('attribute', 'aggregation', 'datetime', 'value'):
        assert column in df.columns
    assert len(df) > 0


# --------------------------------------------------------------------------- #
# xarray + NetCDF export (Ciclo v; optional deps).
# --------------------------------------------------------------------------- #
def test_to_xarray_dims(multipoint_ts):
    pytest.importorskip('xarray')
    dataset = multipoint_ts.to_xarray()
    assert 'time' in dataset.sizes
    assert 'location' in dataset.sizes
    assert ATTR in dataset.data_vars


def test_to_netcdf_returns_bytes(multipoint_ts):
    pytest.importorskip('xarray')
    try:
        data = multipoint_ts.to_netcdf()
    except (ValueError, ImportError) as exc:
        pytest.skip(f'no NetCDF backend available: {exc}')
    assert data is not None and len(data) > 0


# --------------------------------------------------------------------------- #
# Visualisation (headless; optional deps).
# --------------------------------------------------------------------------- #
def test_plot_headless(point_ts):
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # premature fig.show() warns under Agg
        point_ts.plot(attributes=[ATTR])
    assert len(plt.gcf().axes) >= 1
    plt.close('all')


# --------------------------------------------------------------------------- #
# Command-line interface.
# --------------------------------------------------------------------------- #
@pytest.fixture
def runner():
    from click.testing import CliRunner
    return CliRunner()


def test_cli_list_coverages(runner):
    from wtss.cli import cli
    result = runner.invoke(cli, ['list-coverages', '-u', URL])
    assert result.exit_code == 0, result.output
    assert COVERAGE in result.output


def test_cli_describe(runner):
    from wtss.cli import cli
    result = runner.invoke(cli, ['describe', '-u', URL, '-c', COVERAGE])
    assert result.exit_code == 0, result.output


def test_cli_ts(runner):
    from wtss.cli import cli
    result = runner.invoke(cli, [
        'ts', '-u', URL, '-c', COVERAGE, '-a', ATTR,
        '--latitude', str(LAT), '--longitude', str(LON),
        '--start-datetime', START, '--end-datetime', END,
    ])
    assert result.exit_code == 0, result.output
    assert 'timeline' in result.output
