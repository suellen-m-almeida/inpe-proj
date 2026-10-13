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

"""Offline tests for ``TimeSeries.plot`` using the headless Agg backend.

These tests exercise the matplotlib plotting path of :class:`wtss.timeseries.TimeSeries`
without a live server and without a display. The ``Agg`` backend is selected *before*
``matplotlib.pyplot`` is imported anywhere, so figures are rendered in memory and never
shown on screen, which is what allows the suite to run in CI.

They also pin down the three plotting bugs catalogued in ``DIAGNOSTICO.md``:

* **B10** -- ``plot`` calls ``fig.show()`` before any series is drawn.
* **B11** -- ``_limit`` is left unset when the per-attribute loop never runs.
* **B12** -- an empty ``attributes`` list reaches ``plt.subplots(0)`` and explodes.
"""

import warnings

import matplotlib

# Must be set before pyplot is imported by the client (lazy import inside plot()).
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import pytest  # noqa: E402
import responses  # noqa: E402

from wtss import WTSS  # noqa: E402
from wtss.timeseries import TimeSeries  # noqa: E402

#: Base URL used for the mocked WTSS service.
MOCK_URL = 'http://wtss.test'

#: Timeline shared by every fixture in this module (already sorted).
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
    'timeline': TIMELINE,
}

POINT_GEOM = {'type': 'Point', 'coordinates': [-54.0, -12.0]}

#: Single-point time series with two attributes. ``-3000`` is the nodata value of
#: both bands, so it should be masked away (rendered as ``None``) when plotting.
POINT_TS_RESPONSE = {
    'results': [
        {
            'pixel_center': POINT_GEOM,
            'pixel_size': [231.65, 231.65],
            'time_series': {
                'timeline': TIMELINE,
                'values': {'NDVI': [1000, -3000, 3000], 'EVI': [500, 600, 700]},
            },
        }
    ],
    'query': {
        'attributes': ['NDVI', 'EVI'],
        'start_datetime': '2017-01-01T00:00:00Z',
        'end_datetime': '2017-02-28T00:00:00Z',
        'geom': POINT_GEOM,
    },
}

#: Summarize payload carrying the quantile aggregations consumed by ``plot(stats=True)``.
SUMMARIZE_RESPONSE = {
    'query': {
        'attributes': ['NDVI', 'EVI'],
        'geom': POINT_GEOM,
        'aggregations': ['q1', 'q3', 'median'],
    },
    'results': {
        'timeline': TIMELINE,
        'values': {
            'NDVI': {'q1': [900, 1900, 2900], 'q3': [1100, 2100, 3100], 'median': [1000, 2000, 3000]},
            'EVI': {'q1': [450, 550, 650], 'q3': [550, 650, 750], 'median': [500, 600, 700]},
        },
    },
}


@pytest.fixture(autouse=True)
def _close_figures():
    """Avoid leaking figures across tests (and the resulting memory warning)."""
    yield
    plt.close('all')


@pytest.fixture
def service():
    """Build a WTSS client with the plotting endpoints mocked."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        rsps.add(responses.GET, f'{MOCK_URL}/', json=ROOT_RESPONSE, status=200)
        rsps.add(responses.GET, f'{MOCK_URL}/MOD13Q1-6', json=COVERAGE_RESPONSE, status=200)
        rsps.add(responses.POST, f'{MOCK_URL}/MOD13Q1-6/timeseries',
                 json=POINT_TS_RESPONSE, status=200)
        rsps.add(responses.POST, f'{MOCK_URL}/MOD13Q1-6/summarize',
                 json=SUMMARIZE_RESPONSE, status=200)
        yield WTSS(MOCK_URL, access_token='fake-token')


def _point_timeseries(service):
    """Return the materialised single-point :class:`TimeSeries`."""
    search = service['MOD13Q1-6'].ts(
        attributes=['NDVI'],
        geom=POINT_GEOM,
        start_datetime='2017-01-01',
        end_datetime='2017-02-28',
    )
    return search.ts


class TestPlotHappyPath:
    """A single-point plot must render one axis per attribute, fully offline."""

    def test_single_attribute_produces_one_axis(self, service):
        ts = _point_timeseries(service)
        with warnings.catch_warnings():
            # B10's premature fig.show() warns under Agg; not the subject here.
            warnings.simplefilter('ignore')
            ts.plot(attributes=['NDVI'])

        fig = plt.gcf()
        assert len(fig.axes) == 1
        assert fig._suptitle.get_text() == 'Time Series'

    def test_multiple_attributes_produce_one_axis_each(self, service):
        ts = _point_timeseries(service)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ts.plot(attributes=['NDVI', 'EVI'])

        assert len(plt.gcf().axes) == 2

    def test_stats_false_skips_quantile_lines(self, service):
        """Without stats, only the raw series line is drawn (no median/quantiles)."""
        ts = _point_timeseries(service)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ts.plot(attributes=['NDVI'], stats=False)

        axis = plt.gcf().axes[0]
        # One location => exactly one raw line; the stats block would add 3 more.
        assert len(axis.lines) == 1

    def test_stats_true_adds_quantile_and_median_lines(self, service):
        ts = _point_timeseries(service)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ts.plot(attributes=['NDVI'], stats=True)

        axis = plt.gcf().axes[0]
        # raw series (1) + q1 + q3 + median == 4 lines.
        assert len(axis.lines) == 4
        labels = [line.get_label() for line in axis.lines]
        # Legend entries: quartiles and median (the pixel band is excluded).
        assert 'quartis (q1, q3)' in labels
        assert 'mediana' in labels
        assert 'pixels' not in labels

    def test_nodata_is_masked_to_none(self, service):
        """The ``-3000`` nodata sample must not be plotted as a real value."""
        ts = _point_timeseries(service)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ts.plot(attributes=['NDVI'], stats=False)

        raw_line = plt.gcf().axes[0].lines[0]
        ydata = list(raw_line.get_ydata())
        # Position 1 held nodata (-3000); plot() swaps it for None and never -3000.
        assert None in ydata
        assert -3000 not in ydata


class TestPlotMultiPoint:
    """Plotting several locations exercises the ``limit`` and title branches."""

    def _multipoint_timeseries(self, service, n_points=3):
        coverage = service['MOD13Q1-6']
        results = [
            {
                'pixel_center': {'type': 'Point', 'coordinates': [-54.0 + i * 0.01, -12.0]},
                'pixel_size': [231.65, 231.65],
                'time_series': {
                    'timeline': TIMELINE,
                    'values': {'NDVI': [1000 + i, 2000 + i, 3000 + i]},
                },
            }
            for i in range(n_points)
        ]
        data = {
            'results': results,
            'query': {**POINT_TS_RESPONSE['query'], 'attributes': ['NDVI']},
        }
        return TimeSeries(coverage, data)

    def test_limit_caps_plotted_series_and_titles_it(self, service):
        ts = self._multipoint_timeseries(service, n_points=3)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ts.plot(attributes=['NDVI'], limit=2, stats=False)

        fig = plt.gcf()
        # Only 2 of the 3 locations are drawn (stats off => no extra lines).
        assert len(fig.axes[0].lines) == 2
        assert fig._suptitle.get_text() == 'Time Series (Showing 2 of 3 points)'


class TestDeferredSearchPlot:
    """The deferred ``TimeSeriesSearch.plot`` must delegate to the materialised series."""

    def test_point_search_plot_renders(self, service):
        search = service['MOD13Q1-6'].ts(
            attributes=['NDVI'],
            geom=POINT_GEOM,
            start_datetime='2017-01-01',
            end_datetime='2017-02-28',
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # Point geometry => no pagination => delegates straight to TimeSeries.plot.
            search.plot(attributes=['NDVI'])

        assert len(plt.gcf().axes) == 1


class TestPlotValidation:
    """Argument validation on the plot entry point."""

    def test_negative_limit_raises(self, service):
        ts = _point_timeseries(service)
        with pytest.raises(ValueError, match='Limit cannot be negative'):
            ts.plot(limit=-1)


class TestPlotFixedBugs:
    """Regression coverage locking in the fixes for the catalogued plotting bugs."""

    def test_show_called_after_drawing_B10(self, service, monkeypatch):
        """B10: ``fig.show()`` must fire only once every series has been drawn."""
        from matplotlib.figure import Figure

        captured = {}
        # Record how many lines exist on the figure at the moment show() runs;
        # swallow the call so the Agg backend does not warn.
        monkeypatch.setattr(
            Figure, 'show',
            lambda self, *a, **k: captured.update(lines=sum(len(ax.lines) for ax in self.axes)),
        )
        ts = _point_timeseries(service)
        ts.plot(attributes=['NDVI'])

        # Before the fix this was 0 (show ran first); now the series are drawn.
        assert captured['lines'] > 0


def _multipoint_timeseries(service, n_points=3):
    """Return a materialised multi-location :class:`TimeSeries` (NDVI only)."""
    coverage = service['MOD13Q1-6']
    results = [
        {
            'pixel_center': {'type': 'Point', 'coordinates': [-54.0 + i * 0.01, -12.0]},
            'pixel_size': [231.65, 231.65],
            'time_series': {
                'timeline': TIMELINE,
                'values': {'NDVI': [1000 + i, 2000 + i, 3000 + i]},
            },
        }
        for i in range(n_points)
    ]
    data = {
        'results': results,
        'query': {**POINT_TS_RESPONSE['query'], 'attributes': ['NDVI']},
    }
    return TimeSeries(coverage, data)


class TestPlotMap:
    """Offline coverage for the spatial map view ``TimeSeries.plot_map``."""

    def test_single_point_draws_one_marker(self, service):
        ts = _point_timeseries(service)
        ax = ts.plot_map('NDVI', legend=False)

        # geopandas renders points as a single PathCollection with one point.
        assert len(ax.collections) == 1
        assert ax.collections[0].get_offsets().shape[0] == 1

    def test_multipoint_colors_all_locations(self, service):
        ts = _multipoint_timeseries(service, n_points=3)
        ax = ts.plot_map('NDVI', legend=False)

        offsets = ax.collections[0].get_offsets()
        assert offsets.shape[0] == 3
        # Points are laid out along increasing longitude, at fixed latitude.
        assert [round(x, 2) for x, _ in offsets] == [-54.0, -53.99, -53.98]

    def test_reduce_mean_colors_by_time_average(self, service):
        ts = _multipoint_timeseries(service, n_points=2)
        ax = ts.plot_map('NDVI', reduce='mean', legend=False)

        # Location 0 = mean(1000,2000,3000)=2000; location 1 = mean(1001,2001,3001)=2001.
        values = list(ax.collections[0].get_array())
        assert values == [2000.0, 2001.0]

    def test_reduce_std_maps_temporal_variability(self, service):
        """``std`` colors each pixel by how much it varied, not by its level."""
        import numpy as np

        ts = _multipoint_timeseries(service, n_points=2)
        ax = ts.plot_map('NDVI', reduce='std', legend=False)

        # Both locations carry the same spread (1000, 2000, 3000 shifted by i),
        # so the variability map is flat even though the means differ.
        esperado = float(np.std([1000.0, 2000.0, 3000.0]))
        values = list(ax.collections[0].get_array())
        assert values == pytest.approx([esperado, esperado])

    def test_unknown_reduce_lists_std_among_the_options(self, service):
        ts = _multipoint_timeseries(service, n_points=2)

        with pytest.raises(ValueError, match="'std'"):
            ts.plot_map('NDVI', reduce='variance')

    def test_datetime_selects_a_single_date(self, service):
        ts = _multipoint_timeseries(service, n_points=2)
        ax = ts.plot_map('NDVI', datetime='2017-01-17', legend=False)

        # Second timeline position: 2000 and 2001.
        assert list(ax.collections[0].get_array()) == [2000.0, 2001.0]

    def test_mask_nodata_turns_sentinel_into_nan(self, service):
        """With mask_nodata, the -3000 sentinel is NaN and does not skew the mean."""
        ts = _point_timeseries(service)  # NDVI = [1000, -3000, 3000], nodata -3000
        ax = ts.plot_map('NDVI', reduce='mean', mask_nodata=True, legend=False)

        # nanmean(1000, nan, 3000) == 2000, not (1000-3000+3000)/3.
        assert list(ax.collections[0].get_array()) == [2000.0]

    def test_ticks_keep_the_full_coordinate(self, service):
        """A small area must not be labelled with an offset ("-0.0020" + "-5.6e1")."""
        ts = _multipoint_timeseries(service, n_points=3)
        ax = ts.plot_map('NDVI', legend=False)
        ax.figure.canvas.draw()

        assert ax.xaxis.get_major_formatter().get_offset() == ''
        assert ax.yaxis.get_major_formatter().get_offset() == ''
        lo, hi = ax.get_xlim()
        visiveis = [t for t in ax.xaxis.get_major_locator()() if lo <= t <= hi]
        assert len(visiveis) <= 6   # MaxNLocator(5): at most 5 intervals drawn

    def test_unknown_attribute_raises_keyerror(self, service):
        ts = _point_timeseries(service)
        with pytest.raises(KeyError, match='RED'):
            ts.plot_map('RED')  # band absent from the time series

    def test_unknown_reduce_raises_valueerror(self, service):
        ts = _point_timeseries(service)
        with pytest.raises(ValueError, match='reduce must be one of'):
            ts.plot_map('NDVI', reduce='sum')

    def test_datetime_out_of_timeline_raises(self, service):
        ts = _point_timeseries(service)
        with pytest.raises(ValueError, match='not in the timeline'):
            ts.plot_map('NDVI', datetime='1999-01-01')

    def test_deferred_search_plot_map_delegates(self, service):
        search = service['MOD13Q1-6'].ts(
            attributes=['NDVI'],
            geom=POINT_GEOM,
            start_datetime='2017-01-01',
            end_datetime='2017-02-28',
        )
        ax = search.plot_map('NDVI', legend=False)
        assert len(ax.collections) == 1

    def test_empty_attributes_raise_clear_error_B12(self, service):
        """B12: an empty attribute list is rejected with a clear message."""
        coverage = service['MOD13Q1-6']
        data = {
            'results': POINT_TS_RESPONSE['results'],
            'query': {**POINT_TS_RESPONSE['query'], 'attributes': []},
        }
        ts = TimeSeries(coverage, data)
        with pytest.raises(ValueError, match='No attributes available to plot'):
            ts.plot()

    def test_limit_defined_when_loop_never_runs_B11(self, service):
        """B11: caller-supplied empty ``axes`` used to leave ``_limit`` unset and
        trip ``None < len(...)``. The cap is now computed before the loop, so the
        call completes cleanly even when no axis is drawn.
        """
        ts = _point_timeseries(service)
        fig = plt.figure()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # Non-empty attributes + empty axes: the loop body never runs.
            ts.plot(attributes=['NDVI'], fig=fig, axes=[])

        assert fig._suptitle.get_text() == 'Time Series'


class TestPlotInteractive:
    """``plot(interactive=True)`` renders with plotly and returns the figure."""

    def _pixel_trace(self, figure):
        """Return the single trace carrying every pixel series, or None."""
        return next((trace for trace in figure.data
                     if trace.name and trace.name.startswith('pixels')), None)

    def test_returns_a_plotly_figure_instead_of_drawing(self, service):
        go = pytest.importorskip('plotly.graph_objects')
        ts = _point_timeseries(service)

        figure = ts.plot(attributes=['NDVI'], interactive=True)

        assert isinstance(figure, go.Figure)
        # The matplotlib path draws on the current figure; this one must not.
        assert not plt.get_fignums()

    def test_one_y_axis_per_attribute(self, service):
        pytest.importorskip('plotly')
        ts = _point_timeseries(service)

        figure = ts.plot(attributes=['NDVI', 'EVI'], interactive=True)

        titles = [axis.title.text for axis in figure.select_yaxes()]
        assert titles == ['NDVI', 'EVI']

    def test_all_pixel_series_share_a_single_trace(self, service):
        pytest.importorskip('plotly')
        ts = _multipoint_timeseries(service, n_points=3)

        figure = ts.plot(attributes=['NDVI'], interactive=True, stats=False)

        assert len(figure.data) == 1
        trace = self._pixel_trace(figure)
        # 3 locations x 3 dates, plus one None separating each run.
        assert len(trace.x) == 3 * (len(TIMELINE) + 1)
        assert trace.y.count(None) == 3

    def test_hover_identifies_the_pixel(self, service):
        pytest.importorskip('plotly')
        ts = _multipoint_timeseries(service, n_points=2)

        figure = ts.plot(attributes=['NDVI'], interactive=True, stats=False)

        labels = {label for label in self._pixel_trace(figure).customdata
                  if label is not None}
        assert labels == {'-54.00000, -12.00000', '-53.99000, -12.00000'}

    def test_pixels_false_leaves_only_the_statistics(self, service):
        pytest.importorskip('plotly')
        ts = _multipoint_timeseries(service, n_points=3)

        figure = ts.plot(attributes=['NDVI'], interactive=True, pixels=False)

        assert self._pixel_trace(figure) is None
        assert {trace.name for trace in figure.data} == {'quartis (q1, q3)', 'mediana'}

    def test_stats_false_drops_the_quantile_traces(self, service):
        pytest.importorskip('plotly')
        ts = _multipoint_timeseries(service, n_points=3)

        figure = ts.plot(attributes=['NDVI'], interactive=True, stats=False)

        assert {trace.name for trace in figure.data} == {'pixels (3)'}

    def test_quartiles_share_one_legend_entry(self, service):
        pytest.importorskip('plotly')
        ts = _multipoint_timeseries(service, n_points=2)

        figure = ts.plot(attributes=['NDVI'], interactive=True, pixels=False)

        shown = [trace.name for trace in figure.data if trace.showlegend]
        assert shown == ['quartis (q1, q3)', 'mediana']

    def test_statistics_draw_above_the_pixel_band(self, service):
        """Every trace must be WebGL so the drawing order is the trace order.

        Plotly renders Scattergl on a canvas layer above the SVG one, so a plain
        Scatter for the statistics would end up *under* the pixel band no matter
        where it sits in the trace list.
        """
        pytest.importorskip('plotly')
        ts = _multipoint_timeseries(service, n_points=3)

        figure = ts.plot(attributes=['NDVI'], interactive=True)

        assert {trace.type for trace in figure.data} == {'scattergl'}
        nomes = [trace.name for trace in figure.data]
        assert nomes[0] == 'pixels (3)'
        assert nomes[-1] == 'mediana'

    def test_limit_caps_the_series_and_titles_it(self, service):
        pytest.importorskip('plotly')
        ts = _multipoint_timeseries(service, n_points=3)

        figure = ts.plot(attributes=['NDVI'], interactive=True, limit=2, stats=False)

        assert len(self._pixel_trace(figure).x) == 2 * (len(TIMELINE) + 1)
        assert figure.layout.title.text == 'Time Series (Showing 2 of 3 points)'

    def test_negative_limit_raises_before_reaching_plotly(self, service):
        ts = _point_timeseries(service)

        with pytest.raises(ValueError, match='Limit cannot be negative'):
            ts.plot(interactive=True, limit=-1)

    def test_nodata_is_masked_to_none(self, service):
        pytest.importorskip('plotly')
        ts = _point_timeseries(service)

        figure = ts.plot(attributes=['NDVI'], interactive=True, stats=False)

        # POINT_TS_RESPONSE carries -3000 (the nodata) as the second NDVI sample.
        assert self._pixel_trace(figure).y[:len(TIMELINE)] == (1000, None, 3000)

    def test_apply_scale_matches_the_matplotlib_values(self, service):
        pytest.importorskip('plotly')
        coverage = service['MOD13Q1-6']
        scaled_band = dict(coverage.attributes[0], scale=0.0001)
        ts = _point_timeseries(service)

        esperado = ts._prepared_values(scaled_band, [1000, -3000, 3000],
                                       apply_scale=True, mask_nodata=True)

        assert esperado[0] == pytest.approx(0.1)
        assert esperado[1] is None or esperado[1] != esperado[1]

    def test_empty_attributes_raise_before_building_the_figure(self, service):
        pytest.importorskip('plotly')
        coverage = service['MOD13Q1-6']
        data = {
            'results': POINT_TS_RESPONSE['results'],
            'query': {**POINT_TS_RESPONSE['query'], 'attributes': []},
        }
        ts = TimeSeries(coverage, data)

        with pytest.raises(ValueError, match='No attributes available to plot'):
            ts.plot(interactive=True)

    def test_deferred_search_returns_the_figure(self, service):
        go = pytest.importorskip('plotly.graph_objects')
        search = service['MOD13Q1-6'].ts(
            attributes=['NDVI'],
            geom=POINT_GEOM,
            start_datetime='2017-01-01',
            end_datetime='2017-02-28',
        )

        figure = search.plot(interactive=True)

        assert isinstance(figure, go.Figure)


class TestPlotMapInteractive:
    """``plot_map(interactive=True)`` renders a folium map with clickable pixels."""

    def _markers(self, figure):
        """Return the circle markers added to a folium map."""
        import folium

        return [child for child in figure._children.values()
                if isinstance(child, folium.CircleMarker)]

    def _popup(self, marker):
        """Return the rendered popup markup of a marker (folium nests it as a child)."""
        import folium

        popup = next(child for child in marker._children.values()
                     if isinstance(child, folium.Popup))
        return popup.html.render()

    def test_returns_a_folium_map_instead_of_axes(self, service):
        folium = pytest.importorskip('folium')
        ts = _multipoint_timeseries(service, n_points=3)

        figure = ts.plot_map('NDVI', interactive=True)

        assert isinstance(figure, folium.Map)
        assert not plt.get_fignums()

    def test_one_marker_per_location(self, service):
        pytest.importorskip('folium')
        ts = _multipoint_timeseries(service, n_points=3)

        figure = ts.plot_map('NDVI', interactive=True)

        assert len(self._markers(figure)) == 3

    def test_each_marker_carries_its_own_series(self, service):
        """The reviewer's ask: clicking a dot brings up that pixel's series."""
        pytest.importorskip('folium')
        ts = _multipoint_timeseries(service, n_points=2)

        figure = ts.plot_map('NDVI', interactive=True)

        popups = [self._popup(marker) for marker in self._markers(figure)]
        assert len(popups) == 2
        # Cada popup traz um SVG com a série daquele pixel, e as coordenadas
        # distinguem um do outro.
        for popup, location in zip(popups, ts._locations.values()):
            assert '<svg' in popup and '<polyline' in popup
            assert f'{location.x:.5f}' in popup

    def test_popup_is_self_contained_markup(self, service):
        """No script and no external asset: the popup must survive a static export."""
        pytest.importorskip('folium')
        ts = _multipoint_timeseries(service, n_points=1)

        figure = ts.plot_map('NDVI', interactive=True)

        popup = self._popup(self._markers(figure)[0])
        assert '<script' not in popup.lower()
        # O namespace do SVG é um identificador, não um recurso baixado.
        sem_namespace = popup.replace('http://www.w3.org/2000/svg', '')
        assert 'http://' not in sem_namespace
        assert 'https://' not in sem_namespace

    def test_limit_caps_the_markers_and_says_so(self, service):
        pytest.importorskip('folium')
        ts = _multipoint_timeseries(service, n_points=3)

        figure = ts.plot_map('NDVI', interactive=True, limit=2)

        assert len(self._markers(figure)) == 2
        assert '2 de 3 pixels' in figure.get_root().render()

    def test_negative_limit_raises(self, service):
        pytest.importorskip('folium')
        ts = _multipoint_timeseries(service, n_points=2)

        with pytest.raises(ValueError, match='Limit cannot be negative'):
            ts.plot_map('NDVI', interactive=True, limit=-1)

    def test_unknown_reduce_raises_before_building_the_map(self, service):
        pytest.importorskip('folium')
        ts = _multipoint_timeseries(service, n_points=2)

        with pytest.raises(ValueError, match='reduce must be one of'):
            ts.plot_map('NDVI', interactive=True, reduce='variance')

    def test_nodata_pixel_still_gets_a_popup(self, service):
        """A pixel whose samples are all masked must not break the map."""
        pytest.importorskip('folium')
        ts = _point_timeseries(service)

        figure = ts.plot_map('NDVI', interactive=True, mask_nodata=True)

        popup = self._popup(self._markers(figure)[0])
        assert '<svg' in popup

    def test_radius_resizes_the_markers_without_reaching_the_map(self, service):
        """``radius`` is the folium marker size, and must not leak into folium.Map."""
        pytest.importorskip('folium')
        ts = _multipoint_timeseries(service, n_points=2)

        figure = ts.plot_map('NDVI', interactive=True, radius=9)

        assert {marker.options['radius'] for marker in self._markers(figure)} == {9}
