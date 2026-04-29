"""ME-DAQ Raw Pickle Viewer

Minimal viewer for raw pickles (especially ct_*). Pick a file, pick a column,
optionally bin, see raw plot + FFT.

Launch:  panel serve app.py --show --port 5009
"""

import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv
import hvplot.pandas  # noqa: F401  (registers .hvplot accessor)
from bokeh.models import DataRange1d, Range1d
from scipy.signal import find_peaks

pn.extension('tabulator', sizing_mode='stretch_width')
hv.extension('bokeh')

# VanillaTemplate hard-codes a fonts.googleapis.com link for Lato. Remove it
# so the page works on offline machines, and serve a local copy via the
# static-dirs mount instead (the launcher exposes ./static at /assets/).
_vt_resources = pn.template.VanillaTemplate._resources
pn.template.VanillaTemplate._resources = {
    **_vt_resources,
    'css': {k: v for k, v in _vt_resources.get('css', {}).items() if k != 'lato'},
}
pn.config.raw_css.append(
    "@font-face { font-family: 'Lato'; font-style: normal; font-weight: 400; "
    "font-display: swap; "
    "src: url('/assets/fonts/lato.woff2') format('woff2'); }"
)

css_path = Path(__file__).parent / 'static' / 'css' / 'analysis.css'
if css_path.exists():
    pn.config.raw_css.append(css_path.read_text())


HIDDEN_COLS = {'config_keys', 'config_values'}


def list_pickle_files(data_dir):
    """All *.pickle files in data_dir, sorted."""
    try:
        p = Path(data_dir)
        if p.exists():
            return sorted(f.name for f in p.glob('*.pickle'))
    except Exception:
        pass
    return []


def read_config_from_df(df):
    """Pull the {key: value} config dict embedded in raw pickles."""
    cfg = {}
    if 'config_keys' in df.columns and 'config_values' in df.columns:
        keys = df['config_keys'].dropna().tolist()
        vals = df['config_values'].dropna().tolist()
        for k, v in zip(keys, vals):
            cfg[str(k)] = v
    return cfg


def autorange_hook(plot, element):
    """Bokeh hook: y-axis refits when legend items toggle."""
    fig = plot.state
    if hasattr(fig, 'legend') and fig.legend:
        for legend in fig.legend:
            legend.click_policy = 'hide'
    if isinstance(fig.y_range, DataRange1d):
        fig.y_range.only_visible = True
    elif isinstance(fig.y_range, Range1d):
        fig.y_range = DataRange1d(only_visible=True, range_padding=0.05)


class RawViewer:

    def __init__(self):
        self._df = None
        self._sampling_rate = 100_000.0  # default; updated from pickle config
        self._current_file = None
        self._build_widgets()
        self._refresh_files()
        self.layout = self._build_layout()

    # ------------------------------------------------------------------
    # Widgets
    # ------------------------------------------------------------------

    def _build_widgets(self):
        self.data_dir_input = pn.widgets.TextInput(
            name='Data Directory', value='./',
            sizing_mode='stretch_width',
        )
        self.data_dir_input.param.watch(self._on_data_dir_change, 'value')

        self.file_select = pn.widgets.Select(
            name='File', options=[], sizing_mode='stretch_width',
        )
        self.refresh_button = pn.widgets.Button(
            name='Refresh', button_type='default', width=90,
            margin=(22, 5, 5, 5),
        )
        self.refresh_button.on_click(lambda e: self._refresh_files())

        self.load_button = pn.widgets.Button(
            name='Load', button_type='primary', sizing_mode='stretch_width',
        )
        self.load_button.on_click(self._on_load)

        self.column_select = pn.widgets.Select(
            name='Column', options=[], sizing_mode='stretch_width',
        )
        self.column_select.param.watch(self._on_view_change, 'value')

        self.binning_input = pn.widgets.IntInput(
            name='Mean binning (samples)', value=1, start=1, step=10,
            sizing_mode='stretch_width',
        )
        self.binning_input.param.watch(self._on_view_change, 'value')

        self.normalize_checkbox = pn.widgets.Checkbox(
            name='Normalize FFT (0-1)', value=False,
        )
        self.normalize_checkbox.param.watch(self._on_view_change, 'value')

        self.status_pane = pn.pane.Markdown(
            "**Status:** Ready", sizing_mode='stretch_width',
            styles={'word-wrap': 'break-word'},
        )

        self.info_pane = pn.pane.HTML(
            "", sizing_mode='stretch_width',
            styles={'font-size': '0.8em', 'background': '#f8f9fa',
                    'padding': '8px', 'border-radius': '4px'},
        )

        self.signal_plot_pane = pn.Column(
            pn.pane.Markdown(
                "*Load a file and pick a column to see a plot.*",
                styles={'color': '#6c757d', 'padding': '40px 20px',
                        'text-align': 'center'},
            ),
            sizing_mode='stretch_width',
        )

        self.fft_plot_pane = pn.Column(
            pn.pane.Markdown(
                "*FFT will appear here after a column is loaded.*",
                styles={'color': '#6c757d', 'padding': '40px 20px',
                        'text-align': 'center'},
            ),
            sizing_mode='stretch_width',
        )

        self.modes_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=['#', 'Frequency (Hz)', 'Amplitude']),
            disabled=True, show_index=False, layout='fit_data_stretch',
            height=330, sizing_mode='stretch_width',
            formatters={
                'Frequency (Hz)': {'type': 'plaintext'},
                'Amplitude': {'type': 'plaintext'},
            },
        )

        # --- Mode Evolution tab widgets ---
        self.evo_prefix_input = pn.widgets.TextInput(
            name='Filename prefix', value='ct_', placeholder='e.g. ct_2026-04-28',
            sizing_mode='stretch_width',
        )
        self.evo_binning_input = pn.widgets.IntInput(
            name='Mean binning (samples)', value=1, start=1, step=10, width=200,
        )
        self.evo_go_button = pn.widgets.Button(
            name='Go', button_type='primary', width=120,
            margin=(22, 5, 5, 5),
        )
        self.evo_go_button.on_click(self._on_evolution_run)

        self.evo_status_pane = pn.pane.Markdown(
            "*Enter a filename prefix and click Go.*",
            sizing_mode='stretch_width',
        )
        self.evo_plot_pane = pn.Column(
            pn.pane.Markdown(
                "*Mode evolution plot will appear here.*",
                styles={'color': '#6c757d', 'padding': '40px 20px',
                        'text-align': 'center'},
            ),
            sizing_mode='stretch_width',
        )
        self.evo_bar_pane = pn.Column(
            pn.pane.Markdown(
                "*Mode-frequency histogram will appear here.*",
                styles={'color': '#6c757d', 'padding': '40px 20px',
                        'text-align': 'center'},
            ),
            sizing_mode='stretch_width',
        )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self):
        data_card = pn.Card(
            pn.Column(
                self.data_dir_input,
                pn.Row(self.file_select, self.refresh_button),
                self.load_button,
            ),
            title='1. Data',
            collapsible=False,
            styles={'background': 'white', 'border': '1px solid #e9ecef'},
            sizing_mode='stretch_width',
        )

        view_card = pn.Card(
            pn.Column(
                self.column_select,
                self.binning_input,
                self.normalize_checkbox,
            ),
            title='2. View',
            collapsible=False,
            styles={'background': 'white', 'border': '1px solid #e9ecef'},
            sizing_mode='stretch_width',
        )

        status_card = pn.Card(
            pn.Column(self.status_pane, self.info_pane),
            title='Status',
            collapsible=False,
            styles={'background': 'white', 'border': '1px solid #e9ecef'},
            sizing_mode='stretch_width',
        )

        sidebar = pn.Column(
            data_card, view_card, status_card,
            sizing_mode='stretch_width',
        )

        single_file_tab = pn.Column(
            pn.pane.Markdown("### Signal"),
            self.signal_plot_pane,
            pn.pane.Markdown("### FFT"),
            self.fft_plot_pane,
            pn.pane.Markdown("### Dominant modes (top 10)"),
            self.modes_table,
            sizing_mode='stretch_width',
        )

        evolution_tab = pn.Column(
            pn.pane.Markdown(
                "### Mode Evolution Across Files\n"
                "Find every pickle in the data directory whose name starts with "
                "the prefix, FFT the **intensity** column of each (in alphabetical "
                "order), and plot the top-10 dominant modes per file."
            ),
            pn.Row(self.evo_prefix_input, self.evo_binning_input,
                   self.evo_go_button, align='end'),
            self.evo_status_pane,
            self.evo_plot_pane,
            pn.pane.Markdown(
                "#### Mode-frequency histogram (20 Hz buckets)\n"
                "How many files have a top-10 mode in each frequency bucket."
            ),
            self.evo_bar_pane,
            sizing_mode='stretch_width',
        )

        main_content = pn.Tabs(
            ('Single File', single_file_tab),
            ('Mode Evolution', evolution_tab),
            sizing_mode='stretch_width',
        )

        logo_path = Path(__file__).parent / 'static' / 'images' / 'ssrl-logo.png'
        header_content = pn.Row(
            pn.pane.PNG(str(logo_path), height=40, sizing_mode='fixed'),
            pn.pane.Markdown(
                "# Raw Pickle Viewer",
                styles={'margin-left': '15px', 'margin-top': '5px',
                        'color': '#000'},
            ),
            sizing_mode='stretch_width',
        )

        template = pn.template.VanillaTemplate(
            title='',
            header=[header_content],
            sidebar=[sidebar],
            main=[main_content],
            header_background='#f9f6ef',
            header_color='#000',
            sidebar_width=380,
            theme='default',
        )

        tab_title_js = pn.pane.HTML(
            "<script>document.title = 'Raw Pickle Viewer';</script>",
            width=0, height=0, sizing_mode='fixed',
        )
        template.main.insert(0, tab_title_js)
        return template

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_data_dir_change(self, event=None):
        self._refresh_files()

    def _refresh_files(self):
        files = list_pickle_files(self.data_dir_input.value)
        self.file_select.options = files
        if files and self.file_select.value not in files:
            self.file_select.value = files[0]

    def _on_load(self, event):
        filename = self.file_select.value
        if not filename:
            self.status_pane.object = "**Error:** Select a file first."
            return

        path = Path(self.data_dir_input.value) / filename
        if not path.exists():
            self.status_pane.object = f"**Error:** {path} does not exist."
            return

        self.status_pane.object = f"**Loading** `{filename}`..."
        try:
            df = pd.read_pickle(path)
        except Exception as e:
            self.status_pane.object = f"**Error loading file:** {e}"
            traceback.print_exc()
            return

        self._df = df
        self._current_file = filename

        cfg = read_config_from_df(df)
        try:
            self._sampling_rate = float(cfg.get('sampling_rate', 100_000))
        except (TypeError, ValueError):
            self._sampling_rate = 100_000.0

        plot_cols = [c for c in df.columns if c not in HIDDEN_COLS]
        self.column_select.options = plot_cols

        # Pick a sensible default column
        attrs = getattr(df, 'attrs', {}) or {}
        preferred = attrs.get('intensity_col')
        if preferred and preferred in plot_cols:
            self.column_select.value = preferred
        elif plot_cols:
            self.column_select.value = plot_cols[0]

        rows, cols = df.shape
        duration = rows / self._sampling_rate
        info_html = (
            f"<b>{filename}</b><br>"
            f"shape: {rows:,} × {cols} &nbsp;|&nbsp; "
            f"fs: {self._sampling_rate:,.0f} Hz &nbsp;|&nbsp; "
            f"duration: {duration:.3g} s"
        )
        if cfg:
            extras = []
            for k in ('frequency', 'binning', 'col_header', 'scan_status'):
                if k in cfg:
                    extras.append(f"{k}: {cfg[k]}")
            if extras:
                info_html += "<br><span style='color:#666'>" \
                             + " &nbsp;|&nbsp; ".join(extras) + "</span>"
        self.info_pane.object = info_html

        self.status_pane.object = f"**Loaded** `{filename}`"
        self._update_plots()

    def _on_view_change(self, event=None):
        if self._df is None:
            return
        self._update_plots()

    # ------------------------------------------------------------------
    # Mode Evolution callback
    # ------------------------------------------------------------------

    @staticmethod
    def _find_intensity_column(df):
        """Return the intensity column name, or None if not present."""
        attrs = getattr(df, 'attrs', {}) or {}
        col = attrs.get('intensity_col')
        if col and col in df.columns:
            return col
        for c in df.columns:
            if str(c).startswith('intensity_') and c not in HIDDEN_COLS:
                return c
        return None

    def _on_evolution_run(self, event):
        prefix = (self.evo_prefix_input.value or '').strip()
        if not prefix:
            self.evo_status_pane.object = "**Error:** Enter a filename prefix."
            return

        data_dir = self.data_dir_input.value
        all_files = list_pickle_files(data_dir)
        matched = [f for f in all_files if f.startswith(prefix)]
        if not matched:
            self.evo_status_pane.object = (
                f"**No matches** for prefix `{prefix}` in `{data_dir}`."
            )
            self.evo_plot_pane.objects = []
            return

        binning = max(int(self.evo_binning_input.value or 1), 1)
        self.evo_go_button.disabled = True
        try:
            rows = []
            skipped = []
            n_total = len(matched)
            for i, fname in enumerate(matched, start=1):
                self.evo_status_pane.object = (
                    f"**Processing {i}/{n_total}:** `{fname}`"
                )
                fpath = Path(data_dir) / fname
                try:
                    df = pd.read_pickle(fpath)
                except Exception as e:
                    skipped.append((fname, f"load error: {e}"))
                    continue

                col = self._find_intensity_column(df)
                if col is None:
                    skipped.append((fname, "no intensity column"))
                    continue

                cfg = read_config_from_df(df)
                try:
                    fs0 = float(cfg.get('sampling_rate', 100_000))
                except (TypeError, ValueError):
                    fs0 = 100_000.0

                values = pd.to_numeric(df[col], errors='coerce') \
                    .dropna().to_numpy(dtype=float)
                if binning > 1 and len(values) >= binning:
                    m = (len(values) // binning) * binning
                    values = values[:m].reshape(-1, binning).mean(axis=1)
                fs = fs0 / binning
                if len(values) < 4:
                    skipped.append((fname, "too few samples"))
                    continue

                n = len(values)
                half = n // 2
                fft_vals = np.abs(np.fft.fft(values))[1:half]
                freqs = np.fft.fftfreq(n, d=1.0 / fs)[1:half]
                modes = self._top_modes(freqs, fft_vals, n_top=10)
                modes = modes.assign(
                    file=fname, file_idx=i, intensity_col=col,
                )
                rows.append(modes)

            if not rows:
                msg = "**No usable files.**"
                if skipped:
                    msg += " Skipped: " + ", ".join(
                        f"`{f}` ({why})" for f, why in skipped[:5])
                self.evo_status_pane.object = msg
                self.evo_plot_pane.objects = []
                return

            evo_df = pd.concat(rows, ignore_index=True)
            self._update_evolution_plot(evo_df, n_files=len(rows),
                                        n_skipped=len(skipped))
        except Exception as e:
            self.evo_status_pane.object = f"**Error:** {e}"
            traceback.print_exc()
        finally:
            self.evo_go_button.disabled = False

    def _update_evolution_plot(self, evo_df, n_files, n_skipped):
        """Render the mode-evolution scatter plot from the long DataFrame."""
        # Scale amplitude to a marker-size range so weak modes stay visible.
        amps = evo_df['Amplitude'].to_numpy(dtype=float)
        a_min, a_max = float(np.min(amps)), float(np.max(amps))
        if a_max > a_min:
            sizes = 6 + 22 * (amps - a_min) / (a_max - a_min)
        else:
            sizes = np.full_like(amps, 10.0)
        evo_df = evo_df.assign(_size=sizes)

        try:
            scatter = evo_df.hvplot.scatter(
                x='file_idx', y='Frequency (Hz)',
                size='_size', c='#',
                cmap='Viridis', clim=(1, 10),
                hover_cols=['file', '#', 'Amplitude'],
                title='Top-10 Modes per File (size = amplitude, color = rank)',
                xlabel='File index', ylabel='Frequency (Hz)',
                height=500, shared_axes=False,
            ).opts(width=1000, shared_axes=False, hooks=[autorange_hook],
                   colorbar=True, tools=['hover', 'box_zoom', 'wheel_zoom',
                                         'reset', 'pan'])
            self.evo_plot_pane.objects = [pn.pane.HoloViews(
                scatter, sizing_mode='stretch_width', height=525,
                linked_axes=False,
            )]
        except Exception as e:
            self.evo_plot_pane.objects = [pn.pane.Markdown(
                f"**Plot error:** {e}",
                styles={'color': '#dc3545', 'padding': '20px'},
            )]
            return

        # --- 20-Hz bucket histogram: count of distinct files per bucket ---
        try:
            bucket = (np.round(evo_df['Frequency (Hz)'] / 20.0)
                      * 20.0).astype(int)
            hist_df = (evo_df.assign(bucket=bucket)
                       .groupby('bucket')['file'].nunique()
                       .reset_index(name='n_files')
                       .sort_values('bucket'))
            hist_df.rename(columns={'bucket': 'Frequency bucket (Hz)'},
                           inplace=True)
            bar = hist_df.hvplot.bar(
                x='Frequency bucket (Hz)', y='n_files',
                title='Files with a top-10 mode in each 20 Hz bucket',
                xlabel='Frequency bucket (Hz, 20 Hz wide)',
                ylabel='# files',
                height=350, shared_axes=False,
            ).opts(width=1000, shared_axes=False, color='#0072B5',
                   tools=['hover', 'box_zoom', 'wheel_zoom', 'reset', 'pan'],
                   xrotation=45)
            self.evo_bar_pane.objects = [pn.pane.HoloViews(
                bar, sizing_mode='stretch_width', height=375,
                linked_axes=False,
            )]
        except Exception as e:
            self.evo_bar_pane.objects = [pn.pane.Markdown(
                f"**Bar chart error:** {e}",
                styles={'color': '#dc3545', 'padding': '20px'},
            )]

        skipped_str = f", skipped {n_skipped}" if n_skipped else ""
        self.evo_status_pane.object = (
            f"**Done.** Processed {n_files} files{skipped_str}. "
            f"Each file contributes up to 10 dominant modes."
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _binned_signal(self, col):
        """Return (time, values, effective_sampling_rate) for the column,
        applying mean binning if requested."""
        binning = max(int(self.binning_input.value or 1), 1)
        series = pd.to_numeric(self._df[col], errors='coerce').dropna()
        values = series.to_numpy(dtype=float)

        if binning > 1 and len(values) >= binning:
            n = (len(values) // binning) * binning
            values = values[:n].reshape(-1, binning).mean(axis=1)

        fs = self._sampling_rate / binning
        t = np.arange(len(values)) / fs
        return t, values, fs

    def _update_plots(self):
        col = self.column_select.value
        if not col or self._df is None or col not in self._df.columns:
            return

        try:
            t, y, fs = self._binned_signal(col)
        except Exception as e:
            self.signal_plot_pane.objects = [pn.pane.Markdown(
                f"**Plot error:** {e}",
                styles={'color': '#dc3545', 'padding': '20px'},
            )]
            return

        if len(y) == 0:
            self.signal_plot_pane.objects = [pn.pane.Markdown(
                "*Selected column has no numeric data.*",
                styles={'color': '#6c757d', 'padding': '40px 20px',
                        'text-align': 'center'},
            )]
            self.fft_plot_pane.objects = []
            return

        # --- Signal plot ---
        sig_df = pd.DataFrame({'time (s)': t, col: y})
        binning = max(int(self.binning_input.value or 1), 1)
        bin_str = f" (binning {binning}, fs={fs:,.1f} Hz)" if binning > 1 \
                  else f" (fs={fs:,.0f} Hz)"
        try:
            sig_plot = sig_df.hvplot(
                x='time (s)', y=col, title=f'{col}{bin_str}',
                xlabel='time (s)', ylabel=col,
                height=350, shared_axes=False,
            ).opts(width=900, shared_axes=False, hooks=[autorange_hook])
            self.signal_plot_pane.objects = [pn.pane.HoloViews(
                sig_plot, sizing_mode='stretch_width', height=375,
                linked_axes=False,
            )]
        except Exception as e:
            self.signal_plot_pane.objects = [pn.pane.Markdown(
                f"**Plot error:** {e}",
                styles={'color': '#dc3545', 'padding': '20px'},
            )]

        # --- FFT plot ---
        try:
            n = len(y)
            half = n // 2
            if half < 2:
                self.fft_plot_pane.objects = [pn.pane.Markdown(
                    "*Not enough samples for FFT.*",
                    styles={'color': '#6c757d', 'padding': '20px',
                            'text-align': 'center'},
                )]
                self.modes_table.value = pd.DataFrame(
                    columns=['#', 'Frequency (Hz)', 'Amplitude'])
                return
            fft_vals = np.abs(np.fft.fft(y))[1:half]
            freqs = np.fft.fftfreq(n, d=1.0 / fs)[1:half]
            if self.normalize_checkbox.value and fft_vals.max() != 0:
                fft_vals = fft_vals / fft_vals.max()
            fft_df = pd.DataFrame({
                'Frequency (Hz)': freqs, 'Amplitude': fft_vals,
            })
            fft_plot = fft_df.hvplot(
                x='Frequency (Hz)', y='Amplitude',
                title=f'FFT — {col}',
                height=350, shared_axes=False,
            ).opts(width=900, shared_axes=False, hooks=[autorange_hook])
            self.fft_plot_pane.objects = [pn.pane.HoloViews(
                fft_plot, sizing_mode='stretch_width', height=375,
                linked_axes=False,
            )]
            self.modes_table.value = self._top_modes(freqs, fft_vals, n_top=10)
        except Exception as e:
            self.fft_plot_pane.objects = [pn.pane.Markdown(
                f"**FFT error:** {e}",
                styles={'color': '#dc3545', 'padding': '20px'},
            )]
            self.modes_table.value = pd.DataFrame(
                columns=['#', 'Frequency (Hz)', 'Amplitude'])

    @staticmethod
    def _top_modes(freqs, amps, n_top=10):
        """Top-N local maxima of the FFT, ranked by amplitude.

        Falls back to top-N raw bins if find_peaks returns fewer than N.
        """
        peaks, _ = find_peaks(amps)
        if len(peaks) >= n_top:
            order = np.argsort(amps[peaks])[::-1][:n_top]
            idx = peaks[order]
        else:
            idx = np.argsort(amps)[::-1][:n_top]
        idx = idx[np.argsort(amps[idx])[::-1]]
        return pd.DataFrame({
            '#': np.arange(1, len(idx) + 1),
            'Frequency (Hz)': np.round(freqs[idx], 4),
            'Amplitude': amps[idx],
        })


app = RawViewer()
app.layout.servable()
