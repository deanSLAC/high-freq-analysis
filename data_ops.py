"""Pure data operations on raw ME-DAQ pickles.

No UI, no widget state — just functions that take data and return data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


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


def find_intensity_column(df):
    """Return the intensity column name, or None if not present."""
    attrs = getattr(df, 'attrs', {}) or {}
    col = attrs.get('intensity_col')
    if col and col in df.columns:
        return col
    for c in df.columns:
        if str(c).startswith('intensity_') and c not in HIDDEN_COLS:
            return c
    return None


def top_modes(freqs, amps, n_top=10):
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
