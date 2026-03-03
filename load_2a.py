import os
import numpy as np
import mne

# Motor imagery codes (BCI Competition IV 2a) in T session
MI_CODES = {"left": "769", "right": "770", "feet": "771", "tongue": "772"}

# Standard BCICIV2a EEG channel names (22)
BCICIV2A_EEG22 = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz"
]


def _find_event_value_containing(event_id: dict, code_str: str):
    """Return the mapped integer for the annotation key that contains `code_str`."""
    for desc, mapped_int in event_id.items():
        if code_str in str(desc):
            return mapped_int
    return None


def _drop_eog_if_present(raw: mne.io.BaseRaw):
    """Drop EOG channels using common names found in different dataset conversions."""
    eog_candidates = ["EOG-left", "EOG-central", "EOG-right", "EOG1", "EOG2", "EOG3"]
    to_drop = [ch for ch in eog_candidates if ch in raw.ch_names]
    if to_drop:
        raw.drop_channels(to_drop)


def _fix_channel_names_if_generic(raw: mne.io.BaseRaw):
    """
    Some GDF mirrors have generic/duplicated names like 'EEG', 'EEG 1', etc.
    After picking EEG, if nchan == 22, rename to standard BCICIV2a list.
    """
    if raw.info["nchan"] != 22:
        return

    generic = 0
    for ch in raw.ch_names:
        s = str(ch).strip().upper()
        if s.startswith("EEG") or s in ("E", "EE"):
            generic += 1

    if generic >= 10 or len(set(raw.ch_names)) != len(raw.ch_names):
        mapping = {raw.ch_names[i]: BCICIV2A_EEG22[i] for i in range(22)}
        raw.rename_channels(mapping)


def load_subject_2a(
    data_dir: str,
    subj: int,
    session: str = "T",
    tmin: float = 0.5,
    tmax: float = 2.5,
    l_freq: float = 4.0,
    h_freq: float = 40.0,
    notch: float = 50.0,
):
    """
    T-only loader (keeps original function name/signature).
    NOTE: Evaluation session E is not supported because labels are missing in your mirror.

    Returns:
        X: (n_trials, n_channels, n_times)
        y: (n_trials,) labels 0..3 for [left, right, feet, tongue]
        ch_names: list[str]
        sfreq: float
        event_id: dict (for debugging)
    """
    session = session.upper()
    if session != "T":
        raise ValueError(
            "This project version is T-session only. "
            "Your E-session .mat has empty y/trial, so E metrics are not possible."
        )

    gdf_path = os.path.join(data_dir, f"A{subj:02d}T.gdf")
    if not os.path.exists(gdf_path):
        raise FileNotFoundError(f"Missing file: {gdf_path}")

    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)

    # Clean channel names
    raw.rename_channels(lambda x: str(x).strip())

    # Keep only EEG (drop EOG explicitly + pick eeg)
    _drop_eog_if_present(raw)
    picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude="bads")
    raw.pick(picks_eeg)

    # Fix channel names if needed
    _fix_channel_names_if_generic(raw)

    # Filtering
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    if notch is not None:
        raw.notch_filter(freqs=[notch], verbose=False)

    # Events from annotations
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    # Map MI codes (769-772)
    mi_event_id = {}
    for name, code_str in MI_CODES.items():
        v = _find_event_value_containing(event_id, code_str)
        if v is not None:
            mi_event_id[name] = v

    if len(mi_event_id) == 0:
        raise ValueError(
            "No MI codes (769-772) found in T session.\n"
            f"Available annotation keys (first 30): {list(event_id.keys())[:30]}"
        )

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=mi_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )

    # Enforce label order: 0..3 -> left/right/feet/tongue
    order = ["left", "right", "feet", "tongue"]
    inv_map = {mi_event_id[k]: i for i, k in enumerate(order) if k in mi_event_id}
    y_raw = epochs.events[:, 2]
    y = np.array([inv_map[v] for v in y_raw], dtype=int)

    X = epochs.get_data()
    return X, y, epochs.ch_names, float(raw.info["sfreq"]), event_id
