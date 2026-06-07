from __future__ import annotations

import os
from typing import List

import numpy as np

FEATURE_TYPES = [
    'burst',
    'direction',
    'directional_timing',
    'iat',
    'iat_linear',
    'patch',
    'tam',
    'tam_plus',
]


def normalize_feature_type(feature_type: str) -> str:
    if feature_type not in FEATURE_TYPES:
        raise ValueError(f"Unsupported feature type: {feature_type}")
    return feature_type


def parse_trace(path: str | os.PathLike, sanity_check: bool = False) -> np.ndarray:
    import pandas as pd

    trace = pd.read_csv(path, delimiter='\t', header=None)
    trace = np.array(trace)
    trace = trace[trace[:, 0].argsort()]

    if sanity_check:
        cut_off_threshold = 15
        start, end = 0, len(trace)
        ipt_burst = np.diff(trace[:, 0])
        ipt_outlier_inds = np.where(ipt_burst > cut_off_threshold)[0]

        if len(ipt_outlier_inds) > 0:
            outlier_ind_first = ipt_outlier_inds[0]
            if outlier_ind_first < 50:
                start = outlier_ind_first + 1
            outlier_ind_last = ipt_outlier_inds[-1]
            if outlier_ind_last > 50:
                end = outlier_ind_last + 1
        trace = trace[start:end].copy()

        start = -1
        for _, size in trace:
            start += 1
            if size > 0:
                break

        trace = trace[start:].copy()
        trace[:, 0] -= trace[0, 0]
        assert trace[0, 0] == 0
    return trace


def parse_trace_trafficsliver(path: str | os.PathLike) -> List[np.ndarray]:
    import pandas as pd

    df = pd.read_csv(path, delimiter='\t', header=None, skip_blank_lines=False)
    blank_line_indices = df[df.isna().all(axis=1)].index

    arrays = []
    previous_index = 0
    for index in blank_line_indices:
        if previous_index != index:
            arr = np.array(df.iloc[previous_index:index].dropna().reset_index(drop=True))
            arrays.append(arr)
        previous_index = index + 1

    if previous_index < len(df):
        arrays.append(np.array(df.iloc[previous_index:].dropna().reset_index(drop=True)))
    return arrays


def tam(sample: np.ndarray, time_window: float, max_load_time: float, pad_length: int | None = None) -> np.ndarray:
    cut_off_time = min(max_load_time, float(sample[-1, 0]))
    num_bins = int(cut_off_time / time_window) + 1
    bins = np.linspace(0, num_bins * time_window, num_bins).tolist() + [np.inf]

    outgoing = sample[np.sign(sample[:, 1]) > 0]
    incoming = sample[np.sign(sample[:, 1]) < 0]
    cnt_outgoing, _ = np.histogram(outgoing[:, 0], bins=bins)
    cnt_incoming, _ = np.histogram(incoming[:, 0], bins=bins)
    feat = np.stack((cnt_outgoing, cnt_incoming), axis=1)

    assert feat.flatten().sum() == len(sample), (
        f"Sum of feature ({feat.flatten().sum()}) is not equal to trace length ({len(sample)})."
    )
    return _pad_or_trim(feat, pad_length)


def iat_histogram(sample: np.ndarray, time_window: float, max_load_time: float,
                  granularity: int, pad_length: int | None = None) -> np.ndarray:
    cut_off_time = min(max_load_time, float(sample[-1, 0]))
    num_bins = int(cut_off_time / time_window) + 1
    bins = np.linspace(0, num_bins * time_window, num_bins).tolist() + [np.inf]

    mask_outgoing = sample[:, 1] > 0
    iats = np.diff(sample[:, 0], prepend=0)
    iat_bins = np.logspace(-6, -1, granularity).tolist() + [np.inf]
    iat_bins[0] = 0

    feat_outgoing, _, _ = np.histogram2d(sample[:, 0][mask_outgoing], iats[mask_outgoing], bins=(bins, iat_bins))
    feat_incoming, _, _ = np.histogram2d(sample[:, 0][~mask_outgoing], iats[~mask_outgoing], bins=(bins, iat_bins))

    feat = np.stack((feat_outgoing, feat_incoming), axis=-1)
    feat = feat.reshape(-1, granularity * 2)

    assert feat.flatten().sum() == len(sample), (
        f"Sum of feature ({feat.flatten().sum()}) is not equal to trace length ({len(sample)})."
    )
    return _pad_or_trim(feat, pad_length)


def iat_histogram_linear(sample: np.ndarray, time_window: float, max_load_time: float,
                         granularity: int, pad_length: int | None = None) -> np.ndarray:
    cut_off_time = min(max_load_time, float(sample[-1, 0]))
    num_bins = int(cut_off_time / time_window) + 1
    bins = np.linspace(0, num_bins * time_window, num_bins).tolist() + [np.inf]

    iats = np.diff(sample[:, 0], prepend=0)
    iat_bins = np.linspace(0, 0.1, granularity).tolist() + [np.inf]
    mask_outgoing = sample[:, 1] > 0

    feat_outgoing, _, _ = np.histogram2d(sample[:, 0][mask_outgoing], iats[mask_outgoing], bins=(bins, iat_bins))
    feat_incoming, _, _ = np.histogram2d(sample[:, 0][~mask_outgoing], iats[~mask_outgoing], bins=(bins, iat_bins))
    feat = np.stack((feat_outgoing, feat_incoming), axis=-1)
    feat = feat.reshape(-1, granularity * 2)

    assert feat.flatten().sum() == len(sample), (
        f"Sum of feature ({feat.flatten().sum()}) is not equal to trace length ({len(sample)})."
    )
    return _pad_or_trim(feat, pad_length)


def feature_transform(sample: np.ndarray, feature_type: str, seq_length: int, n_tam: int = 1,
                      granularity: int = 9, time_window: float = 0.044) -> np.ndarray:
    feature_type = normalize_feature_type(feature_type)

    if feature_type == 'direction':
        feat = np.sign(sample[:, 1])
    elif feature_type == 'directional_timing':
        feat = sample[:, 0] * np.sign(sample[:, 1])
    elif feature_type == 'tam':
        feat = tam(sample, time_window=0.044, max_load_time=80, pad_length=None)
    elif feature_type == 'tam_plus':
        feats = []
        pad_length = None
        cur_window = 0.044
        for i in range(n_tam):
            cur_window = cur_window * (1 + 0.5 * i)
            feat_once = tam(sample, cur_window, max_load_time=80, pad_length=pad_length)
            pad_length = len(feat_once)
            feats.append(feat_once)
        feat = np.concatenate(feats, axis=1)
    elif feature_type == 'iat':
        feat = iat_histogram(sample, time_window=time_window, max_load_time=80,
                             granularity=granularity, pad_length=None)
    elif feature_type == 'iat_linear':
        feat = iat_histogram_linear(sample, time_window=0.044, max_load_time=80,
                                    granularity=granularity, pad_length=None)
    elif feature_type == 'patch':
        feat = _patch_feature(sample)
    elif feature_type == 'burst':
        feat = _burst_feature(sample)
    else:
        raise NotImplementedError(f"Feature type {feature_type} is not implemented.")

    if len(feat.shape) == 1:
        feat = feat[:, np.newaxis]
    feat = _pad_or_trim(feat, seq_length)
    return np.transpose(feat, (1, 0))


def feature_transform_from_path(path: str, feature_type: str, seq_length: int, n_tam: int = 1,
                                granularity: int = 9, time_window: float = 0.044,
                                sanity_check: bool = True) -> np.ndarray:
    sample = parse_trace(path, sanity_check)
    return feature_transform(sample, feature_type, seq_length, n_tam, granularity, time_window)


def _pad_or_trim(feat: np.ndarray, length: int | None) -> np.ndarray:
    if length is None:
        return feat
    if len(feat) < length:
        pad = np.zeros((length - len(feat), feat.shape[1]))
        return np.concatenate((feat, pad))
    return feat[:length, :]


def _patch_feature(sample: np.ndarray) -> np.ndarray:
    patch_size = 20
    dim = 5
    direction = np.sign(sample[:, 1])
    time = sample[:, 0]
    iat = np.diff(time, prepend=0)

    if len(sample) % patch_size != 0:
        pad = np.zeros((patch_size - len(sample) % patch_size))
        direction = np.concatenate((direction, pad))
        iat = np.concatenate((iat, pad))

    n_patches = len(iat) // patch_size
    is_padding = np.array([0] * len(sample) + [1] * (len(iat) - len(sample))).astype(bool)
    is_outgoing = direction > 0
    time_bins = [0] + np.logspace(-5, -1, dim - 1).tolist() + [np.inf]
    feat_mask = np.zeros((len(time_bins) - 1, len(iat))).astype(bool)

    for i in range(len(time_bins) - 1):
        feat_mask[i] = (iat >= time_bins[i]) & (iat < time_bins[i + 1])

    feat_mask = feat_mask.reshape(dim, n_patches, patch_size)
    is_outgoing = is_outgoing.reshape(n_patches, patch_size)
    is_padding = is_padding.reshape(n_patches, patch_size)

    feat = np.zeros((n_patches, dim * 2))
    for i in range(len(feat_mask)):
        feat[:, 2 * i] = (feat_mask[i] * is_outgoing * (~is_padding)).sum(axis=1)
        feat[:, 2 * i + 1] = (feat_mask[i] * ~is_outgoing * (~is_padding)).sum(axis=1)

    assert feat.sum() == len(sample), f"Sum of patch feature ({feat.sum()}) != trace length ({len(sample)})."
    return feat


def _burst_feature(sample: np.ndarray) -> np.ndarray:
    sample = sample[:, 1]
    mask = np.where(np.sign(sample[:-1]) != np.sign(sample[1:]))[0] + 1
    mask = np.concatenate((mask, [len(sample)]))
    feat = np.diff(mask, prepend=0)
    assert sum(feat) == len(sample), f"Sum of burst feature ({sum(feat)}) != trace length ({len(sample)})."
    return feat
