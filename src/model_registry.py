from __future__ import annotations

import argparse

from features import normalize_feature_type

MODEL_TYPES = ['ares', 'df', 'inception', 'rf', 'tmwf', 'varcnn', 'wfcat']


def normalize_model_name(model: str) -> str:
    if model not in MODEL_TYPES:
        raise ValueError(f"Unsupported model: {model}")
    return model


def model_choices() -> list[str]:
    return MODEL_TYPES


def build_model(args: argparse.Namespace, num_classes: int):
    model_name = normalize_model_name(args.model)
    feature_type = normalize_feature_type(args.feature_type)

    if model_name == 'wfcat':
        from models.wfcat import WFCAT
        _require(feature_type in {'iat', 'iat_linear', 'tam', 'tam_plus'},
                 'WFCAT supports iat, iat_linear, tam, or tam_plus features.')
        iat_bins = 1 if feature_type in {'tam', 'tam_plus'} else args.iat_bins
        return WFCAT(num_classes=num_classes, iat_bins=iat_bins,
                     num_kernels_1d=args.num_kernels, num_kernels_2d=args.num_kernels)

    if model_name == 'rf':
        from models.rf import RFNet
        _require(feature_type in {'tam', 'tam_plus'}, 'RF supports tam or tam_plus features.')
        return RFNet(num_classes=num_classes, in_channel=1)

    if model_name == 'df':
        from models.df import DFNet
        in_channels = _one_dimensional_channels(feature_type)
        return DFNet(length=args.seq_length, num_classes=num_classes, in_channels=in_channels)

    if model_name == 'inception':
        from models.inception import InceptionNet
        _require(feature_type in {'iat', 'iat_linear', 'tam', 'tam_plus'},
                 'InceptionNet supports iat, iat_linear, tam, or tam_plus features.')
        in_channels = 1 if feature_type in {'tam', 'tam_plus'} else args.iat_bins
        return InceptionNet(length=args.seq_length, num_classes=num_classes,
                            in_channels=in_channels, num_kernels=args.num_kernels)

    if model_name == 'tmwf':
        from models.tmwf import TMWF
        _require(feature_type in {'direction', 'directional_timing'},
                 'TMWF supports direction or directional_timing features.')
        _require(args.seq_length == 30720, 'TMWF requires --seq-length 30720.')
        return TMWF(num_classes=num_classes)

    if model_name == 'ares':
        from models.ares import ARES
        _require(feature_type in {'direction', 'directional_timing'},
                 'ARES supports direction or directional_timing features.')
        return ARES(num_classes=num_classes)

    if model_name == 'varcnn':
        from models.varcnn import VarCNN
        _require(feature_type == 'directional_timing', 'VarCNN supports directional_timing features.')
        return VarCNN(num_classes=num_classes)

    raise NotImplementedError(f"Model {args.model} is not implemented.")


def _one_dimensional_channels(feature_type: str) -> int:
    _require(feature_type in {'direction', 'directional_timing', 'burst'},
             'DFNet supports direction, directional_timing, or burst features.')
    return 1


def _require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)
