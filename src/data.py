import argparse
from functools import partial

import numpy as np
import torch
import torch.utils.data as Data

from features import feature_transform, parse_trace, parse_trace_trafficsliver


class SingleDomainDataset(Data.Dataset):
    def __init__(self, args: argparse.Namespace, flist: np.ndarray, labels: np.ndarray, is_train: bool):
        self.args = args
        self.flist = flist
        self.labels = labels
        self.feature_type = self.args.feature_type
        self.is_train = is_train
        self.feature_transform_func = partial(
            feature_transform,
            feature_type=self.feature_type,
            seq_length=self.args.seq_length,
            n_tam=self.args.n_tam,
            granularity=self.args.iat_bins,
            time_window=self.args.time_window,
        )

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        sample_path = self.flist[idx]
        trace = parse_trace(sample_path)
        x = self.feature_transform_func(trace)
        x = torch.from_numpy(x).float().reshape(1, *x.shape)
        y = torch.tensor([self.labels[idx]]).long()
        return x, y

    def __len__(self):
        return len(self.flist)

    @staticmethod
    def collate_fn(batch):
        x, y = zip(*batch)
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)
        return x, y


class TrafficSliverDataset(SingleDomainDataset):
    """Dataset variant for TrafficSliver traces split by blank lines."""

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        sample_path = self.flist[idx]
        traces = parse_trace_trafficsliver(sample_path)
        y = torch.tensor(self.labels[idx]).long()

        if not self.is_train:
            trace = traces[np.random.choice(len(traces))]
            x = self.feature_transform_func(trace)
            x = torch.from_numpy(x).float().reshape(1, *x.shape)
            return x, y.reshape(1)

        xs = []
        ys = []
        for trace in traces:
            x = self.feature_transform_func(trace)
            xs.append(torch.from_numpy(x).float())
            ys.append(y)
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)
