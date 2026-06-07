from typing import Union

import numpy as np
import torch
from ignite.metrics import Metric

from utils import PR_THRES_NUM


class WFMetric(Metric):
    def __init__(self, nmc: int, device: Union[str, torch.device] = torch.device("cpu")):
        self.nmc = nmc
        self._p = 0
        self._n = 0
        self._tp = 0
        self._fp = 0
        super(WFMetric, self).__init__(device=device)

    def reset(self):
        self._p = 0
        self._n = 0
        self._tp = 0
        self._fp = 0
        super(WFMetric, self).reset()

    def update(self, output: tuple, logits: bool = True):
        """
        :param output: tuple of (y_pred, y)
        :param logits: if True, y_pred is logits, else y_pred is labels
        """
        y_pred, y = output[0].detach(), output[1].detach()
        indices = torch.argmax(y_pred, dim=1) if logits else y_pred
        idx_p = y < self.nmc
        idx_n = y == self.nmc

        self._p += torch.sum(idx_p).item()
        self._n += torch.sum(idx_n).item()

        self._tp += (indices[idx_p] == y[idx_p]).sum().item()
        self._fp += (indices[idx_n] != y[idx_n]).sum().item()

    def compute(self):
        """
        :return: tp, fp, p, n
        """
        return self._tp, self._fp, self._p, self._n


class WFPRCurve(Metric):
    def __init__(self, nmc: int, device: Union[str, torch.device] = torch.device("cpu")):
        self.nmc = nmc  # number of monitored classes
        self.thres = np.linspace(0.01, 0.99, PR_THRES_NUM)  # threshold range

        self._tps = np.zeros(len(self.thres))
        self._fps = np.zeros(len(self.thres))
        self._wps = np.zeros(len(self.thres))
        self._fns = np.zeros(len(self.thres))
        self._tns = np.zeros(len(self.thres))

        super(WFPRCurve, self).__init__(device=device)

    def reset(self):
        self._tps = np.zeros(len(self.thres))
        self._fps = np.zeros(len(self.thres))
        self._wps = np.zeros(len(self.thres))
        self._fns = np.zeros(len(self.thres))
        self._tns = np.zeros(len(self.thres))
        super(WFPRCurve, self).reset()

    def update(self, output: tuple, logits: bool = True):
        """
        :param output: tuple of (y_pred, y)
        :param logits: if True, y_pred is logits (before softmax)
        """
        y_pred, y = output[0].detach(), output[1].detach()

        if logits:
            y_pred = torch.nn.functional.softmax(y_pred, dim=1)

        for i, th in enumerate(self.thres):
            confs, indices = torch.max(y_pred, dim=1)
            idx_p = y < self.nmc
            idx_n = y == self.nmc

            idx_pred_as_p = (confs >= th) & (indices < self.nmc)
            idx_pred_as_n = (confs < th) | (indices == self.nmc)

            idx_pred_eq_y = indices == y

            tp = (idx_p & idx_pred_as_p & idx_pred_eq_y).sum().item()  # monitored and predicted as monitored
            fp = (idx_n & idx_pred_as_p).sum().item()  # unmonitored but predicted as monitored
            wp = (idx_p & idx_pred_as_p & (~idx_pred_eq_y)).sum().item()  # monitored but predicted as another monitored
            fn = (idx_p & idx_pred_as_n).sum().item()  # monitored but predicted as unmonitored
            tn = (idx_n & idx_pred_as_n).sum().item()  # unmonitored and predicted as unmonitored

            self._tps[i] += tp
            self._fps[i] += fp
            self._wps[i] += wp
            self._fns[i] += fn
            self._tns[i] += tn

            assert tp + wp + fn == idx_p.sum(), "TP + WP + FN != P"
            assert fp + tn == idx_n.sum(), "FP + TN != N"
            assert idx_p.sum() + idx_n.sum() == len(y), "P + N != Total"

    def compute(self):
        """
        :return: tp, fp, wp, fn, tn
        """
        # stack to (PR_THRES_NUM, 5) shape
        stats = np.stack((self._tps, self._fps, self._wps, self._fns, self._tns), axis=1)
        return stats
