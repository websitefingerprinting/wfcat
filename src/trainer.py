from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from data import SingleDomainDataset, TrafficSliverDataset
from features import normalize_feature_type
from logger import init_logger
from metrics import WFMetric, WFPRCurve
from model_registry import build_model, normalize_model_name
from utils import (PR_THRES_NUM, get_flist_label_multi_domain, get_grad_norm,
                         increment_path, select_fast_slow)


class WFExperiment:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.args.model = normalize_model_name(args.model)
        self.args.feature_type = normalize_feature_type(args.feature_type)

        self.logger = init_logger(self.__class__.__name__, verbose=args.verbose)
        self.device = self._acquire_device()

        assert self.args.page_per_class > 0, 'Page per class must be positive'

        self.args.unmon_inst = args.unmon_inst if args.open_world else 0
        self.nmc, self.flist, self.labels = get_flist_label_multi_domain(
            self.args.data_path,
            mon_cls=self.args.mon_classes,
            mon_inst=self.args.mon_inst,
            unmon_inst=self.args.unmon_inst,
            page_per_class=self.args.page_per_class,
            suffix=self.args.suffix,
        )

        self.nc = len(np.unique(self.labels))
        assert self.nc == self.nmc + int(self.args.open_world), 'Unexpected number of classes'

        self.amp_mode = 'amp' if (not self.args.not_amp) and self.device.type != 'cpu' else None
        self.checkpoint_path = None
        if not self.args.nosave and self.args.mode == 'train':
            self.checkpoint_path = increment_path(
                Path(self.args.model_path) / f'{self.args.model}_{self.args.feature_type}',
                sep='_', exist_ok=self.args.exist_ok, mkdir=True,
            )

        self.logger.info(
            'Data: %d traces | Classes: %d monitored + %d unmonitored | Model: %s | Feature: %s',
            len(self.flist), self.nmc, self.nc - self.nmc, self.args.model, self.args.feature_type,
        )

    def _acquire_device(self) -> torch.device:
        use_gpu = self.args.use_gpu and torch.cuda.is_available()
        if not use_gpu:
            self.logger.info('Use CPU')
            return torch.device('cpu')

        if self.args.use_multi_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.args.devices
        elif 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args.gpu)
        self.logger.info('Use GPU: cuda:0')
        return torch.device('cuda:0')

    def _build_model(self) -> nn.Module:
        model = build_model(self.args, num_classes=self.nc)
        if self.args.use_multi_gpu and torch.cuda.device_count() > 1:
            self.logger.info('Using %d GPUs for training', torch.cuda.device_count())
            model = nn.DataParallel(model)
        return model.to(self.device)

    def _get_data(self, flist: np.ndarray, labels: np.ndarray, is_train=True) -> Tuple[object, DataLoader]:
        dataset_cls = TrafficSliverDataset if self.args.defense == 'trafficsliver' else SingleDomainDataset
        dataset = dataset_cls(self.args, flist, labels, is_train)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=is_train,
                            num_workers=self.args.workers, collate_fn=dataset_cls.collate_fn)
        return dataset, loader

    def run(self, one_fold_only: bool = False):
        if self.args.bandwidth_split != 'none':
            return self._run_bandwidth_split()

        res = np.zeros((PR_THRES_NUM, 5)) if self.args.open_world else np.zeros(4)
        splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=self.args.seed)

        for fold, (train_index, test_index) in enumerate(splitter.split(self.flist, self.labels), start=1):
            if one_fold_only and fold > 1:
                break

            train_list_all, train_labels_all = self.flist[train_index], self.labels[train_index]
            train_list, val_list, train_labels, val_labels = train_test_split(
                train_list_all, train_labels_all, test_size=0.10,
                random_state=self.args.seed, stratify=train_labels_all,
            )
            train_list, train_labels = self._subsample_train_set(train_list, train_labels)
            test_list, test_labels = self.flist[test_index], self.labels[test_index]
            res += self.train(fold, train_list, train_labels, val_list, val_labels, test_list, test_labels)
            self.logger.info('-' * 10)

        self._print_result(res)

    def _run_bandwidth_split(self):
        res = np.zeros((PR_THRES_NUM, 5)) if self.args.open_world else np.zeros(4)
        train_list_all, train_labels_all, test_list, test_labels = select_fast_slow(
            self.flist, self.labels, train_mode=self.args.bandwidth_split, ratio=self.args.bandwidth_ratio,
        )
        train_list, val_list, train_labels, val_labels = train_test_split(
            train_list_all, train_labels_all, test_size=0.10,
            random_state=self.args.seed, stratify=train_labels_all,
        )
        train_list, train_labels = self._subsample_train_set(train_list, train_labels)
        res += self.train(1, train_list, train_labels, val_list, val_labels, test_list, test_labels)
        self._print_result(res)

    def _subsample_train_set(self, train_list: np.ndarray, train_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        selected_files, selected_labels = [], []

        if self.args.mon_inst_train > 0:
            self.logger.info('Use %d monitored instances per class for training', self.args.mon_inst_train)
            for label in range(self.nmc):
                idx = np.where(train_labels == label)[0]
                total_num = min(len(idx), self.args.mon_inst_train)
                idx = np.random.choice(idx, total_num, replace=False)
                selected_files.extend(train_list[idx])
                selected_labels.extend(train_labels[idx])
        else:
            selected_files.extend(train_list[train_labels < self.nmc])
            selected_labels.extend(train_labels[train_labels < self.nmc])

        if self.args.open_world:
            if self.args.unmon_inst_train > 0:
                self.logger.info('Use %d unmonitored instances for training', self.args.unmon_inst_train)
                idx = np.where(train_labels == self.nmc)[0]
                total_num = min(len(idx), self.args.unmon_inst_train)
                idx = np.random.choice(idx, total_num, replace=False)
                selected_files.extend(train_list[idx])
                selected_labels.extend(train_labels[idx])
            else:
                selected_files.extend(train_list[train_labels == self.nmc])
                selected_labels.extend(train_labels[train_labels == self.nmc])

        return np.array(selected_files), np.array(selected_labels)

    @staticmethod
    def train_step(engine: Engine, batch: Tuple, model: nn.Module, optimizer: torch.optim.Optimizer,
                   criterion: nn.Module, device: torch.device, scaler: torch.cuda.amp.GradScaler,
                   clip_value: float, use_amp: bool):
        model.train()
        optimizer.zero_grad()
        x, y = batch
        x, y = x.to(device), y.to(device)

        if use_amp:
            with torch.cuda.amp.autocast():
                y_pred = model(x)
                loss = criterion(y_pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            scaler.step(optimizer)
            scaler.update()
        else:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
        return loss.item()

    def create_supervised_trainer(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                                  criterion: nn.Module, clip_value: float = 1.0,
                                  use_amp: bool = False) -> Engine:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        return Engine(lambda engine, batch: self.train_step(engine, batch, model, optimizer,
                                                            criterion, self.device, scaler,
                                                            clip_value, use_amp))

    def train(self, fold: int, train_list: np.ndarray, train_labels: np.ndarray,
              val_list: np.ndarray, val_labels: np.ndarray,
              test_list: np.ndarray, test_labels: np.ndarray) -> np.ndarray:
        _, train_loader = self._get_data(train_list, train_labels, is_train=True)
        _, val_loader = self._get_data(val_list, val_labels, is_train=False)
        _, test_loader = self._get_data(test_list, test_labels, is_train=False)

        model = self._build_model()
        criterion = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)

        if self.args.mode == 'test':
            checkpoint = self.args.checkpoint or self.args.model_path
            self._load_state_dict(model, checkpoint, strict=True)
            return self.test(model, test_loader, criterion)

        if self.args.pretrained:
            self.logger.info('Loading pretrained checkpoint from %s', self.args.pretrained)
            self._load_state_dict(model, self.args.pretrained, strict=False, drop_classifier=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr0, weight_decay=self.args.weight_decay)
        step_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.2 ** (epoch / self.args.epochs))
        lr_scheduler = LRScheduler(step_scheduler)

        trainer = self.create_supervised_trainer(model, optimizer, criterion, clip_value=5,
                                                 use_amp=(not self.args.not_amp) and self.device.type != 'cpu')
        val_evaluator = create_supervised_evaluator(
            model,
            metrics={'accuracy': WFMetric(self.nmc), 'acc': Accuracy(), 'loss': Loss(criterion)},
            device=self.device,
            amp_mode=self.amp_mode,
        )

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_loss(engine: Engine):
            if self.args.verbose:
                grad_norm = get_grad_norm(model)
                self.logger.info('Fold[%d] | Epoch[%d] | Loss: %.4f | GradNorm: %.4f | LR: %.6g',
                                 fold, engine.state.epoch, engine.state.output, grad_norm,
                                 optimizer.param_groups[0]['lr'])

        @trainer.on(Events.EPOCH_COMPLETED)
        def validate(engine: Engine):
            val_evaluator.run(val_loader)
            metrics = val_evaluator.state.metrics
            if self.args.verbose:
                self.logger.info('Validation Fold[%d] Epoch[%d] | Loss: %.4f | Acc: %.4f | tp: %.0f fp: %.0f p: %.0f n: %.0f',
                                 fold, engine.state.epoch, metrics['loss'], metrics['acc'],
                                 metrics['accuracy'][0], metrics['accuracy'][1],
                                 metrics['accuracy'][2], metrics['accuracy'][3])
            if self.args.test_every_epoch:
                self.test(model, test_loader, criterion)

        trainer.add_event_handler(Events.EPOCH_STARTED, lr_scheduler)
        trainer.run(train_loader, max_epochs=self.args.epochs)
        res = self.test(model, test_loader, criterion)

        if not self.args.nosave:
            assert self.checkpoint_path is not None
            checkpoint_file = self.checkpoint_path / f'fold{fold}.pth'
            torch.save(self._plain_state_dict(model), checkpoint_file)
            self.logger.info('Model saved at %s', checkpoint_file)

        torch.cuda.empty_cache()
        return res

    def test(self, model: nn.Module, test_loader: DataLoader, criterion: nn.Module) -> np.ndarray:
        evaluator = create_supervised_evaluator(
            model,
            metrics={'accuracy': WFMetric(self.nmc), 'acc': Accuracy(), 'loss': Loss(criterion), 'pr': WFPRCurve(self.nmc)},
            device=self.device,
            amp_mode=self.amp_mode,
        )
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics

        if self.args.verbose:
            self.logger.info('Test | Loss: %.4f | Acc: %.4f | tp: %.0f fp: %.0f p: %.0f n: %.0f',
                             metrics['loss'], metrics['acc'], metrics['accuracy'][0], metrics['accuracy'][1],
                             metrics['accuracy'][2], metrics['accuracy'][3])
        return metrics['pr'] if self.args.open_world else np.array(metrics['accuracy'])

    def _print_result(self, res: np.ndarray):
        if self.args.open_world:
            precisions = res[:, 0] / (res[:, 0] + res[:, 1] + res[:, 2] + 1e-6)
            recalls = res[:, 0] / (res[:, 0] + res[:, 2] + res[:, 3] + 1e-6)
            for precision, recall in zip(precisions, recalls):
                print(f'{precision:.4f} {recall:.4f}')
        else:
            print('{:.0f} {:.0f} {:.0f} {:.0f}'.format(res[0], res[1], res[2], res[3]))

    @staticmethod
    def _plain_state_dict(model: nn.Module):
        return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

    @staticmethod
    def _load_state_dict(model: nn.Module, path: str | os.PathLike, strict: bool = True,
                         drop_classifier: bool = False):
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        target = model.module if isinstance(model, nn.DataParallel) else model
        if drop_classifier:
            target_state = target.state_dict()
            state_dict = {
                k: v for k, v in state_dict.items()
                if k in target_state and tuple(v.shape) == tuple(target_state[k].shape)
            }
        target.load_state_dict(state_dict, strict=strict)
