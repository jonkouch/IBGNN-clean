import torch
from typing import Tuple, Optional
from torch_geometric.data import Data
from torch.optim import Optimizer
import sys
import tqdm

import os.path as osp
import os
import time

from additional_classes.dataset import DataSetType
from additional_classes.metrics import MetricType, LossesAndMetrics
from ibgnn.classes import NNTrainArgs, IBGNNArgs
from ibg_approximation.model import IBGApproxModel
from ibg_approximation.utils import set_seed
from ibgnn.utils import ibgnn_final_log_and_print, epoch_log_and_print
from ibgnn.model import IBGNN
from helpers.constants import DECIMAL

import time


class NNTrainer(object):
    def __init__(self, model_args: IBGNNArgs, train_args: NNTrainArgs,
                 seed: int, device):
        super().__init__()
        self.model_args = model_args
        self.train_args = train_args
        self.seed = seed
        self.device = device

    def train_and_test_splits(self, dataset_type: DataSetType, data: Data, path: str,
                              ibg_approx_model: Optional[IBGApproxModel]) -> IBGNN:
        folds = dataset_type.get_folds()
        metrics_list = []
        avg_time = 0
        in_dim = data.x.shape[-1]
        start_time_conv = time.time()
        for num_fold in folds:
            set_seed(seed=self.seed)
            data_split = dataset_type.select_fold(num_fold=num_fold, data=data)
            print_str = f'Fold{num_fold}'

            model = IBGNN(model_args=self.model_args, ibg_approx_model=ibg_approx_model).to(self.device)
            model.set_ibg_approx_after_training()
            optimizer = torch.optim.Adam(model.get_ibgnn_parameters(), lr=self.train_args.lr)


            start_time = time.time()
            with tqdm.tqdm(total=self.train_args.epochs, file=sys.stdout) as pbar:
                best_losses_n_metrics, model, final_epoch = \
                    self.train_and_test(dataset_type=dataset_type, data=data_split, model=model,
                                        optimizer=optimizer, epochs=self.train_args.epochs, pbar=pbar,
                                        fold_str=print_str)

            fold_time = time.time() - start_time
            avg_time += fold_time

            

            # print final
            for name in best_losses_n_metrics._fields:
                print_str += f";{name}={round(getattr(best_losses_n_metrics, name), DECIMAL)}"
            print(print_str)
            print()
            metrics_list.append(torch.tensor(best_losses_n_metrics.get_metrics()))

        conv_time = time.time() - start_time_conv

        metrics_matrix = torch.stack(metrics_list, dim=0)  # (F, 3)
        avg_time /= len(folds)
        model_args_str = f'{self.model_args.get_args_str()}_{self.train_args.lr}_{self.train_args.epochs}'
        path = osp.join(path, model_args_str)

        if not osp.exists(path):
            os.makedirs(path)

        ibgnn_final_log_and_print(metrics_matrix=metrics_matrix, path=path, avg_time=avg_time,
                                  runtime=conv_time, final_epoch=final_epoch)
        return model

    def train_and_test(self, dataset_type: DataSetType, data: Data, model: IBGNN,
                       optimizer: Optimizer, epochs: int, pbar,
                       fold_str: str) -> Tuple[LossesAndMetrics, IBGNN]:
        metric_type = dataset_type.get_metric_type()
        task_loss = metric_type.get_task_loss()
        best_losses_n_metrics = metric_type.get_worst_losses_n_metrics()
        best_model_state_dict = model.state_dict()
        patience = self.train_args.patience if self.train_args.patience > 0 else epochs
        final_epoch = epochs

        for epoch in range(epochs):
            if self.train_args.refine_ibg_epoch > 0:
                if epoch == self.train_args.refine_ibg_epoch:
                    model.allow_ibg_approx_training()
                    approx_params = model.get_ibg_approx_parameters()
                    optimizer.add_param_group({'params': approx_params, 'lr': self.train_args.lr * self.train_args.refine_scale})

            self.train(data=data, model=model, optimizer=optimizer, task_loss=task_loss)
            train_loss, train_metric = self.test(data=data, model=model, task_loss=task_loss, metric_type=metric_type,
                                                 mask_name='train_mask')
            val_loss, val_metric = self.test(data=data, model=model, task_loss=task_loss, metric_type=metric_type,
                                             mask_name='val_mask')
            test_loss, test_metric = self.test(data=data, model=model, task_loss=task_loss, metric_type=metric_type,
                                               mask_name='test_mask')
            losses_n_metrics = \
                LossesAndMetrics(train_loss=train_loss, val_loss=val_loss, test_loss=test_loss,
                                 train_metric=train_metric, val_metric=val_metric, test_metric=test_metric)



            # best metrics
            if metric_type.src_better_than_other(src=losses_n_metrics.val_metric,
                                                 other=best_losses_n_metrics.val_metric):
                patience = self.train_args.patience if self.train_args.patience > 0 else epochs
                best_losses_n_metrics = losses_n_metrics
                best_model_state_dict = model.state_dict()
            else:
                patience -= 1

            log_str = epoch_log_and_print(epoch=epoch, losses_n_metrics=losses_n_metrics,
                                          best_test_metric=best_losses_n_metrics.test_metric, fold_str=fold_str)
            pbar.set_description(log_str)
            pbar.update(n=1)
            

            if (patience == 0):
                final_epoch = epoch
                break




        model.load_state_dict(best_model_state_dict)
        return best_losses_n_metrics, model, final_epoch

    def train(self, data: Data, model: IBGNN, optimizer, task_loss):
        model.train()
        optimizer.zero_grad()

        # forward
        scores = model(x=data.x.to(device=self.device))
        train_mask = data.train_mask.to(device=self.device)
        loss = task_loss(scores[train_mask], data.y.to(device=self.device)[train_mask].squeeze())

        # backward
        loss.backward()
        optimizer.step()

    def test(self, data: Data, model: IBGNN, task_loss, metric_type: MetricType,
             mask_name: str) -> Tuple[float, float]:
        model.eval()

        scores = model(x=data.x.to(device=self.device))
        mask = getattr(data, mask_name).to(device=self.device)

        loss = task_loss(scores[mask], data.y.to(device=self.device)[mask].squeeze()).item()
        metric = metric_type.apply_metric(scores=scores[mask].detach().cpu().numpy(),
                                          target=data.y[mask.cpu()].numpy().squeeze())
        return loss, metric
