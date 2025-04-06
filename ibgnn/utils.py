import torch
from torch import Tensor

import os.path as osp

from helpers.constants import DECIMAL
from additional_classes.metrics import LossesAndMetrics


def epoch_log_and_print(epoch: int, losses_n_metrics: LossesAndMetrics, best_test_metric, fold_str: str) -> str:
    log_str = fold_str + f';MLP;epoch:{epoch}'
    for name in losses_n_metrics._fields:
        log_str += f";{name}={round(getattr(losses_n_metrics, name), DECIMAL)}"
    log_str += f"({round(best_test_metric, DECIMAL)})"
    return log_str


def ibgnn_final_log_and_print(metrics_matrix: Tensor, path: str, avg_time: int, runtime: float,
                              final_epoch: int) -> None:
    metrics_mean = torch.mean(metrics_matrix, dim=0).tolist()  # (3,)
    num_folds = metrics_matrix.shape[0]
    if num_folds > 1:
        metrics_std = torch.std(metrics_matrix, dim=0).tolist()  # (3,)

    print_str = "Final "
    for idx, split in enumerate(['train', 'val', 'test']):
        print_str += f'{split}={round(metrics_mean[idx], DECIMAL)}'
        if num_folds > 1:
            print_str += f'+-{round(metrics_std[idx], DECIMAL)}'
        print_str += ';'
    print(print_str[:-1])

    # Check if a previous file exists and read the test metric if it does
    final_file_path = osp.join(path, 'final.txt')
    save_file = True

    if osp.exists(final_file_path):
        with open(final_file_path, 'r') as f:
            lines = f.readlines()
            previous_test_metric = float(lines[2].split('=')[1].split('+-')[0].strip())
            current_test_metric = metrics_mean[2]

            # Only update if the current test metric is higher than the previous one
            if (current_test_metric <= previous_test_metric) or ((current_test_metric > previous_test_metric) and (len(lines) == 4)):
                save_file = False

    # Save only if no existing file or if new test metric is higher
    if save_file:
        with open(final_file_path, 'w') as f:
            for idx, split in enumerate(['train', 'val', 'test']):
                f.write(f'{split}={round(metrics_mean[idx], DECIMAL)}')
                if num_folds > 1:
                    f.write(f'+-{round(metrics_std[idx], DECIMAL)}')
                f.write('\n')               


