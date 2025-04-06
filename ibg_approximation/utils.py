import torch
from torch import Tensor
from typing import Tuple
from torch_geometric.utils import to_dense_adj
from torch_geometric.typing import Adj
import os
import random
import numpy as np
import re

from cutnorm import compute_cutnorm
from helpers.constants import EPS, ROOT_DIR
from ibg_approximation.classes import IBGApproxArgs, IBGApproxTrainArgs


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False


def get_device() -> str:
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        min_memory_usage = float('inf')
        selected_device = None
        
        for i in range(num_gpus):
            memory_used = torch.cuda.memory_allocated(i)
            if memory_used < min_memory_usage:
                min_memory_usage = memory_used
                selected_device = i

        return f'cuda:{selected_device}'
    else:

        return 'cpu'


def inv_sigmoid(x: Tensor) -> Tensor:
    return torch.log(x / (1 - x))


def transform_eigenvals_and_eigenvecs(eigenvals: Tensor, eigenvecs_src: Tensor, eigenvecs_dst: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    max_eigenvecs_src = torch.max(eigenvecs_src, dim=0)[0]
    new_eigenvecs_src = (eigenvecs_src + EPS) / (max_eigenvecs_src + EPS)

    max_eigenvecs_dst = torch.max(eigenvecs_dst, dim=0)[0]
    new_eigenvecs_dst = (eigenvecs_dst + EPS) / (max_eigenvecs_dst + EPS)

    new_eigenvals = eigenvals * max_eigenvecs_src * max_eigenvecs_dst
    return new_eigenvals, new_eigenvecs_src, new_eigenvecs_dst


def get_cut_norm(model, edge_index: Adj) -> Tuple[float, float]:
    affiliate_mat_dst = model.affiliate_mat_dst
    affiliate_src_times_scale = model.affiliate_src_times_scale

    model_adj = affiliate_src_times_scale @ affiliate_mat_dst.T  # (N, N)
    model_adj = model_adj.cpu().detach().numpy()
    data_adj = to_dense_adj(edge_index=edge_index).squeeze(dim=0)
    data_adj = data_adj.cpu().detach().numpy()
    cutn_round, cutn_sdp, _ = compute_cutnorm(A=model_adj, B=data_adj)
    return cutn_round, cutn_sdp


def exp_path(dataset_name: str, ibg_approx_args: IBGApproxArgs, ibg_approx_train_args: IBGApproxTrainArgs, seed: int,
             load_epoch: int = 0, sparse: bool = False, sparse_scale: float = 1.0) -> str:

    epochs = ibg_approx_train_args.epochs if load_epoch == 0 else load_epoch

    run_folder = f'{dataset_name}_' \
                 f'{ibg_approx_args.num_communities}_' \
                 f'Enc{int(ibg_approx_args.encode_dim)}_' \
                 f'Eig{int(ibg_approx_args.add_eigen)}_' \
                 f'{epochs}_' \
                 f'{ibg_approx_train_args.lr}_' \
                 f'{ibg_approx_train_args.loss_scale}_' \
                 f'dir{int(ibg_approx_train_args.is_undirected)}_' \
                 f'{seed}'
    
    if sparse:
        run_folder += f'_sparse_{sparse_scale}'
                 
    if ibg_approx_train_args.node_drop_ratio > 0:
        run_folder += f'_{ibg_approx_train_args.node_drop_ratio}'
    return os.path.join(ROOT_DIR, 'ibg_approximation_models', run_folder)


def get_init_com_scale_and_affiliate_mat(num_communities: int, init_com_scale: Tensor,
                                         init_affiliate_mat_src: Tensor, init_affiliate_mat_dst: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    
    # com_scale
    com_scale = init_com_scale.repeat_interleave(4)[:num_communities]
    com_scale[2::4] = - com_scale[2::4]
    com_scale[3::4] = - com_scale[3::4]

    # normalize
    pos_support_src = torch.relu(init_affiliate_mat_src)
    neg_support_src = torch.relu(-init_affiliate_mat_src)
    pos_support_dst = torch.relu(init_affiliate_mat_dst)
    neg_support_dst = torch.relu(-init_affiliate_mat_dst)

    # affiliate_mat - sigmoid inverse
    affiliate_mat_src = torch.zeros(size=(init_affiliate_mat_src.shape[0], num_communities), device=init_com_scale.device)
    affiliate_mat_dst = torch.zeros(size=(init_affiliate_mat_dst.shape[0], num_communities), device=init_com_scale.device)

    com_scale[::4], affiliate_mat_src[:, ::4], affiliate_mat_dst[:, ::4] = \
        transform_eigenvals_and_eigenvecs(eigenvals=com_scale[::4], eigenvecs_src=pos_support_src, eigenvecs_dst=pos_support_dst)
    if num_communities % 4 == 0:
        com_scale[1::4], affiliate_mat_src[:, 1::4], affiliate_mat_dst[:, 1::4] = \
            transform_eigenvals_and_eigenvecs(eigenvals=com_scale[1::4], eigenvecs_src=neg_support_src, eigenvecs_dst=neg_support_dst)
        com_scale[2::4], affiliate_mat_src[:, 2::4], affiliate_mat_dst[:, 2::4] = \
            transform_eigenvals_and_eigenvecs(eigenvals=com_scale[2::4], eigenvecs_src=pos_support_src, eigenvecs_dst=neg_support_dst)
        com_scale[3::4], affiliate_mat_src[:, 3::4], affiliate_mat_dst[:, 3::4] = \
            transform_eigenvals_and_eigenvecs(eigenvals=com_scale[3::4], eigenvecs_src=neg_support_src, eigenvecs_dst=pos_support_dst)
    elif num_communities % 4 == 1:
        com_scale[1::4], affiliate_mat_src[:, 1::4], affiliate_mat_dst[:, 1::4] = \
            transform_eigenvals_and_eigenvecs(eigenvals=com_scale[1::4], eigenvecs_src=neg_support_src[:, :-1], eigenvecs_dst=neg_support_dst[:, :-1])
        com_scale[2::4], affiliate_mat_src[:, 2::4], affiliate_mat_dst[:, 2::4] = \
            transform_eigenvals_and_eigenvecs(eigenvals=com_scale[2::4], eigenvecs_src=pos_support_src[:, :-1], eigenvecs_dst=neg_support_dst[:, :-1])
        com_scale[3::4], affiliate_mat_src[:, 3::4], affiliate_mat_dst[:, 3::4] = \
            transform_eigenvals_and_eigenvecs(eigenvals=com_scale[3::4], eigenvecs_src=neg_support_src[:, :-1], eigenvecs_dst=pos_support_dst[:, :-1])
    elif num_communities % 4 == 2:
        com_scale[1::4], affiliate_mat_src[:, 1::4], affiliate_mat_dst[:, 1::4] = \
            transform_eigenvals_and_eigenvecs(eigenvals=com_scale[1::4], eigenvecs_src=neg_support_src, eigenvecs_dst=neg_support_dst)
        com_scale[2::4], affiliate_mat_src[:, 2::4], affiliate_mat_dst[:, 2::4] = \
            transform_eigenvals_and_eigenvecs(eigenvals=com_scale[2::4], eigenvecs_src=pos_support_src[:, :-1], eigenvecs_dst=neg_support_dst[:, :-1])
        com_scale[3::4], affiliate_mat_src[:, 3::4], affiliate_mat_dst[:, 3::4] = \
            transform_eigenvals_and_eigenvecs(eigenvals=com_scale[3::4], eigenvecs_src=neg_support_src[:, :-1], eigenvecs_dst=pos_support_dst[:, :-1])
    elif num_communities % 4 == 3:
        com_scale[1::4], affiliate_mat_src[:, 1::4], affiliate_mat_dst[:, 1::4] = \
            transform_eigenvals_and_eigenvecs(eigenvals=com_scale[1::4], eigenvecs_src=neg_support_src, eigenvecs_dst=neg_support_dst)
        com_scale[2::4], affiliate_mat_src[:, 2::4], affiliate_mat_dst[:, 2::4] = \
            transform_eigenvals_and_eigenvecs(eigenvals=com_scale[2::4], eigenvecs_src=pos_support_src, eigenvecs_dst=neg_support_dst)
        com_scale[3::4], affiliate_mat_src[:, 3::4], affiliate_mat_dst[:, 3::4] = \
            transform_eigenvals_and_eigenvecs(eigenvals=com_scale[3::4], eigenvecs_src=neg_support_src[:, :-1], eigenvecs_dst=pos_support_dst[:, :-1])
    return com_scale, affiliate_mat_src, affiliate_mat_dst
