import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple, Optional
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.data import Data

from ibg_approximation.forbenius_norm_mp import FrobeniusNormMessagePassing
from ibg_approximation.triton.triton_model import TritonFrobeniusNormMP


def calc_inefficient_graphon_loss(model, edge_index: Adj, l_norm: int) -> Tensor:
    bct_minus_a = model.affiliate_src_times_scale @ model.affiliate_mat_dst.T  # (N, N)
    u, v = edge_index
    bct_minus_a[u, v] = bct_minus_a[u, v] - 1
    return torch.sum(bct_minus_a ** 2) / bct_minus_a.shape[0]


def drop_nodes(edge_index: Adj, affiliate_mat_src: Tensor, affiliate_mat_dst: Tensor, node_drop_ratio: float,
               x: OptTensor = None, edge_attr: OptTensor = None) -> Tuple[Tensor, Adj, Tensor, Tensor, OptTensor]:
    # filter nodes
    device = x.device
    prob_vector = (1 - node_drop_ratio) * torch.ones(size=(affiliate_mat_src.shape[0],), device=device)
    indices_kept = torch.bernoulli(prob_vector).type(torch.bool)
    if x is not None:
        x = x[indices_kept]
    affiliate_mat_src = affiliate_mat_src[indices_kept]
    affiliate_mat_dst = affiliate_mat_dst[indices_kept]

    # filter edges
    u, v = edge_index
    u_kept, v_kept = indices_kept[u], indices_kept[v]
    edge_kept = torch.logical_and(u_kept, v_kept)
    edge_index = edge_index[:, edge_kept]
    if edge_attr is not None:
        edge_attr = edge_attr[edge_kept]
        _, edge_attr = edge_attr.unique(return_inverse=True)
        edge_attr = edge_attr.view(edge_index.size(1), -1)

    # edge index reindexing
    _, edge_index = edge_index.unique(return_inverse=True)
    edge_index = edge_index.view(2, -1)
    return edge_index, affiliate_mat_src, affiliate_mat_dst, x, edge_attr


def calc_graphon_loss(data: Data, model, device, node_drop_ratio: float, loss_scale: float,
                      is_spatio_temporal: bool, triton: bool) -> Tuple[Tensor, Tensor, Tensor]:
    if is_spatio_temporal:
        num_time_steps, num_feat, num_nodes = data.x.shape[0], data.x.shape[1], data.x.shape[2]
        edge_weight = data.edge_weight.to(device)
    else:
        num_time_steps, num_nodes, num_feat = 1, data.x.shape[0], data.x.shape[1]
        edge_weight = None

    # global term
    com_scale = model.com_scale
    affiliate_mat_src = model.affiliate_mat_src
    affiliate_mat_dst = model.affiliate_mat_dst
    global_term = torch.trace(affiliate_mat_dst.T @ model.affiliate_dst_times_scale @ affiliate_mat_src.T \
                              @ model.affiliate_src_times_scale)  # (,)

    # node drop
    x = model.encode(x=data.x.to(device))
    edge_index = data.edge_index.to(device)
    if node_drop_ratio > 0:
        edge_index, affiliate_mat_src, affiliate_mat_dst, x, edge_weight = \
            drop_nodes(x=x, edge_index=edge_index, affiliate_mat_src=affiliate_mat_src,
                       affiliate_mat_dst=affiliate_mat_dst, node_drop_ratio=node_drop_ratio, edge_attr=edge_weight)
    if is_spatio_temporal:
        num_edges = (edge_weight ** 2).sum()
    else:
        num_edges = edge_index.shape[1]

    # message passing term
    if triton:
        frobenius_norm_mp = TritonFrobeniusNormMP(com_scale=com_scale).to(device=device)
    else:
        frobenius_norm_mp = FrobeniusNormMessagePassing(com_scale=com_scale, is_kg=False).to(device=device)
    local_term = 2 * frobenius_norm_mp(affiliate_mat_src=affiliate_mat_src, affiliate_mat_dst=affiliate_mat_dst,
                                       edge_index=edge_index, edge_weight=edge_weight)  # (,)
    # loss
    graphon_loss = (global_term - local_term + num_edges) / num_nodes


    if loss_scale > 0:
        x_approx = torch.matmul((affiliate_mat_src * model.com_scale), model.feat_mat_src) \
                   + torch.matmul((affiliate_mat_dst * model.com_scale), model.feat_mat_dst)
        if is_spatio_temporal:
            x_approx = x_approx.unsqueeze(dim=0)
        signal_loss = torch.sum((x - x_approx) ** 2) / (num_feat * num_time_steps)
    else:
        signal_loss = 0
    graphon_re_denom = num_edges / num_nodes
    return graphon_loss + loss_scale * signal_loss, graphon_loss,  (graphon_loss / graphon_re_denom) ** 0.5



def calc_sparse_graphon_loss(data: Data, model, device, node_drop_ratio: float, loss_scale: float,
                      is_spatio_temporal: bool, triton: bool, sparse_scale: float) -> Tuple[Tensor, Tensor, Tensor]:
    if is_spatio_temporal:
        num_time_steps, num_feat, num_nodes = data.x.shape[0], data.x.shape[1], data.x.shape[2]
        edge_weight = data.edge_weight.to(device)
    else:
        num_time_steps, num_nodes, num_feat = 1, data.x.shape[0], data.x.shape[1]
        edge_weight = None

    # global term
    com_scale = model.com_scale
    affiliate_mat_src = model.affiliate_mat_src
    affiliate_mat_dst = model.affiliate_mat_dst
    global_term = torch.trace(affiliate_mat_dst.T @ model.affiliate_dst_times_scale @ affiliate_mat_src.T \
                              @ model.affiliate_src_times_scale)  # (,)

    # node drop
    x = model.encode(x=data.x.to(device))
    edge_index = data.edge_index.to(device)
    if node_drop_ratio > 0:
        edge_index, affiliate_mat_src, affiliate_mat_dst, x, edge_weight = \
            drop_nodes(x=x, edge_index=edge_index, affiliate_mat_src=affiliate_mat_src,
                       affiliate_mat_dst=affiliate_mat_dst, node_drop_ratio=node_drop_ratio, edge_attr=edge_weight)
    if is_spatio_temporal:
        num_edges = (edge_weight ** 2).sum()
    else:
        num_edges = edge_index.shape[1]

    sparse_weight = (sparse_scale * num_edges) / (num_nodes**2 - num_edges)

    # message passing term
    frobenius_norm_mp = FrobeniusNormMessagePassing(com_scale=com_scale, is_kg=False).to(device=device)
    local_term = 2 * frobenius_norm_mp(affiliate_mat_src=affiliate_mat_src, affiliate_mat_dst=affiliate_mat_dst,
                                       edge_index=edge_index, edge_weight=edge_weight)  # (,)

    sparse_term = (1 - sparse_weight) * frobenius_norm_mp(affiliate_mat_src=affiliate_mat_src, 
                                                          affiliate_mat_dst=affiliate_mat_dst,
                                                          edge_index=edge_index, square=True)  # (,)
    # loss
    graphon_loss = (sparse_weight * global_term - local_term + sparse_term + num_edges)

    if loss_scale > 0:
        x_approx = torch.matmul((affiliate_mat_src * model.com_scale), model.feat_mat_src) \
                   + torch.matmul((affiliate_mat_dst * model.com_scale), model.feat_mat_dst)
        if is_spatio_temporal:
            x_approx = x_approx.unsqueeze(dim=0)
        signal_loss = torch.sum((x - x_approx) ** 2) / (num_feat * num_time_steps)
    else:
        signal_loss = 0
    graphon_re_denom = num_edges / num_nodes
    return graphon_loss + loss_scale * signal_loss, graphon_loss,  (graphon_loss / graphon_re_denom) ** 0.5



def calc_graphon_loss_kg(data: Data, model, device, node_drop_ratio: float) -> Tuple[Tensor, Tensor, Tensor]:
    num_nodes, num_feat = data.num_node, model.com_scale.shape[-1]
    train_edge_list = data.train_edge_list
    edge_index = train_edge_list[:, :, :2].T.to(device)
    edge_types = train_edge_list[:, :, 2].T.to(device)

    # global term
    com_scale = model.com_scale
    affiliate_mat_src = model.affiliate_mat_src
    affiliate_mat_dst = model.affiliate_mat_dst
    global_term = torch.trace(affiliate_mat_dst.T @ model.affiliate_dst_times_scale @ affiliate_mat_src.T \
                              @ (affiliate_mat_src * com_scale))  # (,)

    # node drop
    onehot_relations = torch.eye(data.num_relation).float().to(device)  # (R, R)
    edge_feat = model.encode(x=onehot_relations)[edge_types]  # (R, D) -> (E, D)
    if node_drop_ratio > 0:
        edge_index, affiliate_mat_src, affiliate_mat_dst, _, edge_attr = \
            drop_nodes(edge_attr=edge_feat, edge_index=edge_index, affiliate_mat_src=affiliate_mat_src,
                       affiliate_mat_dst=affiliate_mat_dst, node_drop_ratio=node_drop_ratio)
    num_edges = edge_index.shape[1]

    frobenius_norm_mp = FrobeniusNormMessagePassing(com_scale=com_scale, is_kg=True).to(device=device)
    local_term = 2 * frobenius_norm_mp(affiliate_mat_src=affiliate_mat_src, affiliate_mat_dst=affiliate_mat_dst,
                                       edge_index=edge_index)  # (,)
    graphon_loss = (global_term - local_term + num_edges) / num_nodes
    graphon_re_denom = num_edges / num_nodes
    return graphon_loss, graphon_loss,  (graphon_loss / graphon_re_denom) ** 0.5