from torch_geometric.data import Data
from torch.optim import Optimizer
from typing import Tuple

from ibg_approximation.classes import IBGApproxTrainArgs
from ibg_approximation.utils import get_cut_norm
from helpers.constants import DECIMAL
from ibg_approximation.loss import calc_graphon_loss, calc_graphon_loss_kg, calc_sparse_graphon_loss
from ibg_approximation.model import IBGApproxModel
from additional_classes.dataset import DataSetFamily


def train_ibg_approximation(data: Data, model: IBGApproxModel, optimizer: Optimizer, train_args: IBGApproxTrainArgs,
                            pbar, device, dataset_family: DataSetFamily, triton: bool, sparse_loss: bool)\
        -> Tuple[float, float, IBGApproxModel]:
    model.train()

    best_loss, best_graphon_re = float('inf'), float('inf')
    best_model_state_dict = model.state_dict()
    is_kg, is_spatio_temporal = dataset_family.is_kg(), dataset_family.is_spatio_temporal()
    for epoch in range(train_args.epochs):
        optimizer.zero_grad()

        if is_kg:
            loss, graphon_loss, graphon_re = \
                calc_graphon_loss_kg(data=data, model=model, device=device, node_drop_ratio=train_args.node_drop_ratio)
        else:
            if sparse_loss == False:
                loss, graphon_loss, graphon_re = \
                    calc_graphon_loss(data=data, model=model, device=device, node_drop_ratio=train_args.node_drop_ratio,
                                    loss_scale=train_args.loss_scale, is_spatio_temporal=is_spatio_temporal, triton=triton)
            else:
                loss, graphon_loss, graphon_re = \
                    calc_sparse_graphon_loss(data=data, model=model, device=device, node_drop_ratio=train_args.node_drop_ratio,
                                    loss_scale=train_args.loss_scale, is_spatio_temporal=is_spatio_temporal, triton=triton,
                                    sparse_scale=train_args.sparse_scale)

        detach_loss = loss.item()
        graphon_re = graphon_re.item()
        loss.backward()
        optimizer.step()

        # best
        if detach_loss < best_loss:
            best_model_state_dict = model.state_dict()
            best_loss = detach_loss
            best_graphon_re = graphon_re


        # print
        log_str = f';GS;epoch:{epoch};loss={round(detach_loss, DECIMAL)}({round(best_loss, DECIMAL)})'
        log_str += f';g_re={round(graphon_re, DECIMAL)}({round(best_graphon_re, DECIMAL)})'
        if train_args.cut_norm:
            cutn_round, cutn_sdp = get_cut_norm(model=model, edge_index=data.edge_index)
            log_str += f';cutn_round={round(cutn_round, DECIMAL)};cutn_sdp={round(cutn_sdp, DECIMAL)}'
        pbar.set_description(log_str)
        pbar.update(n=1)

    model.eval()
    model.load_state_dict(best_model_state_dict)
    return best_loss, best_graphon_re, model
