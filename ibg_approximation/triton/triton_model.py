import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.nn.conv.simple_conv import MessagePassing
from ibg_approximation.triton.triton_rspmm import RelConvSumAggr
from ibg_approximation.triton.triton_utils import coo_to_csr
import torch_geometric


class TritonFrobeniusNormMP(MessagePassing):
    def __init__(self, com_scale: Tensor):
        super().__init__(aggr='sum')
        self.com_scale = com_scale  # (K,)
        assert self.aggr == "sum", "This implementation is only for sum aggregation"

    def message_and_aggregate(self, edge_index: Adj, x_src: Tensor, x_dst: Tensor):
        # fused computation of message and aggregate steps with the custom rspmm cuda kernel
        # speed up computation by several times
        num_node = x_src.shape[0]
        rowptr, indices = coo_to_csr(edge_index[0], edge_index[1], num_node)
        update = RelConvSumAggr.apply(x_src, x_dst, rowptr, indices, num_node, self.com_scale, 0)
        return update

    def propagate(self, edge_index, size=None, **kwargs):
        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._fused_user_args, edge_index, size, kwargs)
        # TODO: use from packaging.version import parse as parse_version as by default 2.4 > 2.14 which is wrong
        pyg_version = [int(i) for i in torch_geometric.__version__.split(".")]
        col_fn = self.inspector.distribute if pyg_version[1] <= 4 else self.inspector.collect_param_data
        msg_aggr_kwargs = col_fn("message_and_aggregate", coll_dict)
        for hook in self._message_and_aggregate_forward_pre_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs))
            if res is not None:
                edge_index, msg_aggr_kwargs = res
        out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
        for hook in self._message_and_aggregate_forward_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs), out)
            if res is not None:
                out = res
        return out

    def forward(self, affiliate_mat_src: Tensor, affiliate_mat_dst: Tensor, edge_index: Adj, edge_weight: Tensor) -> Tensor:
        return self.propagate(edge_index=edge_index, x_src=affiliate_mat_src, x_dst=affiliate_mat_dst).sum()  # (N, K)