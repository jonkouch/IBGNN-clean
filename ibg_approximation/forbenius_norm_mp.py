from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv.simple_conv import MessagePassing


class FrobeniusNormMessagePassing(MessagePassing):
    def __init__(self, com_scale: Tensor, is_kg: bool):
        super().__init__(aggr='sum')
        self.com_scale = com_scale  # (K,) or (K, D)
        self.is_kg = is_kg

    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: OptTensor = None, square: bool=False) -> Tensor:
        # x_i is the target nodes of dim (E, K)
        # x_j is the source nodes of dim (E, K)
        if not self.is_kg:
            message_term = (x_i * self.com_scale * x_j).sum(dim=1, keepdims=True)  # (E, 1)
            if edge_weight is not None:
                message_term = message_term * edge_weight.unsqueeze(1)
        else:
            x_i = x_i.unsqueeze(-1)  # (E, K, 1)
            x_j = x_j.unsqueeze(-1)  # (E, K, 1)
            com_scale = self.com_scale.unsqueeze(0)  # (1, K, D)
            message_term = (x_i * com_scale * x_j).sum(dim=1)  # (E, D)
            if edge_weight is not None:
                message_term = message_term * edge_weight
        if square:
            message_term = message_term ** 2
        return message_term

    def forward(self, affiliate_mat_src: Tensor, affiliate_mat_dst: Tensor, edge_index: Adj, 
                edge_weight: OptTensor = None, square: bool = False) -> Tensor:
        node_based_result = self.propagate(
            edge_index, 
            x=(affiliate_mat_src, affiliate_mat_dst), # T: dst nodes - agg information, B: src nodes - send information
            edge_weight=edge_weight, square=square)
        # sum over the nodes and channels
        return node_based_result.sum()

        
