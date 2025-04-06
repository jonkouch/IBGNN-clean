from torch import Tensor
from torch.nn import Module

from ibgnn.classes import NNArgs


class ComLayer(Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, ibgnn_args: NNArgs):
        super().__init__()
        self.com_trans = ibgnn_args.get_com_model(dim=in_dim)
        self.com_node_trans = ibgnn_args.get_com_node_model(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
        self.node_trans = ibgnn_args.get_node_model(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
        self.skip_node = ibgnn_args.skip_node and (in_dim == out_dim)        
        

    def forward(self, x: Tensor, feat_mat: Tensor, affiliate_times_scale: Tensor) -> Tensor:
        community_outputs = self.com_trans(feat_mat)  # (K, C)
        out = self.com_node_trans(affiliate_times_scale @ community_outputs)  # (N, C)
        out = out + self.node_trans(x)
        if self.skip_node:
            out = out + x
        return out  # (N, C)




