import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Dropout, Linear, functional as F
from torch_geometric.typing import Adj
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn.norm import LayerNorm

from ibgnn.classes import IBGNNArgs, NNArgs
from ibgnn.layer import ComLayer
from ibg_approximation.model import IBGApproxModel


class NNModel(Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, ibgnn_args: NNArgs):
        super().__init__()
        in_dim_list = [in_dim] + [hidden_dim] * (ibgnn_args.num_layers - 1)
        hidden_dim_list = [hidden_dim] * ibgnn_args.num_layers
        out_dim_list = [hidden_dim] * (ibgnn_args.num_layers - 1) + [out_dim]
        self.src_layers = ModuleList([ComLayer(in_dim=in_channels, hidden_dim=hidden_channels, out_dim=out_channels,
                                               ibgnn_args=ibgnn_args)
                                      for in_channels, hidden_channels, out_channels in
                                        zip(in_dim_list, hidden_dim_list, out_dim_list)])
        
        self.dst_layers = ModuleList([ComLayer(in_dim=in_channels, hidden_dim=hidden_channels, out_dim=out_channels,
                                               ibgnn_args=ibgnn_args)
                                      for in_channels, hidden_channels, out_channels in
                                        zip(in_dim_list, hidden_dim_list, out_dim_list)])
        


        self.dropout = Dropout(ibgnn_args.dropout)
        self.act = ibgnn_args.act_type.get()
        self.jump = ibgnn_args.jump
        if self.jump is not None:
            lin_input_dim = hidden_dim * (ibgnn_args.num_layers - 1) if self.jump == "cat" else hidden_dim
            self.jk = JumpingKnowledge(ibgnn_args.jump, channels=hidden_dim, num_layers=ibgnn_args.num_layers)
            self.lin1 = Linear(lin_input_dim, out_dim)
        
        self.layer_norm = ibgnn_args.layer_norm
        if ibgnn_args.layer_norm:
            self.layer_norms = ModuleList([LayerNorm(hidden_dim) for _ in range(ibgnn_args.num_layers)])
            
        self.normalize = ibgnn_args.normalize
        

    def forward(self, x: Tensor, affiliate_times_scale: Tensor, inv_affiliate_mat: Tensor, type: str) -> Tensor:
        layers = self.src_layers if type == 'src' else self.dst_layers

        xs = []
        for i, layer in enumerate(layers[:-1]):
            feat_mat = inv_affiliate_mat @ x  # (K, C)
            x = layer(x=x, feat_mat=feat_mat, affiliate_times_scale=affiliate_times_scale)  # (N, C)

            x = self.act(x)
            x = self.dropout(x)
            if self.normalize:
                x = F.normalize(x, p=2, dim=-1)

            if self.layer_norm:
                x = self.layer_norms[i](x)
            xs.append(x)

        if self.jump is None:
            feat_mat = inv_affiliate_mat @ x  # (K, C)
            x = layers[-1](x=x, feat_mat=feat_mat, affiliate_times_scale=affiliate_times_scale)
        else:
            x = self.jk(xs)
            x = self.lin1(x)

        return x


class IBGNN(Module):
    def __init__(self, model_args: IBGNNArgs, ibg_approx_model: IBGApproxModel):
        super().__init__()
        self.ibg_approx_model = ibg_approx_model
        self.nn_model = NNModel(in_dim=model_args.encoded_dim, hidden_dim=model_args.hidden_dim,
                                out_dim=model_args.out_dim, ibgnn_args=model_args.ibgnn_args)

        self.ibg_approx_model.eval()

    # gradients
    def train(self, mode: bool = True):
        self.nn_model.train()

    def get_ibgnn_parameters(self):
        return self.nn_model.parameters()

    def get_ibg_approx_parameters(self):
        return self.ibg_approx_model.parameters()

    def set_ibg_approx_after_training(self):
        self.ibg_approx_model.set_matrices_after_ibg_approx_training()
        self.ibg_approx_model.requires_grad_(requires_grad=False)

    def allow_ibg_approx_training(self):
        self.ibg_approx_model.set_matrices_for_refinement()
        self.ibg_approx_model.train()
        self.ibg_approx_model.requires_grad_(requires_grad=True)

    # forward pass
    def encode(self, x: Tensor) -> Tensor:
        return self.ibg_approx_model.encode(x=x)

    def forward(self, x: Tensor) -> Tensor:
        out_src = self.nn_model(x=self.encode(x),
                                affiliate_times_scale=self.ibg_approx_model.affiliate_src_times_scale,
                                inv_affiliate_mat=self.ibg_approx_model.inv_affiliate_mat_src, type='src')
        out_dst = self.nn_model(x=self.encode(x),
                                affiliate_times_scale=self.ibg_approx_model.affiliate_dst_times_scale,
                                inv_affiliate_mat=self.ibg_approx_model.inv_affiliate_mat_dst, type='dst')

        out = (out_src + out_dst) / 2
        return out
