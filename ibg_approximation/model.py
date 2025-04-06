import torch
from torch import Tensor
from torch.nn import Parameter, Module, init, Linear, Identity
import math
from torch_geometric.typing import OptTensor

from ibg_approximation.classes import IBGApproxArgs
from ibg_approximation.utils import get_init_com_scale_and_affiliate_mat, inv_sigmoid
from ibg_approximation.scale_gradient import ScaleGradientModule


class IBGApproxModel(Module):
    def __init__(self, model_args: IBGApproxArgs, is_ibgnn: bool):
        super().__init__()
        num_nodes = model_args.num_nodes
        num_communities = model_args.num_communities
        self.num_communities = num_communities

        # nodes
        self._affiliate_mat_src = Parameter(torch.zeros((num_nodes, num_communities)))
        self._affiliate_mat_dst = Parameter(torch.zeros((num_nodes, num_communities)))
        if model_args.is_kg:
            self.com_scale = Parameter(torch.zeros((num_communities, model_args.encode_dim)))
        else:
            self.com_scale = Parameter(torch.zeros((num_communities,)))
        self._inv_affiliate_mat_src = None
        self._inv_affiliate_mat_dst = None
        
        # features
        self.encoder = None
        if model_args.encode_dim > 0:
            self.encoder = Linear(in_features=model_args.in_dim, out_features=model_args.encode_dim)
        if model_args.time_steps > 1:
            self.feat_mat_src = Parameter(torch.zeros((model_args.time_steps, num_communities, model_args.encoded_dim)))
            self.feat_mat_dst = Parameter(torch.zeros((model_args.time_steps, num_communities, model_args.encoded_dim)))
        else:
            self.feat_mat_src = Parameter(torch.zeros((num_communities, model_args.encoded_dim)))
            self.feat_mat_dst = Parameter(torch.zeros((num_communities, model_args.encoded_dim)))

        # node drop
        if model_args.node_drop_ratio > 0:
            num_left = int(num_nodes * (1 - model_args.node_drop_ratio))
            self.scale_grad = ScaleGradientModule(scale=num_left / num_nodes)
        else:
            self.scale_grad = Identity()

        if not is_ibgnn:
            self.reset_parameters(init_com_scale=model_args.init_com_scale,
                                  init_affiliate_mat_src=model_args.init_affiliate_mat_src,
                                  init_affiliate_mat_dst=model_args.init_affiliate_mat_dst,
                                  init_feat_mat_src=model_args.init_feat_mat_src,
                                  init_feat_mat_dst=model_args.init_feat_mat_dst)

    def reset_parameters(self, init_com_scale: OptTensor, init_affiliate_mat_src: OptTensor, init_affiliate_mat_dst: OptTensor,
                         init_feat_mat_src: OptTensor, init_feat_mat_dst: OptTensor):
        if init_com_scale is None:
            # feat_mat is inspired by torch.nn.modules.linear class Linear (as both are linear transformations)
            bound = 1 / math.sqrt(self.num_communities)
            init.uniform_(self.feat_mat_src, -bound, bound)
            init.uniform_(self.feat_mat_dst, -bound, bound)

            # com_scale is inspired by the bias of torch.nn.modules.linear class Linear
            init.uniform_(self.com_scale, -bound, bound)

            # _affiliate_mat
            init.uniform_(self._affiliate_mat_src, -4, 4)
            init.uniform_(self._affiliate_mat_dst, -4, 4)
        else:
            self.com_scale.data, self._affiliate_mat_src.data, self._affiliate_mat_dst.data =\
                get_init_com_scale_and_affiliate_mat(num_communities=self.num_communities,
                                                     init_com_scale=init_com_scale,
                                                     init_affiliate_mat_src=init_affiliate_mat_src,
                                                     init_affiliate_mat_dst=init_affiliate_mat_dst)
            self._affiliate_mat_src.data = inv_sigmoid(x=self._affiliate_mat_src.data)
            self._affiliate_mat_dst.data = inv_sigmoid(x=self._affiliate_mat_dst.data)
        if init_feat_mat_src is None:
            # feat_mat is inspired by torch.nn.modules.linear class Linear (as both are linear transformations)
            bound = 1 / math.sqrt(self.num_communities)
            init.uniform_(self.feat_mat_src, -bound, bound)
            init.uniform_(self.feat_mat_dst, -bound, bound)
        else:
            self.feat_mat_src.data = init_feat_mat_src
            self.feat_mat_dst.data = init_feat_mat_dst
        
        if self.encoder is not None:
            self.encoder.reset_parameters()

    @property
    def affiliate_mat_src(self) -> Tensor:
        return torch.sigmoid(self.scale_grad(self._affiliate_mat_src))
    
    @property
    def affiliate_mat_dst(self) -> Tensor:
        return torch.sigmoid(self.scale_grad(self._affiliate_mat_dst))
    
    @property
    def affiliate_src_times_scale(self) -> Tensor:
        return self.affiliate_mat_src * self.com_scale

    @property
    def affiliate_dst_times_scale(self) -> Tensor:
        return self.affiliate_mat_dst * self.com_scale

    def set_src_matrices_after_ibg_approx_training(self):
        self._inv_affiliate_mat_src = torch.linalg.pinv(self.affiliate_src_times_scale).detach()
        

    def set_dst_matrices_after_ibg_approx_training(self):
        self._inv_affiliate_mat_dst = torch.linalg.pinv(self.affiliate_dst_times_scale).detach()
        

    def set_matrices_for_refinement(self):
        self._inv_affiliate_mat_src = torch.linalg.pinv(self.affiliate_src_times_scale)
        self._inv_affiliate_mat_dst = torch.linalg.pinv(self.affiliate_dst_times_scale)

    
    def set_matrices_after_ibg_approx_training(self):
        self.set_src_matrices_after_ibg_approx_training()
        self.set_dst_matrices_after_ibg_approx_training()

    @property
    def inv_affiliate_mat_src(self) -> Tensor:
        assert self._inv_affiliate_mat_src is not None,\
            "Call set_matrices_after_ibg_approx_training before using inv_affiliate_mat"
        return self._inv_affiliate_mat_src.to(self.com_scale.device)
    
    @property
    def inv_affiliate_mat_dst(self) -> Tensor:
        assert self._inv_affiliate_mat_dst is not None,\
            "Call set_matrices_after_ibg_approx_training before using inv_affiliate_mat"
        return self._inv_affiliate_mat_dst.to(self.com_scale.device)

    def encode(self, x: Tensor) -> Tensor:
        if self.encoder is not None:
            return self.encoder(x)
        else:
            return x
