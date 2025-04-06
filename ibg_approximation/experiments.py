from argparse import Namespace
import torch
import math

from ibg_approximation.classes import IBGApproxArgs, IBGApproxTrainArgs
from ibg_approximation.utils import set_seed, exp_path, get_init_com_scale_and_affiliate_mat
from helpers.constants import TIME_STEPS
from ibg_approximation.trainer import IBGApproxTrainer
from ibg_approximation.model import IBGApproxModel



class IBGApproxExperiment(object):
    def __init__(self, args: Namespace):
        super().__init__()

        for arg in vars(args):
            value_arg = getattr(args, arg)
            print(f"{arg}: {value_arg}")
            self.__setattr__(arg, value_arg)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_seed(seed=0)

        self.dataset_family = self.dataset_type.get_family()
        self.train_args = IBGApproxTrainArgs(epochs=self.ibg_approx_epochs, lr=self.ibg_approx_lr,
                                             loss_scale=None if self.dataset_family.is_kg() else self.loss_scale,
                                             cut_norm=self.cut_norm,
                                             node_drop_ratio=self.node_drop_ratio,
                                             is_undirected=self.is_undirected,
                                             sparse_scale=self.sparse_scale)
        assert not (self.dataset_family.is_spatio_temporal() and self.node_drop_ratio > 0), \
            f'Spatio-temporal datasets do not work with node_drop_ratio'

    def run(self):
        # load data
        data = self.dataset_type.load(is_ibgnn=False, is_undirected=self.is_undirected,
                                      add_eigen=self.add_eigen)
        if not self.dataset_family.is_kg():
            delattr(data, 'train_mask')
            delattr(data, 'val_mask')
            delattr(data, 'test_mask')

        # initialize model
        init_com_scale, init_affiliate_mat_src, init_affiliate_mat_dst = None, None, None
        init_feat_mat_src, init_feat_mat_dst = None, None
        if self.add_eigen:
            assert hasattr(data, 'eigenvecs_src') and hasattr(data,
                                                              'eigenvecs_dst'), f'No PE found, re-download the data!'

            init_com_scale = data.eigenvals[:math.ceil(self.num_communities / 4)]
            init_affiliate_mat_src = data.eigenvecs_src[:, :math.ceil(self.num_communities / 4)]
            init_affiliate_mat_dst = data.eigenvecs_dst[:, :math.ceil(self.num_communities / 4)]

            
            full_com_scale, full_affiliate_mat_src, full_affiliate_mat_dst = \
                get_init_com_scale_and_affiliate_mat(num_communities=self.num_communities,
                                                    init_com_scale=init_com_scale,
                                                    init_affiliate_mat_src=init_affiliate_mat_src,
                                                    init_affiliate_mat_dst=init_affiliate_mat_dst)





            if self.encode_dim == 0:
                if self.dataset_family.is_spatio_temporal():
                    init_feat_mat_src = torch.linalg.pinv(full_affiliate_mat_src * full_com_scale).unsqueeze(
                        dim=0) @ data.x
                    init_feat_mat_dst = torch.linalg.pinv(full_affiliate_mat_dst * full_com_scale).unsqueeze(
                        dim=0) @ data.x
                elif not self.dataset_family.is_kg():
                    init_feat_mat_src = torch.linalg.pinv(full_affiliate_mat_src * full_com_scale) @ data.x
                    init_feat_mat_dst = torch.linalg.pinv(full_affiliate_mat_dst * full_com_scale) @ data.x
        if hasattr(data, 'eigenvals'):
            delattr(data, 'eigenvals')
            delattr(data, 'eigenvecs_src')
            delattr(data, 'eigenvecs_dst')

        # load args
        time_steps = TIME_STEPS if self.dataset_family.is_spatio_temporal() else 1
        if self.dataset_family.is_kg():
            num_nodes, in_dim = data.num_node, data.num_relation
            in_dim = in_dim + (self.num_hop > 1)
        else:
            num_nodes, in_dim = data.x.shape[-2], data.x.shape[-1]

        model_args = IBGApproxArgs(num_communities=self.num_communities, encode_dim=self.encode_dim,
                                   num_nodes=num_nodes, in_dim=in_dim, add_eigen=self.add_eigen,
                                   node_drop_ratio=self.node_drop_ratio,
                                   init_affiliate_mat_src=init_affiliate_mat_src,
                                   init_affiliate_mat_dst=init_affiliate_mat_dst,
                                   init_com_scale=init_com_scale, init_feat_mat_src=init_feat_mat_src,
                                   init_feat_mat_dst=init_feat_mat_dst, time_steps=time_steps, 
                                   is_kg=self.dataset_family.is_kg())
        model = IBGApproxModel(model_args=model_args, is_ibgnn=False).to(device=self.device)

        # train
        path = exp_path(dataset_name=self.dataset_type.name, ibg_approx_args=model_args,
                        ibg_approx_train_args=self.train_args, seed=0, load_epoch=self.load_epoch,
                        sparse=self.sparse, sparse_scale=self.sparse_scale)
        trainer = IBGApproxTrainer(train_args=self.train_args, seed=0,
                                   device=self.device, exp_path=path)
        model = trainer.train(model=model, data=data, dataset_family=self.dataset_family, triton=self.triton, sparse_loss=self.sparse)
