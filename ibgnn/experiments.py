from argparse import Namespace
import torch
import os.path as osp


from ibg_approximation.classes import IBGApproxArgs, IBGApproxTrainArgs
from ibg_approximation.utils import set_seed, exp_path, get_device
from helpers.constants import DECIMAL, TIME_STEPS
from ibgnn.classes import NNArgs, NNTrainArgs, IBGNNArgs
from ibgnn.trainer import NNTrainer
from ibg_approximation.model import IBGApproxModel


class NNExperiment(object):
    def __init__(self, args: Namespace):
        super().__init__()
        for arg in vars(args):
            value_arg = getattr(args, arg)
            print(f"{arg}: {value_arg}")
            self.__setattr__(arg, value_arg)

        self.device = torch.device(get_device())
        set_seed(seed=0)

        self.metric_type = self.dataset_type.get_metric_type()
        self.task_loss = self.metric_type.get_task_loss()
        self.ibg_approx_train_args = IBGApproxTrainArgs(epochs=self.ibg_approx_epochs, lr=self.ibg_approx_lr, 
                                                     loss_scale=self.loss_scale, 
                                                     cut_norm=self.cut_norm, node_drop_ratio=self.node_drop_ratio,
                                                     is_undirected=self.is_undirected,
                                                     sparse_scale=self.sparse_scale)
        assert not self.dataset_type.get_family().is_spatio_temporal(), \
            "To run spatio_temporal datasets use ibgnn_spatio_temporal/run_static_Graph.py"

    def run(self):
        # load data
        data = self.dataset_type.load(is_ibgnn=True, is_undirected=self.is_undirected, add_eigen=self.add_eigen)
        act_type = self.dataset_type.activation_type()

        # load ibg_approximation args
        out_dim = self.metric_type.get_out_dim(data=data)
        time_steps = TIME_STEPS if self.dataset_type.get_family().is_spatio_temporal() else 1
        num_nodes, in_dim = data.x.shape[-2], data.x.shape[-1]
        ibg_approx_model_args = IBGApproxArgs(num_communities=self.num_communities, encode_dim=self.encode_dim,
                                           num_nodes=num_nodes, in_dim=in_dim, add_eigen=self.add_eigen,
                                           node_drop_ratio=self.node_drop_ratio, init_affiliate_mat_src=None,
                                           init_affiliate_mat_dst=None, init_com_scale=None, init_feat_mat_src=None,
                                           init_feat_mat_dst=None, time_steps=time_steps, is_kg=self.dataset_type.get_family().is_kg())
        path = exp_path(dataset_name=self.dataset_type.name, ibg_approx_args=ibg_approx_model_args,
                        ibg_approx_train_args=self.ibg_approx_train_args, seed=0, sparse=self.sparse, sparse_scale=self.sparse_scale)

        # load ibg_approximation model
        print('Loading IBG Approximation Model')
        ibg_approx_model = IBGApproxModel(model_args=ibg_approx_model_args, is_ibgnn=True).to(self.device)
        state_dict = torch.load(osp.join(path, 'model.pt'))
        if 'feat_mat_src' in state_dict:
            del state_dict['feat_mat_src']
            del state_dict['feat_mat_dst']
        if hasattr(ibg_approx_model, 'feat_mat_src'):
            delattr(ibg_approx_model, 'feat_mat_src')
            delattr(ibg_approx_model, 'feat_mat_dst')

            
        ibg_approx_model.load_state_dict(state_dict)
        best_loss, best_grahon_re = torch.load(osp.join(path, 'loss.pt'))
        print(f'Final GS;loss={round(best_loss, DECIMAL)};g_re={round(best_grahon_re, DECIMAL)}')

        # load ibgnn args
        ibgnn_args = NNArgs(dropout=self.dropout, skip_com=self.skip_com, skip_node=self.skip_node, act_type=act_type,
                            num_layers=self.num_layers, num_communities=self.num_communities,
                            normalize=self.normalize, jump=self.jump, layer_norm=self.layer_norm,
                            ibgnn_com_type=self.ibgnn_com_type, com_layers=self.com_layers,
                            ibgnn_com_node_type=self.ibgnn_com_node_type, com_node_layers=self.com_node_layers,
                            ibgnn_node_type=self.ibgnn_node_type, node_layers=self.node_layers,
                            pool_type=self.pool_type)
        
        model_args = IBGNNArgs(encoded_dim=ibg_approx_model_args.encoded_dim, hidden_dim=self.hidden_dim,
                              out_dim=out_dim,
                              ibgnn_args=ibgnn_args)
        train_args = NNTrainArgs(epochs=self.epochs, lr=self.lr, patience=self.patience, refine_ibg_epoch=self.refine_ibg_epoch, refine_scale=self.refine_scale)

        # train
        trainer = NNTrainer(model_args=model_args, train_args=train_args, seed=self.seed, device=self.device)
        trainer.train_and_test_splits(dataset_type=self.dataset_type, data=data, ibg_approx_model=ibg_approx_model, path=path)
