import os.path
import torch
from torch_geometric.data import Data
import sys
import tqdm
import os.path as osp

from ibg_approximation.classes import IBGApproxTrainArgs
from ibg_approximation.model import IBGApproxModel
from ibg_approximation.utils import set_seed
from ibg_approximation.train import train_ibg_approximation
from helpers.constants import DECIMAL
from additional_classes.dataset import DataSetFamily


class IBGApproxTrainer(object):

    def __init__(self, train_args: IBGApproxTrainArgs, seed: int, device, exp_path: str):
        super().__init__()
        self.train_args = train_args
        self.seed = seed
        self.device = device
        self.exp_path = exp_path

    def train(self, model: IBGApproxModel, data: Data, dataset_family: DataSetFamily, load_epoch: int = 0, triton: bool = False, sparse_loss: bool = False):
        set_seed(seed=self.seed)
        ibg_approx_optimizer = torch.optim.Adam(model.parameters(), lr=self.train_args.lr)

        # load model
        if load_epoch > 0:
            best_loss, best_grahon_re = torch.load(osp.join(self.exp_path, 'loss.pt'))
            ibg_approx_optimizer.load_state_dict(torch.load(osp.join(self.exp_path, 'optimizer.pt')))
            model.load_state_dict(torch.load(osp.join(self.exp_path, 'model.pt')))

            print(f'Loading model at epochs {load_epoch}')
            print(f'Training for an additional {self.train_args.epochs} epochs')
            print(f'Loaded GS;loss={round(best_loss, DECIMAL)};g_re={round(best_grahon_re, DECIMAL)}')

            total_epochs = int(self.train_args.epochs) + int(load_epoch)
            self.exp_path = self.exp_path.replace(f'{load_epoch}', f'{total_epochs}')

        # train model
        with tqdm.tqdm(total=self.train_args.epochs, file=sys.stdout) as pbar:
            best_loss, best_grahon_re, model = \
                train_ibg_approximation(data=data, model=model, optimizer=ibg_approx_optimizer,
                                        train_args=self.train_args,
                                        pbar=pbar, device=self.device,
                                        dataset_family=dataset_family, triton=triton, sparse_loss=sparse_loss)

        # Save
        print('Saving Decomp Model')
        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)

        torch.save(model.state_dict(), osp.join(self.exp_path, 'model.pt'))
        torch.save((best_loss, best_grahon_re), osp.join(self.exp_path, 'loss.pt'))
        torch.save(ibg_approx_optimizer.state_dict(), osp.join(self.exp_path, 'optimizer.pt'))

        # print
        print(f'Final GS;loss={round(best_loss, DECIMAL)};g_re={round(best_grahon_re, DECIMAL)}')
        return model
