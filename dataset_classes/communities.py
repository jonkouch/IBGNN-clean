import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.nn.conv.simple_conv import SimpleConv



# topology global vars
COMMUNITY_SIZE = 200
NUM_COMMUNITIES = 3
# feature global vars
NUM_MP = 3
ADJ_NOISE_RATIO = 0.7
FEAT_NOISE_RATIO = 0
class Communities(InMemoryDataset):
    def __init__(self):
        super().__init__()
        self.num_communities = NUM_COMMUNITIES
        self.num_mp = NUM_MP
        self.community_size = COMMUNITY_SIZE
        self.adj_noise_ratio = ADJ_NOISE_RATIO
        self.feat_noise_ratio = FEAT_NOISE_RATIO
        self._data = self.create_data()
    def gett(self, idx: int) -> Data:
        return self._data
    def create_communities_graph(self):
        com_scale = torch.arange(start=1, end=self.num_communities + 1) * 0.1 - 0.2
        com_scale[0] = 1
        com_scale[1] = 1
        com_scale[2] = 1
        # num_nodes = 1300
        # block1 = 400
        # block2 = 300
        # block3 = 200
        # block4 = 400
        # total = block1+block2+block3+block4
        # community_start_src = torch.tensor([0, block1+block2, block1])
        # community_end_src = torch.tensor([block1+block2, total, block1+block2])
        # community_start_dst = torch.tensor([block1, 0, block1+block2])
        # community_end_dst = torch.tensor([total, block1+block2, total])

        num_nodes = 1300
        community_start_src = torch.tensor([0, 400, 700])
        community_end_src = torch.tensor([400, 700, 1300])
        community_start_dst = torch.tensor([400, 400, 0])
        community_end_dst = torch.tensor([1300, 700, 700])
        # create adj mat
        adj_mat = torch.zeros(size=(1300, 1300))
        affiliate_mat_src = []
        affiliate_mat_dst = []

        for community_idx in range(self.num_communities):
            affiliate_vec_src = torch.zeros(size=(num_nodes, 1))
            affiliate_vec_dst = torch.zeros(size=(num_nodes, 1))
            # fill source
            start_src = community_start_src[community_idx]
            end_src = community_end_src[community_idx]
            if start_src < end_src:
                affiliate_vec_src[start_src: end_src, 0] = 1.0
            else:
                affiliate_vec_src[start_src:, 0] = 1.0
                affiliate_vec_src[:end_src, 0] = 1.0
            # fill destination
            start_dst = community_start_dst[community_idx]
            end_dst = community_end_dst[community_idx]
            if start_dst < end_dst:
                affiliate_vec_dst[start_dst: end_dst, 0] = 1.0
            else:
                affiliate_vec_dst[start_dst:, 0] = 1.0
            if community_idx == 1:
                affiliate_vec_dst[900: 1300, 0] = 1.0
            adj_mat += (affiliate_vec_src * com_scale[community_idx]) @ affiliate_vec_dst.T
            affiliate_mat_src.append(affiliate_vec_src.squeeze(dim=1))
            affiliate_mat_dst.append(affiliate_vec_dst.squeeze(dim=1))


        # makes sure adj is normalized
        norm_com_scale = (com_scale - adj_mat.min()) / (adj_mat.max() - adj_mat.min())
        adj_mat = (adj_mat - adj_mat.min()) / (adj_mat.max() - adj_mat.min())
        affiliate_mat_src = torch.stack(affiliate_mat_src, dim=1)
        affiliate_mat_dst = torch.stack(affiliate_mat_dst, dim=1)
        # add noise to adj where it is 1
        adj_clone = adj_mat.clone() * (1 - self.adj_noise_ratio)
        adj_clone = torch.bernoulli(adj_clone)
        adj_mat[adj_clone == 0] = 0
        # save as directed graph
        edge_index = dense_to_sparse(adj=adj_mat)[0]
        return edge_index, norm_com_scale, affiliate_mat_src, affiliate_mat_dst


    def create_data(self) -> Data:
        edge_index, norm_com_scale, affiliate_mat_src, affiliate_mat_dst = self.create_communities_graph()
        # regression target - the community scale of each node
        y = 10 * affiliate_mat_src * norm_com_scale  # (N, K)
        # add noise to features
        x = y.clone()
        std_x = torch.std(x, dim=0)  # (K,)
        x = x + self.feat_noise_ratio * std_x.unsqueeze(dim=0) * torch.randn(size=x.size())  # (N, K)
        # propagate features
        simple_conv = SimpleConv(aggr='mean')
        for _ in range(self.num_mp):
            x = simple_conv(x=x, edge_index=edge_index)
        # create masks for train/val/test
        num_nodes = x.shape[0]
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[::3] = 1
        val_mask[1::3] = 1
        test_mask[2::3] = 1
        return Data(x=x, edge_index=edge_index.long(), y=y,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, com_scale=norm_com_scale)
    
if __name__ == '__main__':
    dataset = Communities()