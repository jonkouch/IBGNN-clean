from typing import Any, Optional

import numpy as np
import torch

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_scipy_sparse_matrix

import scipy

# Similar to torch_geometric.transforms.add_positional_encoding


def add_node_attr(
    data: Data,
    value: Any,
    attr_name: Optional[str] = None,
) -> Data:
    if attr_name is None:
        if data.x is not None:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data


class AddLargestAbsEigenvecAndEigenval(BaseTransform):
    def __init__(
        self,
        k: int,
        is_undirected: bool = False,
        **kwargs: Any,
    ) -> None:
        self.k = k
        self.is_undirected = is_undirected
        self.kwargs = kwargs

    def forward(self, data: Data) -> Data:
        if not hasattr(data, 'x'):  # Knowledge graph
            edge_index = data.graph.train_edge_list[:2, :]
            num_nodes = data.num_entity
        elif data.x.dim() >= 3:  # spatio-temporal
            num_nodes = data.num_nodes
            edge_index = data.edge_index
        else:
            num_nodes = data.x.shape[0]
            edge_index = data.edge_index

        # process edge_index
        if self.k > num_nodes:
            self.k = num_nodes
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
        if self.k > num_nodes * 0.8:
            u, s, vt = scipy.linalg.svd(adj.todense())
        else:
            u, s, vt = scipy.sparse.linalg.svds(  # additional_classes: ignore
                adj,
                k=self.k,
                return_singular_vectors=True,
                **self.kwargs,
            )

        # changes
        largest_indices = np.abs(s).argsort()[::-1]

        eig_vals = np.real(s[largest_indices])
        left_vecs = np.real(u[:, largest_indices])
        right_vecs = np.real(vt.T[:, largest_indices])

        eig_vals = torch.from_numpy(eig_vals[:self.k])
        pe_left = torch.from_numpy(left_vecs[:, :self.k])
        pe_right = torch.from_numpy(right_vecs[:, :self.k])

        sign = -1 + 2 * torch.randint(0, 2, (self.k, ))
        pe_left *= sign
        pe_right *= sign
        
        if hasattr(data, 'graph'):
            setattr(data.graph, 'eigenvals', eig_vals)
            setattr(data.graph, 'eigenvecs_src', pe_left)
            setattr(data.graph, 'eigenvecs_dst', pe_right)
        else:
            data = add_node_attr(data, pe_left, attr_name='eigenvecs_src')
            data = add_node_attr(data, pe_right, attr_name='eigenvecs_dst')
            data = add_node_attr(data, eig_vals, attr_name='eigenvals')
        return data
