import torch
from torch import Tensor
from torch.nn import Module, Linear, Sequential, ReLU, GELU



class DeepSets_fancy(Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, pool: str = 'mean'):
        super().__init__()

        set_layers = []
        # set_layers.append(ReLU())
        set_layers.append(Linear(in_dim, hidden_dim))
        

        node_layers = []
        node_layers.append(ReLU())
        node_layers.append(Linear(in_dim, hidden_dim))

        pool_layers = []
        # pool_layers.append(ReLU())
        pool_layers.append(Linear(out_dim, out_dim))        

        for _ in range(max(0, num_layers - 2)):
            set_layers.append(ReLU())
            set_layers.append(Linear(hidden_dim, hidden_dim))
            node_layers.append(ReLU())
            node_layers.append(Linear(hidden_dim, hidden_dim))
            pool_layers.append(ReLU())
            pool_layers.append(Linear(out_dim, out_dim))
        
        set_layers.append(ReLU())
        set_layers.append(Linear(hidden_dim, out_dim))
        node_layers.append(ReLU())
        node_layers.append(Linear(hidden_dim, out_dim))
            
        if pool == 'mean':
            self.pool = lambda x: torch.mean(x, dim=0, keepdim=True)
        elif pool == 'max':
            self.pool = lambda x: torch.max(x, dim=0, keepdim=True)[0]

        self.set_transform = Sequential(*set_layers)
        self.element_transform = Sequential(*node_layers)
        self.pool_transform = Sequential(*pool_layers)


    def forward(self, x: Tensor) -> Tensor:
        return self.pool_transform(self.pool(self.set_transform(x))) + self.element_transform(x)



class DeepSets(Module):
    def __init__(self, in_dim: int, out_dim: int, pool: str = 'mean'):
        super().__init__()

        self.set_trans = Linear(in_dim, out_dim)
        self.element_trans = Linear(in_dim, out_dim)
        self.relu = ReLU()
       
        if pool == 'mean':
            self.pool = lambda x: torch.mean(x, dim=0, keepdim=True)
        elif pool == 'max':
            self.pool = lambda x: torch.max(x, dim=0, keepdim=True)[0]
        elif pool == 'sum':
            self.pool = lambda x: torch.sum(x, dim=0, keepdim=True)
        else:
            raise ValueError(f'Pool type {pool} not supported')


    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.set_trans(x)) + self.element_trans(x)
