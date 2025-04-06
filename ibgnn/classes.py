from typing import NamedTuple, Optional
from torch.nn import Module, Linear
from enum import Enum, auto

from additional_classes.activation import ActivationType
from ibgnn.set_layers import DeepSets_fancy
from ibgnn.modules import basic, MAT, deepsets


class NNTypeNode(Enum):
    # community-node and node types
    linear = auto()
    deepsets = auto()
    deepsets_fancy = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return NNTypeNode[s]
        except KeyError:
            raise ValueError()



class NNTypeCom(Enum):
    """
        an object for the different activation types for community level
    """
    # community types
    unconstrained = auto()
    basic = auto()
    deepsets = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return NNTypeCom[s]
        except KeyError:
            raise ValueError()

    def is_mat(self) -> bool:
        return self is NNTypeCom.unconstrained


class NNArgs(NamedTuple):
    num_layers: int
    
    dropout: float
    skip_com: bool
    skip_node: bool
    act_type: ActivationType
    num_communities: int
    normalize: bool
    jump: Optional[str]
    layer_norm: bool

    ibgnn_com_type: NNTypeCom
    com_layers: int

    ibgnn_com_node_type: NNTypeNode
    com_node_layers: int

    ibgnn_node_type: NNTypeNode
    node_layers: int

    pool_type: str

    def get_com_model(self, dim: int) -> Optional[Module]:
        if self.ibgnn_com_type is NNTypeCom.unconstrained:
            return MAT(num_communities=self.num_communities, out_dim=dim)
        elif self.ibgnn_com_type is NNTypeCom.basic:
            return basic(in_dim=dim, hidden_dim=dim, out_dim=dim,
                       num_layers=self.com_layers, dropout=self.dropout,
                       skip_com=self.skip_com, act_type=self.act_type)
        elif self.ibgnn_com_type is NNTypeCom.deepsets:
            return deepsets(in_dim=dim, hidden_dim=dim, out_dim=dim,
                         num_layers=self.com_layers, dropout=self.dropout,
                         skip_com=self.skip_com, act_type=self.act_type,
                         pool=self.pool_type)
        else:
            raise ValueError(f'NNType {self.ibgnn_com_type.name} not supported')




    def get_com_node_model(self, in_dim: int, hidden_dim: int, out_dim: int) -> Optional[Module]:
        if self.ibgnn_com_node_type is NNTypeNode.linear:
            return Linear(in_features=in_dim, out_features=out_dim)
        elif self.ibgnn_com_node_type is NNTypeNode.deepsets:
            return deepsets(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                         num_layers=self.node_layers, dropout=self.dropout,
                         skip_com=False, act_type=self.act_type,
                         pool=self.pool_type)
        elif self.ibgnn_com_node_type is NNTypeNode.deepsets_fancy:
            return DeepSets_fancy(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=self.com_node_layers,
                                  pool=self.pool_type)
        else:
            raise ValueError(f'NNType {self.ibgnn_com_node_type.name} not supported')


    def get_node_model(self, in_dim: int, hidden_dim: int, out_dim: int) -> Optional[Module]:
        if self.ibgnn_node_type is NNTypeNode.linear:
            return Linear(in_features=in_dim, out_features=out_dim)
        elif self.ibgnn_node_type is NNTypeNode.deepsets:
            return deepsets(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                         num_layers=self.node_layers, dropout=self.dropout,
                         skip_com=False, act_type=self.act_type,
                         pool=self.pool_type)
        elif self.ibgnn_node_type is NNTypeNode.deepsets_fancy:
            return DeepSets_fancy(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                                  num_layers=self.node_layers, pool=self.pool_type)
        else:
            raise ValueError(f'NNType {self.ibgnn_node_type.name} not supported')



class IBGNNArgs(NamedTuple):
    encoded_dim: int
    hidden_dim: int
    out_dim: int
    ibgnn_args: NNArgs

    def get_args_str(self) -> str:
        ibgnn_args = self.ibgnn_args
        return f'{ibgnn_args.num_layers}_{self.hidden_dim}_{ibgnn_args.ibgnn_com_type.name}_{ibgnn_args.com_layers}_{ibgnn_args.ibgnn_com_node_type.name}_{ibgnn_args.com_node_layers}_{ibgnn_args.ibgnn_node_type.name}_{ibgnn_args.node_layers}_{ibgnn_args.dropout}_{ibgnn_args.skip_com}_{ibgnn_args.skip_node}'


class NNTrainArgs(NamedTuple):
    epochs: int
    lr: float
    patience: int
    refine_ibg_epoch: int
    refine_scale: float
