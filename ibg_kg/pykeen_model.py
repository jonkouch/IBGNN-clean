import torch.nn
from torch.nn import Parameter, init, functional as F
from pykeen.nn.modules import Interaction
from pykeen.models import ERModel
from pykeen.typing import FloatTensor, Hint



from collections.abc import Mapping
from typing import Any, ClassVar, Optional
from pykeen.constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from pykeen.typing import FloatTensor, Hint, Initializer, Normalizer

class ibgE(ERModel[FloatTensor, FloatTensor, FloatTensor]):

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE, rank=dict(type=int, low=2, high=2048, log=True)
    )

    def __init__(
        self,
        num_relations: int,
        num_communities: int = 16,
        embedding_dim: int = 100,
        one_hot: bool = False,
        normalize: bool = True,
        entity_initializer: Hint[Initializer] = None,
        entity_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        entity_normalizer: Hint[Normalizer] = None,
        entity_normalizer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_initializer: Hint[Initializer] = None,
        relation_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> None:
        r"""Initialize the model.

        :param entity_initializer: Entity initializer function. Defaults to None
        :param entity_initializer_kwargs: Keyword arguments to be used when calling the entity initializer
        :param entity_normalizer: Entity normalizer function. Defaults to None
        :param entity_normalizer_kwargs: Keyword arguments to be used when calling the entity normalizer
        :param relation_initializer: Relation initializer function. Defaults to None
        :param relation_initializer_kwargs: Keyword arguments to be used when calling the relation initializer
        :param kwargs: Remaining keyword arguments passed through to :class:`~pykeen.models.ERModel`.
        """
        super().__init__(
            interaction=IBGInteraction,
            interaction_kwargs=dict(num_relations=num_relations, num_communities=num_communities, hidden_dim=embedding_dim,
                                    one_hot=one_hot, normalize=normalize),
            entity_representations_kwargs=[
                # head representation
                dict(
                    shape=num_communities,
                    initializer=entity_initializer,
                    initializer_kwargs=entity_initializer_kwargs,
                    normalizer=entity_normalizer,
                    normalizer_kwargs=entity_normalizer_kwargs,
                ),
                # tail representation
                dict(
                    shape=num_communities,
                    initializer=entity_initializer,
                    initializer_kwargs=entity_initializer_kwargs,
                    normalizer=entity_normalizer,
                    normalizer_kwargs=entity_normalizer_kwargs,
                ),
            ],
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer,
                initializer_kwargs=relation_initializer_kwargs,
            ),
            **kwargs,
        )

class IBGInteraction(Interaction[FloatTensor, FloatTensor, FloatTensor]):

    entity_shape = ("k", "k")
    relation_shape = ("d",)
    _head_indices = (0,)
    _tail_indices = (1,)

    def __init__(self, num_relations: int, num_communities: int, hidden_dim: int, one_hot: bool = False, normalize: bool = True) -> None:
        super().__init__()
        # The weights of this MLP will be learned.
        self.com_scale = Parameter(torch.zeros((num_communities, hidden_dim)))
        self.num_relations = num_relations

        if one_hot:
            self.one_hot_relations = torch.eye(num_relations)
        else:
            self.one_hot_relations = None

        self.normalize = normalize



        self.reset_parameters()


    def reset_parameters(self):
        init.uniform(self.com_scale, -0.5, 0.5)


    def normalize_entities(self, h, r, t):
        if self.normalize:
            r = F.normalize(r, p=2, dim=-1)
        return F.sigmoid(h), r, F.sigmoid(t)

    def forward(self, h: FloatTensor, r: FloatTensor, t: FloatTensor) -> FloatTensor:
        h, r, t = self.normalize_entities(h, r, t)

        if t.dim() == 2:    # Training
            scores = self.predict_train_triplets(h, r, t)

        elif t.shape[1] == 1:   # predicting all possible heads
            scores = self.predict_head_scores(h, r, t)

        else:   # predicting all possible tails
            scores = self.predict_tail_scores(h, r, t)
        

        return scores

    def predict_train_triplets(self, h, r, t):
        """
        Standard triplets case
        """
        com_scale = self.com_scale.unsqueeze(1).transpose(0, 2)  # (D, 1, K)
        t = t.unsqueeze(2)  # (B, K, 1)
        h_com_emb = (h * com_scale).permute(1, 0, 2)     # (B, D, K)
        triplets_emb = torch.bmm(h_com_emb, t).squeeze()    # (B, D)
        triplets_distances = torch.sqrt(((r - triplets_emb) ** 2).sum(dim=-1))    # (B, Neg+1)
        return - triplets_distances


    def predict_head_scores(self, h, r, t):
        """
        Score all possible heads for a batch of rt pairs
        h: (1, Neg + 1, K)
        r: (B, 1, D)
        t: (B, 1, K)
        """
        h = h.squeeze()    # (B, K)
        com_scale = self.com_scale.unsqueeze(1).transpose(0, 2)  # (D, 1, K)

        head_com_emb = (h * com_scale).permute(1, 0, 2)     # (Neg+1, D, K)
        t = t.unsqueeze(2)  # (B, 1, 1, K)

        triplets_emb = (t * head_com_emb).sum(dim=-1)   # (B, Neg+1, D)

        distances = torch.sqrt(((r - triplets_emb) ** 2).sum(dim=-1))    # (B, N)
        return - distances


    def predict_tail_scores(self, h, r, t):
        """
        Score all possible trails for a batch of hr pairs
        h: (B, 1, K)
        r: (B, 1, D)
        t: (1, Neg + 1, K)
        """
        t = t.squeeze()    # (B, K)
        com_scale = self.com_scale.unsqueeze(1).transpose(0, 2)  # (D, 1, K)

        tail_com_emb = (com_scale * t).permute(1, 0, 2)     # (Neg+1, D, K)

        h = h.unsqueeze(2)  # (B, 1, 1, K)

        triplets_emb = (h * tail_com_emb).sum(dim=-1)    # (B, Neg+1, D)

        distances = torch.sqrt(((r - triplets_emb) ** 2).sum(dim=-1))    # (B, Neg+1)
        return - distances