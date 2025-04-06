from enum import Enum, auto
from torch_geometric.data import Data, download_url
import torch_geometric.transforms as T
from torch_geometric.datasets import LINKXDataset, Flickr, Reddit
from dataset_classes.wikipedia import WikipediaNetwork
from dataset_classes.hetero import HeterophilousGraphDataset
from dataset_classes.citation import CitationFull
from ogb.nodeproppred import PygNodePropPredDataset
import os.path as osp
import os
import torch
from typing import List
import numpy as np
from pykeen.datasets import UMLS, Kinships


from helpers.constants import ROOT_DIR, MAX_NUM_COMMUNITIES
from additional_classes.metrics import MetricType
from additional_classes.activation import ActivationType
from dataset_classes.communities import Communities
from dataset_classes.snap import SNAPDataset
from helpers.eigen_transform import AddLargestAbsEigenvecAndEigenval
from dataset_classes.spatio_temporal import ibg_approx_load
from dataset_classes.snap_patents import load_snap_patents_mat

from dataset_classes.utils import set_uniform_train_val_test_split, get_mask, even_quantile_labels, process_pykeen_dataset


class DataSetFamily(Enum):
    synthetic = auto()
    heterophilic = auto()
    wiki = auto()
    link = auto()
    snap_patents = auto()
    spatio_temporal = auto()
    homophilic = auto()
    knowledge_graph = auto()
    arxiv = auto()
    sgd = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return DataSetFamily[s]
        except KeyError:
            raise ValueError()

    def is_spatio_temporal(self) -> bool:
        return self is DataSetFamily.spatio_temporal

    def is_kg(self) -> bool:
        return self is DataSetFamily.knowledge_graph


class DataSetType(Enum):
    """
        an object for the different dataset_classes
    """
    # communities
    communities = auto()

    # heterophilic
    tolokers = auto()
    roman_empire = auto()
    minesweeper = auto()
    amazon_ratings = auto()
    questions = auto()


    # wiki
    squirrel = auto()
    chameleon = auto()

    # link
    twitch_gamers = auto()
    penn94 = auto()

    # snap_patents
    snap_patents = auto()

    # homophilic
    citeseer = auto()
    cora_ml = auto()

    # arxiv
    ogbn_arxiv = auto()
    arxiv_year = auto()

    # knowledge graph
    umls = auto()
    kinships = auto()

    # sgd
    flickr = auto()
    reddit = auto()

    # spatio temporal
    air = auto()
    bay = auto()
    la = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return DataSetType[s]
        except KeyError:
            raise ValueError()
        
    def get_family(self) -> DataSetFamily:
        if self in [DataSetType.communities]:
            return DataSetFamily.synthetic
        elif self in [DataSetType.tolokers, DataSetType.roman_empire, DataSetType.minesweeper, DataSetType.amazon_ratings, DataSetType.questions]:
            return DataSetFamily.heterophilic
        elif self in [DataSetType.squirrel, DataSetType.chameleon]:
            return DataSetFamily.wiki
        elif self in [DataSetType.penn94, DataSetType.twitch_gamers]:
            return DataSetFamily.link
        elif self is DataSetType.snap_patents:
            return DataSetFamily.snap_patents
        elif self in [DataSetType.citeseer, DataSetType.cora_ml]:
            return DataSetFamily.homophilic
        elif self in [DataSetType.ogbn_arxiv, DataSetType.arxiv_year]:
            return DataSetFamily.arxiv
        elif self in [DataSetType.flickr, DataSetType.reddit]:
            return DataSetFamily.sgd
        elif self in [DataSetType.air, DataSetType.la, DataSetType.bay]:
            return DataSetFamily.spatio_temporal
        elif self in [DataSetType.kinships, DataSetType.umls]:
            return DataSetFamily.knowledge_graph
        else:
            raise ValueError(f'DataSetType {self.name} not supported in dataloader')

    def get_folds(self) -> List[int]:
        if self.get_family() in [DataSetFamily.synthetic, DataSetFamily.spatio_temporal, DataSetFamily.knowledge_graph, DataSetFamily.arxiv, DataSetFamily.sgd]:
            return list(range(1))
        elif self.get_family() in [DataSetFamily.heterophilic, DataSetFamily.wiki]:
            return list(range(10))
            # return list(range(1))
        elif self.get_family() in [DataSetFamily.link, DataSetFamily.snap_patents, DataSetFamily.homophilic]:
            return list(range(5))
        else:
            raise ValueError(f'DataSetType {self.name} not supported in dataloader')

    def load(self, is_ibgnn: bool, is_undirected: bool, add_eigen: bool) -> List[Data]:
        pre_transform = None
        if not (is_ibgnn and self.get_family() is DataSetFamily.spatio_temporal):
            transforms = []
            if is_undirected:
                transforms.append(T.ToUndirected())
            if add_eigen:
                transforms.append(AddLargestAbsEigenvecAndEigenval(k=MAX_NUM_COMMUNITIES))
            pre_transform = T.Compose(transforms)
        root = osp.join(ROOT_DIR, f'datasets')        
        if self is DataSetType.communities:
            dataset = Communities()
            dataset._data = pre_transform(data=dataset._data)
        elif self.get_family() is DataSetFamily.heterophilic:
            name = self.name.replace('_', '-').capitalize()
            dataset = HeterophilousGraphDataset(root=root, name=name, pre_transform=pre_transform)
        elif self.get_family() is DataSetFamily.wiki:
            # Same as Geom-GCN
            # transform=T.NormalizeFeatures()
            transform = None
            dataset = WikipediaNetwork(root=root, name=self.name, geom_gcn_preprocess=True,
                                       pre_transform=pre_transform, transform=transform)
        elif self is DataSetType.twitch_gamers:
            dataset = SNAPDataset(root=root, name=self.name, pre_transform=pre_transform)
            dataset._data.y = torch.from_numpy(dataset._data.y)
        elif self is DataSetType.penn94:
            dataset = LINKXDataset(root=root, name=self.name, pre_transform=pre_transform)
        elif self is DataSetType.snap_patents:
            dataset = load_snap_patents_mat(n_classes=5, root=root, pre_transform=pre_transform)
        elif self.get_family() is DataSetFamily.spatio_temporal: 
            dataset = ibg_approx_load(dataset_name=self.name, pre_transform=pre_transform, symmetric=is_undirected)
        elif self.get_family() is DataSetFamily.homophilic:
            dataset = CitationFull(root=root, name=self.name, pre_transform=pre_transform)
        elif self is DataSetType.flickr:
            root = osp.join(root, 'flickr')
            dataset = Flickr(root=root, pre_transform=pre_transform, transform=T.NormalizeFeatures())
        elif self is DataSetType.reddit:
            root = osp.join(root, 'reddit')
            dataset = Reddit(root=root, pre_transform=pre_transform, transform=T.NormalizeFeatures())
        elif self is DataSetType.ogbn_arxiv:
            dataset = PygNodePropPredDataset(name=self.name.replace('_', '-'), root=root, pre_transform=pre_transform)
        elif self is DataSetType.arxiv_year:
            # arxiv-year uses the same graph and features as ogbn-arxiv, but with different labels
            dataset = PygNodePropPredDataset(name="ogbn-arxiv", pre_transform=pre_transform, root=root)
            y = even_quantile_labels(dataset._data.node_year.flatten().numpy(), nclasses=5, verbose=False)
            dataset._data.y = torch.as_tensor(y).reshape(-1, 1)
            # Tran, val and test masks are required during preprocessing. Setting them here to dummy values as
            # they are overwritten later for this dataset (see get_dataset_split function below)
            dataset._data.train_mask, dataset._data.val_mask, dataset._data.test_mask = 0, 0, 0
            # Create directory for this dataset
            os.makedirs(osp.join(root, self.name.replace("-", "_"), "raw"), exist_ok=True)
        elif self is DataSetType.kinships:
            dataset = Kinships()
            dataset = process_pykeen_dataset(dataset)

        elif self is DataSetType.umls:
            dataset = UMLS()
            dataset = process_pykeen_dataset(dataset)
        else:
            raise ValueError(f'DataSetType {self.name} not supported in dataloader')
        
        if self.get_family() not in [DataSetFamily.snap_patents, DataSetFamily.knowledge_graph]:
            data = dataset[0]
        else:
            data = dataset
              
        return data

    def select_fold(self, data: Data, num_fold: int) -> Data:
        if self.get_family() in [DataSetFamily.synthetic, DataSetFamily.knowledge_graph, DataSetFamily.sgd]:
            return data
        elif self is DataSetType.ogbn_arxiv:
            split_idx = data.get_idx_split()
            train_idx = split_idx['train']
            val_idx = split_idx['valid']
            test_idx = split_idx['test']
            train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True
            setattr(data, 'train_mask', train_mask)
            setattr(data, 'val_mask', val_mask)
            setattr(data, 'test_mask', test_mask)
            return data

        elif self is DataSetType.arxiv_year:
            # Datasets from https://arxiv.org/pdf/2110.14446.pdf have five splits stored
            # in https://github.com/CUAI/Non-Homophily-Large-Scale/tree/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/data/splits
            num_nodes = data["y"].shape[0]
            github_url = f"https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/splits/"
            name = self.name.replace("_", "-")
            split_file_name = f"{name}-splits.npy"
            root = osp.join(ROOT_DIR, f'datasets')
            
            local_dir = osp.join(root, self.name.replace("-", "_"), "raw")

            download_url(osp.join(github_url, split_file_name), local_dir, log=False)
            splits = np.load(osp.join(local_dir, split_file_name), allow_pickle=True)
            split_idx = splits[num_fold % len(splits)]

            train_mask = get_mask(split_idx["train"], num_nodes)
            val_mask = get_mask(split_idx["valid"], num_nodes)
            test_mask = get_mask(split_idx["test"], num_nodes)
            setattr(data, 'train_mask', train_mask)
            setattr(data, 'val_mask', val_mask)
            setattr(data, 'test_mask', test_mask)
            return data
        
        elif self.get_family() is DataSetFamily.wiki:
            # geom-gcn splits (60, 20, 20)
            device = data.x.device
            fold_path = osp.join(ROOT_DIR, f'folds/{self.name}_split_0.6_0.2_{num_fold}.npz')
            with np.load(fold_path) as folds_file:
                train_mask = torch.tensor(folds_file['train_mask'], dtype=torch.bool, device=device)
                val_mask = torch.tensor(folds_file['val_mask'], dtype=torch.bool, device=device)
                test_mask = torch.tensor(folds_file['test_mask'], dtype=torch.bool, device=device)

            setattr(data, 'train_mask', train_mask)
            setattr(data, 'val_mask', val_mask)
            setattr(data, 'test_mask', test_mask)

            if hasattr(data, 'non_valid_samples'):
                data.train_mask[data.non_valid_samples] = False
                data.test_mask[data.non_valid_samples] = False
                data.val_mask[data.non_valid_samples] = False
            return data
        
        elif (self.get_family() is DataSetFamily.heterophilic) or (self is DataSetType.penn94):
            data_copy = data.clone()
            data_copy.train_mask = data_copy.train_mask[:, num_fold]
            data_copy.val_mask = data_copy.val_mask[:, num_fold]
            data_copy.test_mask = data_copy.test_mask[:, num_fold]
            return data_copy
        
        elif self in [DataSetType.twitch_gamers, DataSetType.snap_patents]:
            name = self.name.replace('_', '-')
            fold_path = osp.join(ROOT_DIR, f'folds/{name}-splits.npy')
            folds_file = np.load(fold_path, allow_pickle=True)
            setattr(data, 'train_mask', torch.as_tensor(folds_file[num_fold]['train']))
            setattr(data, 'val_mask', torch.as_tensor(folds_file[num_fold]['valid']))
            setattr(data, 'test_mask', torch.as_tensor(folds_file[num_fold]['test']))
            return data
        elif self.get_family() is DataSetFamily.homophilic:
            data = set_uniform_train_val_test_split(num_fold, data, train_ratio=0.5, val_ratio=0.25)
            return data
        else:
            raise ValueError(f'DataSetType {self.name} not supported in select_fold')

    def get_metric_type(self) -> MetricType:
        if self is DataSetType.reddit:
            return MetricType.MicroF1
        if self.get_family() in [DataSetFamily.wiki, DataSetFamily.link, DataSetFamily.snap_patents, DataSetFamily.homophilic, DataSetFamily.arxiv, DataSetFamily.sgd] \
                                or self in [DataSetType.roman_empire, DataSetType.amazon_ratings]:
            return MetricType.ACCURACY
        elif self in [DataSetType.tolokers, DataSetType.minesweeper, DataSetType.questions]:
            return MetricType.AUC_ROC
        elif self.get_family() in [DataSetFamily.synthetic, DataSetFamily.spatio_temporal]:
            return MetricType.MAE
        elif self.get_family() is DataSetFamily.knowledge_graph:
            return MetricType.MRR  # explore
        else:
            raise ValueError(f'DataSetType {self.name} not supported in dataloader')

    def is_communities(self) -> bool:
        return self.get_family() is DataSetFamily.synthetic

    def activation_type(self) -> ActivationType:
        if self.get_family() is DataSetFamily.heterophilic:
            return ActivationType.GELU
        else:
            return ActivationType.RELU
