from os import path as osp
from argparse import Namespace
import torch

from ibg_kg.parser import ibg_approx_parse_arguments
from ibg_kg.utils import exp_path
from ibg_kg.pykeen_model import ibgE

from helpers.constants import ROOT_DIR

from pykeen.pipeline import pipeline
from pykeen.datasets import Kinships, UMLS



def get_dataset(name: str):
    if name == "kinships":
        return Kinships()
    elif name == "umls":
        return UMLS()
    else:
        raise ValueError(f"Unknown dataset: {name}")


class KGExperiment(object):
    def __init__(self, args: Namespace):
        super().__init__()
        for arg in vars(args):
            value_arg = getattr(args, arg)
            print(f"{arg}: {value_arg}")
            self.__setattr__(arg, value_arg)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_family = self.dataset_type.get_family()

    def run(self):
        # load data

        dataset = self.dataset_type.load(is_ibgnn=False, is_undirected=False, add_eigen=False)
        in_dim = dataset[0].target_edge_type.max() + 1
        encode_dim = in_dim if self.one_hot else self.encode_dim
        
        
        path = exp_path(dataset_name=self.dataset_type.name, num_communities=self.num_communities,
                        encode_dim=encode_dim, epochs=self.ibg_approx_epochs, normalize=self.normalize,
                        lr=self.ibg_approx_lr, num_negative=self.num_negative)


        dataset = get_dataset(self.dataset_type.name)
        training, validation, testing  = dataset.training, None, None
        model = ibgE(num_relations=in_dim, num_communities=self.num_communities, embedding_dim=self.encode_dim, one_hot=self.one_hot,
                         normalize=self.normalize, triples_factory=training)


        loss = "BCEWithLogitsLoss"
        loss_kwargs = dict(reduction=self.reduction)

        result = pipeline(
            model=model,
            dataset = dataset,
            dataset_kwargs=dict(
                create_inverse_triples=True
              ),
            training_kwargs=dict(
                num_epochs=self.ibg_approx_epochs,
                use_tqdm_batch=False,
                batch_size=self.batch_size,
            ),
            validation=validation,
            testing=testing,
            random_seed=0,
            loss=loss,
            loss_kwargs=loss_kwargs,
            negative_sampler='basic',
            negative_sampler_kwargs=dict(
                num_negs_per_pos=self.num_negative,
                filtered=True,
            ),

            device=self.device,
            optimizer_kwargs=dict(lr=self.ibg_approx_lr),
        )

        save_path = osp.join(ROOT_DIR, 'pykeen_results', path)
        result.save_to_directory(f"{save_path}")

        results_metric = result.metric_results.to_dict()

        metrics = ["inverse_harmonic_mean_rank", "hits_at_1", "hits_at_3", "hits_at_10"]

        for metric in metrics:
            value = results_metric["both"]["realistic"][metric]
            print(f"test_{metric} = {value}")



if __name__ == "__main__":
    args = ibg_approx_parse_arguments()
    experiment = KGExperiment(args)
    experiment.run()