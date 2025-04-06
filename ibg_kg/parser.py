from argparse import ArgumentParser

from additional_classes.dataset import DataSetType


def ibg_approx_parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--dataset_type", dest="dataset_type", default=DataSetType.umls,
                        type=DataSetType.from_string, choices=list(DataSetType), required=False)

    # ibg_approximation train args
    parser.add_argument("--ibg_approx_epochs", dest="ibg_approx_epochs", default=250, type=int, required=False)
    parser.add_argument("--ibg_approx_lr", dest="ibg_approx_lr", default=0.005, type=float, required=False)

    # ibg_approximation args
    parser.add_argument("--num_communities", dest="num_communities", default=15, type=int, required=False)
    parser.add_argument("--encode_dim", dest="encode_dim", default=48, type=int, required=False)
    parser.add_argument("--one_hot", dest="one_hot", default=False, action='store_true', required=False)
    
    # KG training params
    parser.add_argument("--normalize", dest="normalize", default=False, action='store_true', required=False)
    parser.add_argument("--batch_size", dest="batch_size", default=1024, type=int, required=False)
    parser.add_argument("--num_negative", dest="num_negative", default=128, type=int, required=False)
    parser.add_argument("--reduction", dest="reduction", default="mean", choices=["mean", "sum"], type=str, required=False)
    return parser.parse_args()
