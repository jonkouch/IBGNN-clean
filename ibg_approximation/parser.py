from argparse import ArgumentParser

from additional_classes.dataset import DataSetType


def ibg_approx_parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--dataset_type", dest="dataset_type", default=DataSetType.chameleon,
                        type=DataSetType.from_string, choices=list(DataSetType), required=False)

    # ibg_approximation train args
    parser.add_argument("--ibg_approx_epochs", dest="ibg_approx_epochs", default=100, type=int, required=False)
    parser.add_argument("--ibg_approx_lr", dest="ibg_approx_lr", default=5e-2, type=float, required=False)
    parser.add_argument("--loss_scale", dest="loss_scale", default=0.0, type=float, required=False)
    parser.add_argument("--cut_norm", dest="cut_norm", default=False, action='store_true', required=False)
    parser.add_argument("--add_eigen", dest="add_eigen", default=False, action='store_true', required=False)
    parser.add_argument("--node_drop_ratio", dest="node_drop_ratio", default=0.0, type=float, required=False)
    parser.add_argument("--is_undirected", dest="is_undirected", default=False, action='store_true', required=False)
    parser.add_argument("--sparse", dest="sparse", default=False, action='store_true', required=False)
    parser.add_argument("--sparse_scale", dest="sparse_scale", default=20, type=float, required=False)


    # ibg_approximation args
    parser.add_argument("--num_communities", dest="num_communities", default=150, type=int, required=False)
    parser.add_argument("--encode_dim", dest="encode_dim", default=0, type=int, required=False)
    
    parser.add_argument("--load_epoch", dest="load_epoch", default=0, type=int, required=False)  # explore
    parser.add_argument("--triton", dest="triton", default=False, action='store_true', required=False)

    # reproduce
    return parser.parse_args()
