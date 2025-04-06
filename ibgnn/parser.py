from argparse import ArgumentParser

from additional_classes.dataset import DataSetType
from ibgnn.classes import NNTypeCom, NNTypeNode


def ibgnn_parse_arguments():
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
    parser.add_argument("--sparse_scale", dest="sparse_scale", default=0, type=float, required=False)
    
    # ibg_approximation args
    parser.add_argument("--num_communities", dest="num_communities", default=150, type=int, required=False)
    parser.add_argument("--encode_dim", dest="encode_dim", default=0, type=int, required=False)

    # ibgnn args
    parser.add_argument("--num_layers", dest="num_layers", default=5, type=int, required=False)
    parser.add_argument("--hidden_dim", dest="hidden_dim", default=128, type=int, required=False)
    parser.add_argument("--dropout", dest="dropout", default=0.2, type=float, required=False)


    # ibgnn types
    parser.add_argument("--ibgnn_com_type", dest="ibgnn_com_type", default=NNTypeCom.unconstrained,
                        type=NNTypeCom.from_string, choices=list(NNTypeCom), required=False)
    parser.add_argument("--com_layers", dest="com_layers", default=1, type=int, required=False)

    parser.add_argument("--ibgnn_com_node_type", dest="ibgnn_com_node_type", default=NNTypeNode.deepsets_fancy,
                        type=NNTypeNode.from_string, choices=list(NNTypeNode), required=False)
    parser.add_argument("--com_node_layers", dest="com_node_layers", default=3, type=int, required=False)

    parser.add_argument("--ibgnn_node_type", dest="ibgnn_node_type", default=NNTypeNode.deepsets_fancy,
                        type=NNTypeNode.from_string, choices=list(NNTypeNode), required=False)
    parser.add_argument("--node_layers", dest="node_layers", default=3, type=int, required=False)

    parser.add_argument("--pool_type", dest="pool_type", default='max', choices=['sum', 'mean', 'max'], required=False)

    # extra tricks
    parser.add_argument("--skip_com", dest="skip_com", default=False, action='store_true', required=False)
    parser.add_argument("--skip_node", dest="skip_node", default=False, action='store_true', required=False)
    parser.add_argument("--normalize", dest="normalize", default=False, action='store_true', required=False)
    parser.add_argument("--jump", dest="jump", default="cat", choices=[None, 'cat', 'max', 'lstm'], required=False)
    parser.add_argument("--layer_norm", dest="layer_norm", default=False, action='store_true', required=False)


    # ibgnn train args
    parser.add_argument("--epochs", dest="epochs", default=1250, type=int, required=False)
    parser.add_argument("--lr", dest="lr", default=3e-3, type=float, required=False)
    parser.add_argument("--patience", dest="patience", default=300, type=int, required=False)
    parser.add_argument("--refine_ibg_epoch", dest="refine_ibg_epoch", default=0, type=int, required=False)
    parser.add_argument("--refine_scale", dest="refine_scale", default=1e-4, type=float, required=False)


    parser.add_argument("--seed", dest="seed", default=0, type=int, required=False)
    
    return parser.parse_args()



    