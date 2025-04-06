from torch import zeros, bool
import numpy as np
import torch
from math import floor
from torch_geometric.data import Data

def get_mask(idx, num_nodes):
    """
    Given a tensor of ids and a number of nodes, return a boolean mask of size num_nodes which is set to True at indices
    in `idx`, and to False for other indices.
    """
    mask = zeros(num_nodes, dtype=bool)
    mask[idx] = 1
    return mask


# TODO: write where this is from
def set_uniform_train_val_test_split(seed, data, train_ratio=0.5, val_ratio=0.25):
        rnd_state = np.random.RandomState(seed)
        num_nodes = data.y.shape[0]

        # Some nodes have labels -1 (i.e. unlabeled), so we need to exclude them
        labeled_nodes = torch.where(data.y != -1)[0]
        num_labeled_nodes = labeled_nodes.shape[0]
        num_train = floor(num_labeled_nodes * train_ratio)
        num_val = floor(num_labeled_nodes * val_ratio)

        idxs = list(range(num_labeled_nodes))
        # Shuffle in place
        rnd_state.shuffle(idxs)

        train_idx = idxs[:num_train]
        val_idx = idxs[num_train : num_train + num_val]
        test_idx = idxs[num_train + num_val :]

        train_idx = labeled_nodes[train_idx]
        val_idx = labeled_nodes[val_idx]
        test_idx = labeled_nodes[test_idx]

        train_mask = get_mask(train_idx, num_nodes)
        val_mask = get_mask(val_idx, num_nodes)
        test_mask = get_mask(test_idx, num_nodes)

        setattr(data, 'train_mask', train_mask)
        setattr(data, 'val_mask', val_mask)
        setattr(data, 'test_mask', test_mask)

        return data


# Taken verbatim from https://github.com/CUAI/Non-Homophily-Large-Scale/blob/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/data_utils.py#L39
def even_quantile_labels(vals, nclasses, verbose=True):
    """partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int64)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print("Class Label Intervals:")
        for class_idx, interval in enumerate(interval_lst):
            print(f"Class {class_idx}: [{interval[0]}, {interval[1]})]")
    return label


def process_pykeen_dataset(dataset):
    num_nodes = dataset.training.num_entities
    train_edge_index = dataset.training.mapped_triples[:, [0, 2]].T
    train_edge_type = dataset.training.mapped_triples[:, 1]
    valid_edge_index = dataset.validation.mapped_triples[:, [0, 2]].T
    valid_edge_type = dataset.validation.mapped_triples[:, 1]
    test_edge_index = dataset.testing.mapped_triples[:, [0, 2]].T
    test_edge_type = dataset.testing.mapped_triples[:, 1]

    edge_index = torch.cat([train_edge_index, valid_edge_index, test_edge_index], dim=1)
    edge_type = torch.cat([train_edge_type, valid_edge_type, test_edge_type])

    train_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                          target_edge_index=train_edge_index, target_edge_type=train_edge_type)
    valid_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                          target_edge_index=valid_edge_index, target_edge_type=valid_edge_type)
    test_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                          target_edge_index=test_edge_index, target_edge_type=test_edge_type)


    return [train_data, valid_data, test_data]