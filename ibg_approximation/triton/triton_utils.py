from torch_geometric.utils.sparse import index2ptr
from torch_geometric.utils import index_sort


def coo_to_csr(row, col, num_nodes=None):
    # Row is the source node, col is the destination node.
    if num_nodes is None:
        num_nodes = int(row.max()) + 1
    row, perm = index_sort(row, max_value=num_nodes)
    col = col[perm]
    rowptr = index2ptr(row, num_nodes)
    return rowptr, col