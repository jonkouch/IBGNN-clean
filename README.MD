#  Efficient Learning on Large Graphs using a Densifying Regularity Lemma

This repository contains the official code base of the paper **[Efficient Learning on Large Graphs using a Densifying Regularity Lemma]()**.

[PyG]: https://pytorch-geometric.readthedocs.io/en/latest/

## Installation ##
To reproduce the results please use Python 3.9, PyTorch version 2.3.0, Cuda 11.8, PyG version 2.5.3, and torchmetrics.

```bash
pip3 install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip3 install torch-geometric==2.5.3
pip3 install torchmetrics
pip3 install pandas
pip3 install neptune
pip3 install matplotlib
pip3 install einops
pip3 install triton
pip3 install git+https://github.com/TorchSpatiotemporal/tsl.git
pip3 install pykeen
```
Note that the pykeen package is very unstable and might require changing its source code.

## Running

### IBG approximation - node classification and spatio-temporal experiments 

The script we use to run the IBG approximation is ``./ibg_approximation_main.py``.
Note that the script should be run with ``.`` as the main directory or source root.

The parameters of the script are:

- ``--dataset_type``: name of the dataset.
- ``--ibg_approx_epochs``: the number of epochs for the approximation.
- ``--ibg_approx_lr``: the lr for the approximation.
- ``--loss_scale``: the scaler $\lambda$ between the graph and signal loss.
- ``--cut_norm``: whether or not to calculate the cut norm using the cutnorm package https://github.com/pingkoc/cutnorm/tree/master (which is already inserted to this repo.
- ``--add_eigen``: whether or not to use the eigen initialization.
- ``--node_drop_ratio``: the ratio of nodes dropped from the graph.
- ``--is_undirected``: whether or not to load the graph as undirected.
- ``--sparse``: whether or not to use the densifying loss present in the paper.
- ``--sparse_scale``: the scale $\Gamma$ of the densifying loss.
- ``--num_communities``: the number of communities used in the approximation.
- ``--encode_dim``: the output dimension of the linear encoder that is applied to the features. If set to zero, no encoding is applied.
- ``--load_epoch``: To continue training from a checkpoint, set the epoch to load.
- ``--triton``: whether or not to use triton for faster computation of the densifying loss.

To perform experiments over the tolokers dataset for 50 communities, 10 epochs, a lr of 0.001 and $\lambda=0.5$, using the densifying loss with sacle $\Gamma=5.0$, while removing 0% of the graph: 
```bash
python -u ibg_approximation_main.py --dataset_type tolokers --num_communities 50 --ibg_approx_epochs 10 --ibg_approx_lr 0.001 --loss_scale 0.5 --sparse --sparse_scale 5.0 --node_drop_ratio 0.0
```

### IBG-NN - node classification experiments

After an IBG has been trained. The script we use to run IBG-NN is ``./ibgnn_main.py``.
Note that the script should be run with ``.`` as the main directory or source root.

Make sure that the following parameters match those used for the IBG approximation:

- ``--dataset_type``: name of the dataset.
- ``--ibg_approx_epochs``: the number of epochs used for the approximation.
- ``--ibg_approx_lr``: the lr for the approximation.
- ``--loss_scale``: the scaler $\lambda$ between the graph and signal loss.
- ``--add_eigen``: whether or not to use the eigen initialization.
- ``--node_drop_ratio``: the ratio of nodes dropped from the graph.
- ``--is_undirected``: whether or not to load the graph as undirected.
- ``--sparse``: whether or not to use the densifying loss present in the paper.
- ``--sparse_scale``: the scale $\Gamma$ of the densifying loss.
- ``--num_communities``: the number of communities used in the approximation.
- ``--encode_dim``: the output dimension of the linear encoder that is applied to the features. If set to zero, no encoding is applied.


The new parameters for the scripts are:
- ``--num_layers``: the number of IBG-NN layers.
- ``--hidden_dim``: the hidden dimension.
- ``--dropout``: the dropout ratio.
- ``--ibgnn_com_type``: the network variation for community processing.
The available options are: unconstrained, basic and deepsets.
- ``--ibgnn_com_layers``: the number of layers used for community processing.
- ``--ibgnn_com_node_type``: the type of node used for node-community processing.
The available options are: linear, deepsets and deepsets_fancy.
- ``--ibgnn_com_node_layers``: the number of layers used for node-community processing.
- ``--ibgnn_node_type``: the type of node used for node-node processing.
The available options are: linear, deepsets and deepsets_fancy.
- ``--ibgnn_node_layers``: the number of layers used for node-node processing.
- ``--pool_type``: the type of pooling used for the network layers.
The available options are: mean, max and sum.
- ``--skip_com``: whether or not to include a residual connection in the community processing.
- ``--skip_node``: whether or not to include a residual connection in the node processing.
- ``--normalize``: whether or not to normalize the output of the network.
- ``--jumping``: whether or not to use the jumping knowledge network.
The available options are: None, cat, max and lstm.
- ``--layer_norm``: whether or not to use layer normalization.
- ``--epochs``: the number of epochs used for the fitting.
- ``--lr``: the learning used for the fitting.
- ``--patience``: the patience for early stopping.
- ``--seed``: a seed to set random processes.

To perform experiments over a 3-layered IBG-NN model with a hidden dimension of 128 on the tolokers dataset for 300 epochs with a lr of 0.03 using the previous IBG approximation:
```bash
python -u ibgnn_main.py --dataset_type tolokers --num_communities 50 --ibg_approx_epochs 10 --ibg_approx_lr 0.001 --loss_scale 0.5 --node_drop_ratio 0.0 --sparse --sparse_scale 5.0 --num_layers 3 --hidden_dim 128 --dropout 0.5 --ibgnn_com_type deepsets --ibgnn_com_layers 2 --ibgnn_com_node_type deepsets --ibgnn_com_node_layers 2 --ibgnn_node_type deepsets_fancy --ibgnn_node_layers 2 --pool_type mean --skip_com --skip_node --normalize --jumping cat --layer_norm --epochs 300 --lr 0.03 --patience 50 --seed 42

```

### IBG-NN - spatio-temporal experiments

After an IBG has been trained. We build over the codebase https://github.com/Graph-Machine-Learning-Group/taming-local-effects-stgnns for spatio-temporal experiments.
The script we use to run IBG-NN is ``./ibgnn_spatio_temporal_main.py``.
Note that the script should be run with ``.`` as the main directory or source root.
Also note that subsampling is not implemented for these datasets (node_drop_ratio=0) and nodes shouldn't be removed from the graph ($\lambda=0$). 

Make sure that the following parameters match those used for the IBG approximation:

- ``dataset=``: name of the dataset.  
- ``model.ibg_approx_train_args.epochs=``: the number of epochs used for the approximation.  
- ``model.ibg_approx_train_args.lr=``: the lr for the approximation.  
- ``model.ibg_approx_train_args.loss_scale=``: the scaler $\lambda$ between the graph and signal loss.  
- ``model.ibg_approx_args.add_eigen=``: whether or not to use the eigen initialization.  
- ``model.ibg_approx_args.num_communities=``: the number of communities used in the approximation.  
- ``model.ibg_approx_args.encode_dim=``: the output dimension of the linear encoder that is applied to the features. If set to zero, no encoding is applied.  

The new parameters for the script are:  
- ``model.ibgnn_args.num_layers=``: the number of IBG-NN layers.  
- ``model.hidden_dim=``: the hidden dimension.  
- ``model.ibgnn_args.dropout=``: the dropout ratio.  
- ``model.ibgnn_args.ibgnn_com_type=``: the network variation for community processing.  
  The available options are: unconstrained, basic and deepsets.  
- ``model.ibgnn_args.com_layers=``: the number of layers used for community processing.  
- ``model.ibgnn_args.ibgnn_com_node_type=``: the type of node used for node-community processing.  
  The available options are: linear, deepsets and deepsets_fancy.  
- ``model.ibgnn_args.com_node_layers=``: the number of layers used for node-community processing.  
- ``model.ibgnn_args.ibgnn_node_type=``: the type of node used for node-node processing.  
  The available options are: linear, deepsets and deepsets_fancy.  
- ``model.ibgnn_args.node_layers=``: the number of layers used for node-node processing.  
- ``model.ibgnn_args.pool_type=``: the type of pooling used for the network layers.  
  The available options are: mean, max and sum.  
- ``model.ibgnn_args.skip_com=``: whether or not to include a residual connection in the community processing.  
- ``model.ibgnn_args.skip_node=``: whether or not to include a residual connection in the node processing.  
- ``model.ibgnn_args.normalize=``: whether or not to normalize the output of the network.  
- ``model.ibgnn_args.jump=``: whether or not to use the jumping knowledge network.  
  The available options are: None, cat, max and lstm.  
- ``model.ibgnn_args.layer_norm=``: whether or not to use layer normalization.  
- ``epochs=``: the number of epochs used for the fitting.  
- ``optimizer.hparams.lr=``: the learning rate used for the fitting.  
- ``patience=``: the patience for early stopping.  
- ``seed=``: a seed to set random processes.

To perform experiments over a 3 layered IBG-NN model with a hidden dimension of 128 on the METR-LA dataset for 300 epochs with a lr of 0.03 using an IBG approximation with 50 communities, 10 epochs and a lr of 0.001: 
```bash
python -u ibg_approximation_main.py --dataset_type la --num_communities 50 --ibg_approx_epochs 10 --ibg_approx_lr 0.001 --loss_scale 0.0 --node_drop_ratio 0.0
python3 -u ~/Jonathan/MSC/DICG/ICGNN/icgnn_spatio_temporal_main.py model.project=IBG/la-icgnn dataset=la model.ibg_approx_args.num_communities=50 model.ibg_approx_args.encode_dim=0 model.ibg_approx_args.node_drop_ratio=0.0 model.ibg_approx_args.add_eigen=True model.ibg_approx_train_args.num_hop=1 model.ibg_approx_train_args.epochs=20000 model.ibg_approx_train_args.lr=0.05 model.ibg_approx_train_args.loss_scale=1.0 model.ibgnn_args.num_layers=5 model.ibgnn_args.dropout=0.0 model.ibgnn_args.skip_com=False model.ibgnn_args.skip_node=True model.ibgnn_args.normalize=True model.ibgnn_args.layer_norm=False model.ibgnn_args.jump=max model.ibgnn_args.ibgnn_com_type=unconstrained model.ibgnn_args.com_layers=1 model.ibgnn_args.ibgnn_com_node_type=deepsets_fancy model.ibgnn_args.com_node_layers=3 model.ibgnn_args.ibgnn_node_type=linear model.ibgnn_args.node_layers=1 model.ibgnn_args.pool_type=max model.hidden_dim=128 epochs=300 optimizer.hparams.lr=0.001 patience=10 seed=0
```


# ibgE - Knowledge Graph Embedding

The script we use to run the KGE experiments is ./run_kg_benchmarks.py.
Note that the script should be run with . as the main directory or source root.

The parameters of the script are:

- `--dataset_type`: name of the dataset.
- `--ibg_approx_epochs`: the number of epochs for the approximation.
- `--ibg_approx_lr`: the lr for the approximation.
- `--num_communities`: the number of communities used in the approximation.
- `--encode_dim`: the output dimension of the linear encoder that is applied to the features. If set to zero, no encoding is applied.
- `--one_hot`: whether or not to one-hot encode the features.
- `--normalize`: whether or not to normalize the output.
- `--batch_size`: the batch size used during training.
- `--num_negative`: the number of negative samples per positive sample.
- `--reduction`: the method to reduce the loss. The available options are: `mean` and `sum`.

To perform experiments over the UMLS dataset using 15 communities, 250 approximation epochs, a lr of 0.005, with 128 negative samples, batch size 1024 and mean reduction:
```bash
python -u run_kg_benchmarks.py --dataset_type umls --num_communities 15 --ibg_approx_epochs 250 --ibg_approx_lr 0.005  --encode_dim 48 --normalize --batch_size 1024 --num_negative 128 --reduction mean

## Cite

If you make use of this code, or its accompanying [paper](), please cite this work as follows:
```bibtex

```