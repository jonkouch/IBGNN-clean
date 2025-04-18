import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tsl import logger
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import MetrLA, PemsBay
from tsl.datasets.pems_benchmarks import PeMS03, PeMS04, PeMS07, PeMS08
from tsl.experiment import Experiment
from tsl.metrics import torch_metrics

from ibg_approximation.utils import set_seed
from ibgnn_spatio_temporal.lib.datasets import LocalGlobalGPVARDataset, AirQuality
from ibgnn_spatio_temporal.lib.nn import models, EmbeddingPredictor
from ibgnn_spatio_temporal.lib.utils import find_devices, cfg_to_python


def get_model_class(model_str):
    # Basic models  #####################################################
    if model_str == 'ttg_iso':
        model = models.TimeThenGraphIsoModel
    elif model_str == 'ttg_aniso':
        model = models.TimeThenGraphAnisoModel
    elif model_str == 'tag_iso':
        model = models.TimeAndGraphIsoModel
    elif model_str == 'tag_aniso':
        model = models.TimeAndGraphAnisoModel
    # Baseline models  ##################################################
    elif model_str == 'rnn':
        model = models.RNNModel
    elif model_str == 'fcrnn':
        model = models.FCRNNModel
    elif model_str == 'local_rnn':
        model = models.LocalRNNModel
    # SOTA baseline models  #############################################
    elif model_str == 'dcrnn':
        model = models.DCRNNModel
    elif model_str == 'gwnet':
        model = models.GraphWaveNetModel
    elif model_str == 'agcrn':
        model = models.AGCRNModel
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model


def get_dataset(dataset_cfg):
    name = dataset_cfg.name
    if name == 'la':
        dataset = MetrLA(impute_zeros=True)
    elif name == 'bay':
        dataset = PemsBay()
    elif name == 'pems3':
        dataset = PeMS03()
    elif name == 'pems4':
        dataset = PeMS04()
    elif name == 'pems7':
        dataset = PeMS07()
    elif name == 'pems8':
        dataset = PeMS08()
    elif name == 'air':
        dataset = AirQuality(impute_nans=True)
    elif name == 'gpvar':
        dataset = LocalGlobalGPVARDataset(**dataset_cfg.hparams, p_max=0)
    elif name == 'lgpvar':
        dataset = LocalGlobalGPVARDataset(**dataset_cfg.hparams)
    else:
        raise ValueError(f"Dataset {name} not available.")
    return dataset


def run_traffic(cfg: DictConfig):
    ########################################
    # data module                          #
    ########################################
    dataset = get_dataset(cfg.dataset)

    covariates = dict()
    if cfg.get('add_exogenous'):
        assert not isinstance(dataset, LocalGlobalGPVARDataset)
        # encode time of the day and use it as exogenous variable
        day_sin_cos = dataset.datetime_encoded('day').values
        weekdays = dataset.datetime_onehot('weekday').values
        covariates.update(u=np.concatenate([day_sin_cos, weekdays], axis=-1))

    torch_dataset = SpatioTemporalDataset(target=dataset.dataframe(),
                                          mask=dataset.mask,
                                          covariates=covariates,
                                          horizon=cfg.horizon,
                                          window=cfg.window,
                                          stride=cfg.stride)
    if cfg.get('mask_as_exog', False) and 'u' in torch_dataset:
        torch_dataset.update_input_map(u=['u', 'mask'])

    scale_axis = (0,) if cfg.get('scale_axis') == 'node' else (0, 1)
    transform = {
        'target': StandardScaler(axis=scale_axis)
    }

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=dataset.get_splitter(**cfg.dataset.splitting),
        batch_size=cfg.batch_size,
        workers=cfg.workers
    )
    dm.setup()

    # get adjacency matrix
    adj = dataset.get_connectivity(**cfg.dataset.connectivity,
                                   train_slice=dm.train_slice,
                                   force_symmetric=True,
                                   )
    dm.torch_dataset.set_connectivity(adj)

    ########################################
    # Create model                         #
    ########################################

    if cfg.model.name == 'ibgnn':
        from ibg_approximation.classes import IBGApproxArgs, IBGApproxTrainArgs
        from ibgnn.classes import NNArgs, IBGNNArgs, NNTypeCom, NNTypeNode
        from ibg_approximation.model import IBGApproxModel
        from ibgnn.model_spatio_temporal import TTSIBGNN
        from ibg_approximation.utils import exp_path
        from additional_classes.activation import ActivationType
        import os.path as osp

        cfg.model.ibgnn_args.ibgnn_node_type = NNTypeNode.from_string(cfg.model.ibgnn_args.ibgnn_node_type)
        cfg.model.ibgnn_args.ibgnn_com_node_type = NNTypeNode.from_string(cfg.model.ibgnn_args.ibgnn_com_node_type)
        cfg.model.ibgnn_args.ibgnn_com_type = NNTypeCom.from_string(cfg.model.ibgnn_args.ibgnn_com_type)
        
        ibg_approx_model_args = IBGApproxArgs(num_nodes=torch_dataset.n_nodes,
                                           in_dim=1,
                                           time_steps=torch_dataset.window,
                                           init_affiliate_mat_src=None,
                                           init_affiliate_mat_dst=None,
                                           init_com_scale=None,
                                           init_feat_mat_src=None,
                                           init_feat_mat_dst=None,
                                           is_kg=False,
                                           **cfg.model.ibg_approx_args)
        ibg_approx_train_args = IBGApproxTrainArgs(**cfg.model.ibg_approx_train_args)
        path = exp_path(dataset_name=cfg.dataset.name,
                        ibg_approx_args=ibg_approx_model_args,
                        ibg_approx_train_args=ibg_approx_train_args, seed=0,
                        sparse=True, sparse_scale=cfg.model.ibg_approx_train_args.sparse_scale)
        # load ibg_approximation model
        print('Loading IBG Approximation Model')
        ibg_approx_model = IBGApproxModel(model_args=ibg_approx_model_args, is_ibgnn=True)
        state_dict = torch.load(osp.join(path, 'model.pt'),
                                lambda loc, state: loc)
        if 'feat_mat_src' in state_dict:
            del state_dict['feat_mat_src']
            del state_dict['feat_mat_dst']
        if hasattr(ibg_approx_model, 'feat_mat_src'):
            delattr(ibg_approx_model, 'feat_mat_src')
            delattr(ibg_approx_model, 'feat_mat_dst')
        ibg_approx_model.load_state_dict(state_dict)
        ibg_approx_model.encoder = None
        # ibg_approx_model.requires_grad_(requires_grad=False)
        ibgnn_args = NNArgs(num_communities=cfg.model.ibg_approx_args.num_communities,
                               act_type=ActivationType.RELU,
                               **cfg.model.ibgnn_args)
        model_args = IBGNNArgs(encoded_dim=cfg.model.hidden_dim,
                              hidden_dim=cfg.model.hidden_dim,
                              out_dim=cfg.model.hidden_dim,
                              ibgnn_args=ibgnn_args)
        model_cls = TTSIBGNN
        d_exog = torch_dataset.input_map.u.shape[-1] if 'u' in torch_dataset else 0
        model_kwargs = dict(model_args=model_args, ibg_approx_model=ibg_approx_model,
                            n_nodes=torch_dataset.n_nodes,
                            input_size=torch_dataset.n_channels,
                            exog_size=d_exog,
                            output_size=torch_dataset.n_channels,
                            embedding_cfg=cfg.get('embedding'),
                            horizon=torch_dataset.horizon)
        
        cfg.model.ibgnn_args.ibgnn_node_type = NNTypeNode.from_string(cfg.model.ibgnn_args.ibgnn_node_type.name)
        cfg.model.ibgnn_args.ibgnn_com_node_type = NNTypeNode.from_string(cfg.model.ibgnn_args.ibgnn_com_node_type.name)
        cfg.model.ibgnn_args.ibgnn_com_type = NNTypeCom.from_string(cfg.model.ibgnn_args.ibgnn_com_type.name)
    else:
        model_cls = get_model_class(cfg.model.name)

        d_exog = torch_dataset.input_map.u.shape[-1] if 'u' in torch_dataset else 0
        model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                            input_size=torch_dataset.n_channels,
                            exog_size=d_exog,
                            output_size=torch_dataset.n_channels,
                            weighted_graph=torch_dataset.edge_weight is not None,
                            embedding_cfg=cfg.get('embedding'),
                            horizon=torch_dataset.horizon)

        model_cls.filter_model_args_(model_kwargs)
        model_kwargs.update(cfg.model.hparams)

    ########################################
    # predictor                            #
    ########################################

    loss_fn = torch_metrics.MaskedMAE()

    log_metrics = {'mae': torch_metrics.MaskedMAE(),
                   'mre': torch_metrics.MaskedMRE(),
                   'mse': torch_metrics.MaskedMSE()}

    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    # setup predictor
    predictor = EmbeddingPredictor(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        loss_fn=loss_fn,
        metrics=log_metrics,
        add_params_epoch=0,
        beta=cfg_to_python(cfg.regularization_weight),
        embedding_var=cfg.embedding.get('initial_var', 0.2),
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scale_target=cfg.scale_target,
    )

    if cfg.model.name == 'ibgnn':
        predictor.model.set_ibg_approx_after_training()



    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        patience=cfg.patience,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    trainer = Trainer(max_epochs=cfg.epochs,
                      limit_train_batches=cfg.train_batches,
                      default_root_dir=cfg.run.dir,
                      logger=None,
                      accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      devices=find_devices(1),
                      gradient_clip_val=cfg.grad_clip_val,
                      callbacks=[early_stop_callback, checkpoint_callback])

    load_model_path = cfg.get('load_model_path')
    if load_model_path is not None:
        predictor.load_model(load_model_path)
    else:
        trainer.fit(predictor, train_dataloaders=dm.train_dataloader(),
                    val_dataloaders=dm.val_dataloader())
        # Load best model
        storage = torch.load(checkpoint_callback.best_model_path,
                             lambda storage, loc: storage)
        predictor.load_state_dict(storage['state_dict'])

    ########################################
    # testing                              #
    ########################################

    predictor.freeze()
    trainer.test(predictor, dataloaders=dm.test_dataloader())


if __name__ == '__main__':
    set_seed(0)
    exp = Experiment(run_fn=run_traffic, config_path='ibgnn_spatio_temporal/config/static',
                     config_name='default')
    res = exp.run()
    logger.info(res)
