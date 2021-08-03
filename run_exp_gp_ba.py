#
# mDKL
#
# Copyright (c) Siemens AG, 2021
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

import copy
from functools import partial
import gc
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd
import mlflow
import gpytorch
from sklearn.metrics import mean_squared_error, r2_score

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import ignite
from ignite.contrib.handlers.mlflow_logger import MLflowLogger, \
    global_step_from_engine
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.contrib.metrics.regression import R2Score
from ignite.engine import Events
from ignite.handlers import DiskSaver
from ignite.metrics import MeanSquaredError, Average

from data_utils import BoneAge, get_tranform, get_augm_data_loaders_in_file, \
    prepare_batch
from gp_layer import SVGPLayer
from model_utils import get_pretrained_models, LinearModel, \
    get_initial_inducing_points, DKLModel, create_dkl_trainer, \
    create_dkl_evaluator, EpochOutputStore, CheckPointAfter
from logging_conf import logger
from plot_utils import residual_plot


def run(batch_size=64, lr=3e-4, alpha=0.0, num_gp=1, num_inducing=10,
        dim_gp_in=50, model_name='resnet', epoch=50, pretrain=True, fold_idx=0,
        device=torch.device('cpu'), exp_name='dataset_xxx',
        run_name='model_xxx', seed=42):
    """Run the experiment with a given setting.

    Args:
        batch_size (int): Batch size.
        lr (float): The value of the learning rate, possibly from lrfinder.
        alpha (float): The value of weight decay (a.k.a. regularization).
        num_gp (int): The number of involved GPs.
        num_inducing (int): The multiple of inducing points. The total number
            of inducing points is num_inducing x batch_size.
        dim_gp_in (int): The input dimension of the GP input layer.
        model_name (str): The name of the backbone.
        epoch (int): The number of training epochs.
        pretrain (bool): Whether load the weights in pretrained models.
        fold_idx (int): The index of the training/validation set.
        device (torch.device or str): The device to load the models.
        exp_name (str): The name of the experiments with a format of
            dataset+xxx, which defines the experiment name inside MLflow.
        run_name (str): The name of the run with a format of
            [model_name]_linear_regressor, which defines the run name inside
            MLflow.
        seed (int): The number of the random seed to ensure the reproducibility.

    Returns:
        None: The evolution of training loss and evaluation loss are saved to
            MLflow.

    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    df_path = Path('../boneage.csv')
    im_path = Path(f'{str(Path.home())}/boneage/')

    df_split = pd.read_csv('../boneage_idx_split.csv',
                           index_col=0)

    custom_tranform = get_tranform(mean=(0.183, 0.183, 0.183),
                                   std=(0.166, 0.166, 0.166),
                                   bbox=False)

    train_loader, train_evaluator_loader, valid_loader, test_loader = \
        get_augm_data_loaders_in_file(df_path, im_path, df_split,
                                      train_batch_size=batch_size,
                                      valid_batch_size=128,
                                      custom_tranform=custom_tranform,
                                      datasetclass=BoneAge,
                                      fname_col='id',
                                      n_fold=fold_idx, augm=True)

    backbone, n_feature = get_pretrained_models(model_name=model_name,
                                                pretrain=pretrain)

    feature_extractor = LinearModel(feature_extractor=backbone,
                                    num_features=n_feature,
                                    output_dim=dim_gp_in)
    feature_extractor = feature_extractor.to(device)

    inducing_points = get_initial_inducing_points(feature_extractor,
                                                  train_loader,
                                                  device,
                                                  num_inducing=num_inducing)

    gp_layer = SVGPLayer(inducing_points=inducing_points)

    model = DKLModel(feature_extractor=feature_extractor, gp_layer=gp_layer)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model = model.to(device)
    likelihood = likelihood.to(device)

    model_bk = copy.deepcopy(model)
    likelihood_bk = copy.deepcopy(likelihood)

    # learning rate from lrfinder
    # weight decay from empirical values / regression training
    optimizer = Adam([{'params': model.feature_extractor.parameters(),
                       'weight_decay': alpha,
                       'lr': lr
                       },
                      {'params': model.gp_layer.parameters()},
                      {'params': likelihood.parameters()}
                      ], lr=0.01)

    # mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer,
    #                                     num_data=len(train_loader.dataset))
    mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model.gp_layer,
                                                num_data=len(train_loader.dataset))

    def train_output_transform(x, y, y_pred, loss):
        return {'y': y, 'y_pred': y_pred, 'loss': loss.item()}

    trainer = create_dkl_trainer(model, likelihood, optimizer, mll,
                                 device=device,
                                 prepare_batch=prepare_batch,
                                 output_transform=train_output_transform)

    train_metrics = {'mll': Average(output_transform=lambda x: -x['loss']),
                     'mse': MeanSquaredError(),
                     'r2score': R2Score()
                     }

    for name, metric in train_metrics.items():
        metric.attach(trainer, name)

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer,
                output_transform=lambda out: {'batch mll': -out['loss']})

    # evaluators
    val_metrics = {'mse': MeanSquaredError(),
                   'r2score': R2Score()
                   }

    train_evaluator = create_dkl_evaluator(model, likelihood, metrics=val_metrics,
                                           device=device,
                                           prepare_batch=prepare_batch
                                           )
    pbar.attach(train_evaluator)

    evaluator = create_dkl_evaluator(model, likelihood, metrics=val_metrics,
                                     device=device,
                                     prepare_batch=prepare_batch
                                     )
    pbar.attach(evaluator)

    # eos_train = EpochOutputStore(output_transform=lambda out: (out['y_pred'],
    #                                                            out['y']))
    eos_train = EpochOutputStore()
    eos_valid = EpochOutputStore()
    eos_train.attach(train_evaluator)
    eos_valid.attach(evaluator)

    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name=run_name):

        mlflow_logger = MLflowLogger()

        mlflow_logger.log_params({
            'seed': seed,
            'batch_size': batch_size,
            'num_epoch': epoch,
            'model': model_name,
            'number inducing points': int(num_gp * num_inducing * batch_size),
            'gp_input_dim': dim_gp_in,
            'weight_decay': alpha,
            'fold_index': fold_idx,
            'pytorch version': torch.__version__,
            'ignite version': ignite.__version__,
            'cuda version': torch.version.cuda,
            'device name': torch.cuda.get_device_name(0)
        })

        # handlers for evaluator
        # note, this actually calls the evaluator
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(valid_loader)
            metrics = evaluator.state.metrics
            pbar.log_message(f"Validation Results "
                             f"- Epoch: {engine.state.epoch} "
                             f"- Mean Square Error: {metrics['mse']:.4f} "
                             f"- R squared: {metrics['r2score']:.2f}"
                             )
            log_metrics = {f'validation {k}': v for k, v in metrics.items()}
            mlflow_logger.log_metrics(log_metrics, step=engine.state.epoch)

        temp_name = f'temp_{uuid.uuid4()}'

        def score_function(engine):
            return -engine.state.metrics['mse']

        to_save = {'model': model,
                   'likelihood': likelihood,
                   # 'optimizer': optimizer
                   }
        handler = CheckPointAfter(start_epoch=int(0.9 * epoch),
                                  to_save=to_save,
                                  save_handler=DiskSaver(f'./{temp_name}',
                                                         create_dir=True),
                                  n_saved=1,
                                  filename_prefix='best',
                                  score_function=score_function,
                                  score_name='val_mse',
                                  global_step_transform=global_step_from_engine(
                                      trainer))

        evaluator.add_event_handler(Events.COMPLETED, handler)

        # handlers for trainer
        @trainer.on(Events.EPOCH_COMPLETED(every=10))
        def log_training_results(engine):
            train_evaluator.run(train_evaluator_loader)
            metrics = train_evaluator.state.metrics
            pbar.log_message(f"Training Set "
                             f"- Epoch: {engine.state.epoch} "
                             # f"- MLL: {metrics['mll']:.4f}  "
                             f"- Mean Square Error: {metrics['mse']:.4f} "
                             f"- R squared: {metrics['r2score']:.2f}"
                             )

        def log_plots(engine, label='valid'):
            train_hist_y_p, train_hist_y = eos_train.get_output(to_numpy=True)
            val_hist_y_p, val_hist_y = eos_valid.get_output(to_numpy=True)

            residual_plot(train_hist_y, train_hist_y_p,
                          val_hist_y, val_hist_y_p, dp=temp_name,
                          n_epoch=engine.state.epoch, label=f'y_{label}')

        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=10),
                                  log_plots, label='valid')

        def final_evaluation(engine):
            to_load = {'model': model_bk,
                       'likelihood': likelihood_bk,
                       }
            last_checkpoint_fp = f'./{temp_name}/{handler.last_checkpoint}'
            print(last_checkpoint_fp)
            checkpoint = torch.load(last_checkpoint_fp, map_location=device)
            CheckPointAfter.load_objects(to_load=to_load, checkpoint=checkpoint)
            logger.info('The best model on validation is reloaded for '
                        'evaluation on the test set')

            model_bk.eval()
            likelihood_bk.eval()

            y_true_list = []
            y_pred_list = []
            for batch in test_loader:
                with torch.no_grad():
                    x, y = prepare_batch(batch, device=device,
                                         non_blocking=False)
                    output = model_bk(x)
                    y_pred = output.mean
                    y_true_list.append(y)
                    y_pred_list.append(y_pred)

            y_true = torch.cat(y_true_list, dim=0).cpu().numpy()
            y_pred = torch.cat(y_pred_list, dim=0).cpu().numpy()

            test_mse = mean_squared_error(y_true, y_pred)
            test_r2score = r2_score(y_true, y_pred)

            pbar.log_message(f"Testing Results "
                             f"- Epoch: {engine.state.epoch} "
                             f"- Mean Square Error: {test_mse:.4f} "
                             f"- R squared of x: {test_r2score:.2f}"
                             )
            log_metrics = {'test mse': test_mse,
                           'test r2score': test_r2score,
                           }
            mlflow_logger.log_metrics(log_metrics, step=engine.state.epoch)

        trainer.add_event_handler(Events.COMPLETED, final_evaluation)
        # trainer.add_event_handler(Events.COMPLETED, log_plots, label='test')

        @trainer.on(Events.COMPLETED)
        def save_model_to_mlflow(engine):
            mlflow_logger.log_artifacts(f'./{temp_name}/')
            try:
                shutil.rmtree(temp_name)
            except FileNotFoundError:
                logger.warning('Temp drectory not found!')
                raise

        # log training loss at each iteration
        mlflow_logger.attach_output_handler(trainer,
                                            event_name=Events.ITERATION_COMPLETED,
                                            tag='training',
                                            output_transform=lambda out: {
                                                'batch_mll': -out['loss']}
                                            )

        # setup `global_step_transform=global_step_from_engine(trainer)` to
        # take the epoch of the `trainer` instead of `train_evaluator`.
        mlflow_logger.attach_output_handler(trainer,
                                            event_name=Events.EPOCH_COMPLETED,
                                            tag='training',
                                            metric_names=['mll',
                                                          'mse',
                                                          'r2score']
                                            )

        # Attach the logger to the trainer to log optimizer's parameters,
        # e.g. learning rate at each iteration
        mlflow_logger.attach_opt_params_handler(trainer,
                                                event_name=Events.ITERATION_STARTED,
                                                optimizer=optimizer,
                                                param_name='lr'
                                                )

        _ = trainer.run(train_loader, max_epochs=epoch)


if __name__ == '__main__':
    sd = 42
    dc = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    bs = 128
    n_epoch = 100
    pt = True
    n_gp = 1
    n_idc = 10
    dim_gp_input = 50
    a = 1e-4  # a small value is a good default
    lrt = 3e-5  # this is for cnn parameters
    exp = 'boneage_augm'
    m_name = 'resnet'

    r_name = f'{m_name}_ppgp'

    for f in range(5):
        run(batch_size=bs, lr=lrt, alpha=a, num_gp=n_gp, num_inducing=n_idc,
            dim_gp_in=dim_gp_input, model_name=m_name,
            epoch=n_epoch, pretrain=pt, fold_idx=f, device=dc, exp_name=exp,
            run_name=r_name, seed=sd)
        gc.collect()
