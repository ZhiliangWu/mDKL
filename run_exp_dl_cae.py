#
# mDKL
#
# Copyright (c) Siemens AG, 2021
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

import gc
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd
import mlflow

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

import ignite
from ignite.contrib.handlers.mlflow_logger import MLflowLogger, \
    global_step_from_engine
from ignite.contrib.handlers import FastaiLRFinder, ProgressBar
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.contrib.metrics.regression import R2Score
from ignite.engine import Events, create_supervised_trainer, \
    create_supervised_evaluator
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import MeanSquaredError

from data_utils import AugmBBoxDatasetAll, get_tranform, \
    get_augm_data_loaders_in_file, prepare_batch
from logging_conf import logger
from model_utils import get_model, EpochOutputStore, CheckPointAfter, LinearModel
from plot_utils import joint_plot_xy, residual_plot
from cae_dl import run_cae


def run_lr(batch_size=64, lr=3e-4, alpha=0.1, model_name='resnet', epoch=50,
           pretrain=True, fold_idx=0, device=torch.device('cpu'),
           exp_name='dataset_xxx', run_name='model_xxx', seed=42, augm=True):
    """Run the experiment with cae as a pre-training step.

    Args:
        batch_size (int): Batch size.
        lr (float): The value of the learning rate, possibly from lrfinder.
        alpha (float): The value of weight decay (a.k.a. regularization).
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
        augm (bool): Whether use augmentation.

    Returns:
        None: The evolution of training loss and evaluation loss are saved to
            MLflow.

    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    cae, n_feature = run_cae(batch_size=50,
                             lr=3e-4,
                             alpha=1e-5,
                             model_name=model_name,
                             epoch=200,
                             pretrain=pretrain,
                             fold_idx=fold_idx,
                             device=device,
                             exp_name=exp_name,
                             run_name=f'{model_name}_cae',
                             seed=seed)

    model = LinearModel(feature_extractor=cae.encoder, num_features=n_feature,
                        output_dim=4)

    model = model.to(device)

    df_path = Path('../DL_lung_0.5.csv')
    im_path = Path(f'{str(Path.home())}/Key_DL/')

    df_split = pd.read_csv('../DL_lung_0.5_idx_split.csv', index_col=0)

    custom_tranform = get_tranform(mean=(0.389, 0.389, 0.389),
                                   std=(0.240, 0.240, 0.240),
                                   bbox=True)

    train_loader, train_evaluator_loader, valid_loader, test_loader =  \
        get_augm_data_loaders_in_file(df_path, im_path, df_split,
                                      train_batch_size=batch_size,
                                      valid_batch_size=128,
                                      custom_tranform=custom_tranform,
                                      datasetclass=AugmBBoxDatasetAll,
                                      fname_col='File_name',
                                      n_fold=fold_idx, augm=augm,
                                      albu_format=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=alpha)
    # step_scheduler = StepLR(optimizer, step_size=int(epoch/2), gamma=0.1)
    # scheduler = LRScheduler(step_scheduler)

    def train_output_transform(x, y, y_pred, loss):
        return {'both': (y_pred, y),
                'x': (y_pred[:, 0], y[:, 0]),
                'y': (y_pred[:, 1], y[:, 1]),
                'w': (y_pred[:, 2], y[:, 2]),
                'h': (y_pred[:, 3], y[:, 3]),
                'loss': loss.item()
                }

    trainer = create_supervised_trainer(model, optimizer,
                                        nn.MSELoss(),
                                        # nn.L1Loss(),
                                        device=device,
                                        prepare_batch=prepare_batch,
                                        output_transform=train_output_transform
                                        )

    # trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer,
                output_transform=lambda out: {'batch loss': out['loss']})

    # evaluators
    def eva_output_tranform(x, y, y_pred):
        return {'both': (y_pred, y),
                'x': (y_pred[:, 0], y[:, 0]),
                'y': (y_pred[:, 1], y[:, 1]),
                'w': (y_pred[:, 2], y[:, 2]),
                'h': (y_pred[:, 3], y[:, 3]),
                }

    val_metrics = {
                   'mse': MeanSquaredError(output_transform=lambda out: out['both']),
                   'r2score_x': R2Score(output_transform=lambda out: out['x']),
                   'r2score_y': R2Score(output_transform=lambda out: out['y']),
                   'r2score_w': R2Score(output_transform=lambda out: out['w']),
                   'r2score_h': R2Score(output_transform=lambda out: out['h'])
                   }

    # add metrics to trainer, evaluating based on a moving model / moving data
    for name, metric in val_metrics.items():
        metric.attach(trainer, name)

    # evaluate the model on the training dataset
    train_evaluator = create_supervised_evaluator(model, metrics=val_metrics,
                                                  device=device,
                                                  prepare_batch=prepare_batch,
                                                  output_transform=eva_output_tranform
                                                  )

    pbar.attach(train_evaluator)

    evaluator = create_supervised_evaluator(model, metrics=val_metrics,
                                            device=device,
                                            prepare_batch=prepare_batch,
                                            output_transform=eva_output_tranform
                                            )
    pbar.attach(evaluator)

    eos_x_train = EpochOutputStore(output_transform=lambda out: out['x'])
    eos_y_train = EpochOutputStore(output_transform=lambda out: out['y'])
    eos_w_train = EpochOutputStore(output_transform=lambda out: out['w'])
    eos_h_train = EpochOutputStore(output_transform=lambda out: out['h'])
    eos_x_train.attach(train_evaluator)
    eos_y_train.attach(train_evaluator)
    eos_w_train.attach(train_evaluator)
    eos_h_train.attach(train_evaluator)

    eos_x_val = EpochOutputStore(output_transform=lambda out: out['x'])
    eos_y_val = EpochOutputStore(output_transform=lambda out: out['y'])
    eos_w_val = EpochOutputStore(output_transform=lambda out: out['w'])
    eos_h_val = EpochOutputStore(output_transform=lambda out: out['h'])
    eos_x_val.attach(evaluator)
    eos_y_val.attach(evaluator)
    eos_w_val.attach(evaluator)
    eos_h_val.attach(evaluator)

    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name=run_name):

        mlflow_logger = MLflowLogger()

        mlflow_logger.log_params({
            'seed': seed,
            'batch_size': batch_size,
            'num_epoch': epoch,
            'model': model_name,
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
                             f"- R squared of x: {metrics['r2score_x']:.2f}"
                             f"- R squared of y: {metrics['r2score_y']:.2f}"
                             f"- R squared of w: {metrics['r2score_w']:.2f}"
                             f"- R squared of h: {metrics['r2score_h']:.2f}"
                             )
            log_metrics = {f'validation {k}': v for k, v in metrics.items()}
            mlflow_logger.log_metrics(log_metrics, step=engine.state.epoch)

        temp_name = f'temp_{uuid.uuid4()}'

        def score_function(engine):
            return -engine.state.metrics['mse']

        to_save = {'model': model,
                   # 'optimizer': optimizer
                   }
        handler = CheckPointAfter(start_epoch=int(0.9 * epoch),
                                  to_save=to_save,
                                  save_handler=DiskSaver(f'./{temp_name}',
                                                         create_dir=True),
                                  n_saved=2,
                                  filename_prefix='best',
                                  score_function=score_function,
                                  score_name="val_mse",
                                  global_step_transform=global_step_from_engine(trainer))

        evaluator.add_event_handler(Events.COMPLETED, handler)

        # handlers for trainer
        @trainer.on(Events.EPOCH_COMPLETED(every=10))
        def log_training_results(engine):
            train_evaluator.run(train_evaluator_loader)
            metrics = train_evaluator.state.metrics
            pbar.log_message(f"Evaluation Training Set "
                             f"- Epoch: {engine.state.epoch} "
                             f"- Mean Square Error: {metrics['mse']:.4f} "
                             f"- R squared of x: {metrics['r2score_x']:.2f}"
                             f"- R squared of y: {metrics['r2score_y']:.2f}"
                             f"- R squared of w: {metrics['r2score_w']:.2f}"
                             f"- R squared of h: {metrics['r2score_h']:.2f}"
                             )

        def log_plots(engine, label='valid'):
            train_hist_x_p, train_hist_x = eos_x_train.get_output(to_numpy=True)
            train_hist_y_p, train_hist_y = eos_y_train.get_output(to_numpy=True)
            train_hist_w_p, train_hist_w = eos_w_train.get_output(to_numpy=True)
            train_hist_h_p, train_hist_h = eos_h_train.get_output(to_numpy=True)
            val_hist_x_p, val_hist_x = eos_x_val.get_output(to_numpy=True)
            val_hist_y_p, val_hist_y = eos_y_val.get_output(to_numpy=True)
            val_hist_w_p, val_hist_w = eos_w_val.get_output(to_numpy=True)
            val_hist_h_p, val_hist_h = eos_h_val.get_output(to_numpy=True)

            # joint plot for predicted location values
            joint_plot_xy(train_hist_x_p, train_hist_y_p,
                          val_hist_x_p, val_hist_y_p,
                          dp=temp_name, n_epoch=engine.state.epoch,
                          label=f'pred_{label}')

            # residual for x
            residual_plot(train_hist_x, train_hist_x_p,
                          val_hist_x, val_hist_x_p, dp=temp_name,
                          n_epoch=engine.state.epoch, label=f'x_{label}')

            # residual for y
            residual_plot(train_hist_y, train_hist_y_p,
                          val_hist_y, val_hist_y_p, dp=temp_name,
                          n_epoch=engine.state.epoch, label=f'y_{label}')

            # residual for w
            residual_plot(train_hist_w, train_hist_w_p,
                          val_hist_w, val_hist_w_p, dp=temp_name,
                          n_epoch=engine.state.epoch, label=f'w_{label}')

            # residual for h
            residual_plot(train_hist_h, train_hist_h_p,
                          val_hist_h, val_hist_h_p, dp=temp_name,
                          n_epoch=engine.state.epoch, label=f'h_{label}')

        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=10),
                                  log_plots, label='valid')

        @trainer.on(Events.COMPLETED)
        def log_plot(engine):
            _, train_hist_x = eos_x_train.get_output(to_numpy=True)
            _, train_hist_y = eos_y_train.get_output(to_numpy=True)
            _, train_hist_w = eos_w_train.get_output(to_numpy=True)
            _, train_hist_h = eos_h_train.get_output(to_numpy=True)
            _, val_hist_x = eos_x_val.get_output(to_numpy=True)
            _, val_hist_y = eos_y_val.get_output(to_numpy=True)
            _, val_hist_w = eos_w_val.get_output(to_numpy=True)
            _, val_hist_h = eos_h_val.get_output(to_numpy=True)

            joint_plot_xy(train_hist_x, train_hist_y, val_hist_x, val_hist_y,
                          dp=temp_name, n_epoch=engine.state.epoch, label='true'
                          )

            joint_plot_xy(train_hist_w, train_hist_h, val_hist_w, val_hist_h,
                          dp=temp_name, n_epoch=engine.state.epoch,
                          label='hw_true'
                          )

        def final_evaluation(engine):
            to_load = to_save
            last_checkpoint_fp = f'./{temp_name}/{handler.last_checkpoint}'
            checkpoint = torch.load(last_checkpoint_fp, map_location=device)
            CheckPointAfter.load_objects(to_load=to_load, checkpoint=checkpoint)
            logger.info('The best model on validation is reloaded for '
                        'evaluation on the test set')
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            pbar.log_message(f"Testing Results "
                             f"- Epoch: {engine.state.epoch} "
                             f"- Mean Square Error: {metrics['mse']:.4f} "
                             f"- R squared of x: {metrics['r2score_x']:.2f}"
                             f"- R squared of y: {metrics['r2score_y']:.2f}"
                             f"- R squared of w: {metrics['r2score_w']:.2f}"
                             f"- R squared of h: {metrics['r2score_h']:.2f}"
                             )
            log_metrics = {f'test {k}': v for k, v in metrics.items()}
            mlflow_logger.log_metrics(log_metrics, step=engine.state.epoch)

        trainer.add_event_handler(Events.COMPLETED, final_evaluation)
        trainer.add_event_handler(Events.COMPLETED, log_plots, label='test')

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
                                                'batch_loss': out['loss']}
                                            )

        # setup `global_step_transform=global_step_from_engine(trainer)` to
        # take the epoch of the `trainer` instead of `train_evaluator`.
        mlflow_logger.attach_output_handler(trainer,
                                            event_name=Events.EPOCH_COMPLETED,
                                            tag='training',
                                            metric_names=['mse',
                                                          'r2score_x',
                                                          'r2score_y',
                                                          'r2score_w',
                                                          'r2score_h',
                                                          ]
                                            )

        # Attach the logger to the trainer to log optimizer's parameters,
        # e.g. learning rate at each iteration
        mlflow_logger.attach_opt_params_handler(trainer,
                                                event_name=Events.ITERATION_STARTED,
                                                optimizer=optimizer,
                                                param_name='lr'
                                                )

        _ = trainer.run(train_loader, max_epochs=epoch)
        
        return model


if __name__ == '__main__':
    sd = 42
    dc = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    bs = 128
    n_epoch = 100
    pt = True
    exp = 'DL_lung_final'
    m_name = 'resnet'

    a = 1e-4
    lrt = 1e-4  # this is from lr finder

    r_name = f'{m_name}_linear_regressor_albu_cae'

    for f in range(5):
        run_lr(batch_size=bs, lr=lrt, alpha=a, model_name=m_name,
               epoch=n_epoch, pretrain=pt, fold_idx=f, device=dc,
               exp_name=exp, run_name=r_name, seed=sd)
        gc.collect()
