#
# mDKL
#
# Copyright (c) Siemens AG, 2021
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

from functools import partial
import gc
from pathlib import Path
import shutil
import uuid

import mlflow
import numpy as np
import pandas as pd

import torch
from torch import nn

import ignite
from ignite.contrib.handlers.mlflow_logger import MLflowLogger, \
    global_step_from_engine
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics.regression import R2Score
from ignite.engine import Events, create_supervised_trainer, \
    create_supervised_evaluator
from ignite.handlers import DiskSaver
from ignite.metrics import MeanSquaredError, Average

from data_utils import BoneAge, get_tranform, get_augm_data_loaders_in_file,\
    prepare_batch
from plot_utils import residual_plot
from logging_conf import logger
from model_utils import LinearModel, EpochOutputStore, CheckPointAfter
from pml_ba import run_metric_learning


def run(batch_size=64, lr=3e-4, alpha=1e-3, model_name='resnet', epoch=50,
        pretrain=True, fold_idx=0, device=torch.device('cpu'),
        exp_name='dataset_xxx', run_name='model_xxx', seed=42):
    """Run the experiment with metric learning as the pre-training.

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

       Returns:
           None: The evolution of training loss and evaluation loss are saved to
               MLflow.

       """

    np.random.seed(seed)
    torch.manual_seed(seed)

    backbone, n_feature = run_metric_learning(batch_size=batch_size,
                                              lr=1e-4,
                                              alpha=alpha,
                                              model_name=model_name,
                                              epoch=epoch,
                                              pretrain=pretrain,
                                              fold_idx=fold_idx,
                                              exp_name=exp_name,
                                              run_name=f'{model_name}_metrics',
                                              device=device,
                                              seed=seed)

    model = LinearModel(feature_extractor=backbone, num_features=n_feature,
                        output_dim=1)

    model = model.to(device)

    df_path = Path('../boneage.csv')
    im_path = Path(f'{str(Path.home())}/boneage/')

    df_split = pd.read_csv('../boneage_idx_split.csv', index_col=0)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=alpha)

    # step_scheduler = StepLR(optimizer, step_size=int(epoch/2), gamma=0.1)
    # scheduler = LRScheduler(step_scheduler)

    def train_output_transform(x, y, y_pred, loss):
        return {'y': y, 'y_pred': y_pred, 'loss': loss.item()}

    prepare_batch_reshape = partial(prepare_batch, new_shape=(-1, 1))
    trainer = create_supervised_trainer(model, optimizer,
                                        nn.MSELoss(),
                                        device=device,
                                        prepare_batch=prepare_batch_reshape,
                                        output_transform=train_output_transform
                                        )

    # trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

    train_metrics = {'mse': Average(output_transform=lambda x: x['loss']),
                     }

    for name, metric in train_metrics.items():
        metric.attach(trainer, name)

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer,
                output_transform=lambda out: {'batch loss': out['loss']})

    # evaluators
    val_metrics = {'mse': MeanSquaredError(),
                   'r2score': R2Score()
                   }

    for name, metric in val_metrics.items():
        metric.attach(trainer, name)

    train_evaluator = create_supervised_evaluator(model, metrics=val_metrics,
                                                  device=device,
                                                  prepare_batch=prepare_batch_reshape
                                                  )
    pbar.attach(train_evaluator)

    evaluator = create_supervised_evaluator(model, metrics=val_metrics,
                                            device=device,
                                            prepare_batch=prepare_batch_reshape
                                            )
    pbar.attach(evaluator)

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
                             f"- Mean Square Error: {metrics['mse']:.4f} "
                             # f"- R squared: {metrics['r2score']:.2f}"
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
            to_load = to_save
            last_checkpoint_fp = f'./{temp_name}/{handler.last_checkpoint}'
            checkpoint = torch.load(last_checkpoint_fp, map_location=device)
            CheckPointAfter.load_objects(to_load=to_load, checkpoint=checkpoint)
            logger.info('The best model on validation is reloaded for '
                        'evaluation on the test set')
            train_evaluator.run(test_loader)
            metrics = train_evaluator.state.metrics
            pbar.log_message(f"Testing Results "
                             f"- Epoch: {engine.state.epoch} "
                             f"- Mean Square Error: {metrics['mse']:.4f} "
                             f"- R squared: {metrics['r2score']:.2f}"
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


if __name__ == '__main__':
    sd = 42
    dc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bs = 128
    n_epoch = 100
    pt = False
    exp = 'boneage_augm_no_transfer'
    m_name = 'resnet'

    # r_name = f'{m_name}_lrf'
    # lr_finder(batch_size=bs, alpha=1e-4, end_lr=10, diverge_th=2,
    #           model_name=m_name, pretrain=pt, fold_idx=fold, device=dc,
    #           exp_name=exp, run_name=r_name, seed=sd)

    a = 1e-4
    lrt = 3e-5  # this is from lr finder

    r_name = f'{m_name}_linear_regressor_metric'

    for f in range(5):
        run(batch_size=bs, lr=lrt, alpha=a, model_name=m_name,
            epoch=n_epoch, pretrain=pt, fold_idx=f, device=dc, exp_name=exp,
            run_name=r_name, seed=sd)
        gc.collect()
