#
# mDKL
#
# Copyright (c) Siemens AG, 2021
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

import gc
import copy
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

from data_utils import AugmBBoxDatasetAll, get_tranform, \
    get_augm_data_loaders_in_file, prepare_batch
from gp_layer import IMTSVGPLayer
from model_utils import get_pretrained_models, LinearModel, \
    get_initial_inducing_points, DKLModel, create_dkl_trainer, \
    create_dkl_evaluator, EpochOutputStore, CheckPointAfter
from logging_conf import logger
from cae_dl import run_cae
from plot_utils import joint_plot_xy, residual_plot


def run(batch_size=64, lr=3e-4, alpha=0.0, num_gp=4, num_inducing=10,
        dim_gp_in=50, model_name='resnet', epoch=50, pretrain=True, fold_idx=0,
        device=torch.device('cpu'), exp_name='dataset_xxx',
        run_name='model_xxx', seed=42):
    """Run the experiment with cae as a pre-training step.

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

    cae, n_feature = run_cae(batch_size=batch_size,
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

    feature_extractor = LinearModel(feature_extractor=cae.encoder,
                                    num_features=n_feature,
                                    output_dim=dim_gp_in)
    feature_extractor = feature_extractor.to(device)

    df_path = Path('../DL_lung_0.5.csv')
    im_path = Path(f'{str(Path.home())}/Key_DL/')

    df_split = pd.read_csv('../DL_lung_0.5_idx_split.csv',
                           index_col=0)

    custom_tranform = get_tranform(mean=(0.389, 0.389, 0.389),
                                   std=(0.240, 0.240, 0.240),
                                   bbox=True)

    train_loader, train_evaluator_loader, valid_loader, test_loader = \
        get_augm_data_loaders_in_file(df_path, im_path, df_split,
                                      train_batch_size=batch_size,
                                      valid_batch_size=128,
                                      custom_tranform=custom_tranform,
                                      datasetclass=AugmBBoxDatasetAll,
                                      fname_col='File_name',
                                      n_fold=fold_idx,
                                      augm=True,
                                      albu_format=True)

    # shape is of (#inducing_points x gp_input_dim)
    inducing_points = get_initial_inducing_points(feature_extractor,
                                                  train_loader,
                                                  device,
                                                  num_inducing=num_inducing)

    # turn into (n_gp x #inducing_points x gp_input_dim)
    # initialize the inducing point for x and y as the same ones
    # for independent gps, each gp corresponds to one task / output
    multi_inducing_points = torch.stack([inducing_points for _ in range(num_gp)],
                                        dim=0)

    mt_gp_layer = IMTSVGPLayer(inducing_points=multi_inducing_points,
                               num_tasks=4)

    model = DKLModel(feature_extractor=feature_extractor, gp_layer=mt_gp_layer)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=4)

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
        return {'both': (y_pred, y),
                'x': (y_pred[:, 0], y[:, 0]),
                'y': (y_pred[:, 1], y[:, 1]),
                'w': (y_pred[:, 2], y[:, 2]),
                'h': (y_pred[:, 3], y[:, 3]),
                'loss': loss.item()
                }

    trainer = create_dkl_trainer(model, likelihood, optimizer, mll,
                                 device=device,
                                 prepare_batch=prepare_batch,
                                 output_transform=train_output_transform)

    train_metrics = {'mll': Average(output_transform=lambda x: -x['loss']),
                     'mse': MeanSquaredError(output_transform=lambda out: out['both']),
                     'r2score_x': R2Score(output_transform=lambda out: out['x']),
                     'r2score_y': R2Score(output_transform=lambda out: out['y']),
                     'r2score_w': R2Score(output_transform=lambda out: out['w']),
                     'r2score_h': R2Score(output_transform=lambda out: out['h'])
                     }

    # add metrics to trainer, evaluateing based on a moving model / moving data
    for name, metric in train_metrics.items():
        metric.attach(trainer, name)

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer,
                output_transform=lambda out: {'batch mll': -out['loss']})

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

    # evlauating the model on the training dataset
    train_evaluator = create_dkl_evaluator(model, likelihood,
                                           metrics=val_metrics,
                                           device=device,
                                           prepare_batch=prepare_batch,
                                           output_transform=eva_output_tranform
                                           )

    pbar.attach(train_evaluator)

    evaluator = create_dkl_evaluator(model, likelihood, metrics=val_metrics,
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
                   'likelihood': likelihood,
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
                                  global_step_transform=global_step_from_engine(
                                      trainer))

        evaluator.add_event_handler(Events.COMPLETED, handler)

        # handlers for trainer
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            metrics = engine.state.metrics
            pbar.log_message(f"Moving Training Set "
                             f"- Epoch: {engine.state.epoch} "
                             f"- MLL: {metrics['mll']:.4f}  "
                             f"- Mean Square Error: {metrics['mse']:.4f} "
                             f"- R squared of x: {metrics['r2score_x']:.2f}"
                             f"- R squared of y: {metrics['r2score_y']:.2f}"
                             f"- R squared of w: {metrics['r2score_w']:.2f}"
                             f"- R squared of h: {metrics['r2score_h']:.2f}"
                             )

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
            # due to gpytorch has possibly a bug of loading saved model to
            # training models directly, a deep copied version of the original
            # model is used here for loading and evaluation
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
            test_r2score = r2_score(y_true, y_pred, multioutput='raw_values')

            pbar.log_message(f"Testing Results "
                             f"- Epoch: {engine.state.epoch} "
                             f"- Mean Square Error: {test_mse*4:.4f} "
                             f"- R squared of x: {test_r2score[0]:.2f}"
                             f"- R squared of y: {test_r2score[1]:.2f}"
                             f"- R squared of w: {test_r2score[2]:.2f}"
                             f"- R squared of h: {test_r2score[3]:.2f}"
                             )
            log_metrics = {'test mse': test_mse * 4,
                           'test r2score_x': test_r2score[0],
                           'test r2score_y': test_r2score[1],
                           'test r2score_w': test_r2score[2],
                           'test r2score_h': test_r2score[3],
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


if __name__ == '__main__':
    sd = 42
    dc = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    fold = 0
    bs = 128
    n_epoch = 100
    pt = True
    n_gp = 4
    n_idc = 10
    dim_gp_input = 50

    a = 1e-4  # a small value is a good default
    lrt = 1e-4  # this is for cnn parameters
    exp = 'DL_lung_final'
    m_name = 'resnet'

    r_name = f'{m_name}_ppgp_albu_cae'

    for f in range(5):
        run(batch_size=bs, lr=lrt, alpha=a, num_gp=n_gp, num_inducing=n_idc,
            dim_gp_in=dim_gp_input, model_name=m_name, epoch=n_epoch,
            pretrain=pt, fold_idx=f, device=dc, exp_name=exp,
            run_name=r_name, seed=sd)
        gc.collect()
