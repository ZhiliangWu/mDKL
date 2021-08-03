#
# mDKL
#
# Copyright (c) Siemens AG, 2021
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

from pathlib import Path
import gc
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

import ignite
from ignite.contrib.handlers.mlflow_logger import MLflowLogger, \
    global_step_from_engine
from ignite.contrib.handlers import FastaiLRFinder, ProgressBar
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.engine import Events, create_supervised_trainer, \
    create_supervised_evaluator
from ignite.handlers import DiskSaver, EarlyStopping, Checkpoint
from ignite.metrics import MeanSquaredError

from data_utils import get_tranform, get_augm_data_loaders_in_file, BoneAge, \
    prepare_batch_cae
from logging_conf import logger
from model_utils import get_pretrained_models
from cae_utils import BasicBlock, InvResNet, ConvolutionalAutoencoder


def lr_finder(batch_size, alpha, end_lr=10, diverge_th=5, model_name='resnet',
              pretrain=True, fold_idx=0, device=torch.device('cpu'),
              exp_name='dataset_xxx',
              run_name='model_lrf', seed=42):
    """Find a suitable learning rate.
     More theory see https://www.jeremyjordan.me/nn-learning-rate/.

     Args:
         batch_size (int): Batch size.
         alpha (float): The value of weight decay (a.k.a. regularization).
         end_lr (int or float): The upper bound of the tested learning rate.
         diverge_th (int or float): The threshold for the divergence.
         model_name (str): The name of the backbone.
         pretrain (bool): Whether load the weights in pretrained models.
         fold_idx (int): The index of the training/validation set.
         device (torch.device or str): The device to load the models.
         exp_name (str): The name of the experiments with a format of
             dataset+xxx, which defines the experiment name inside MLflow.
         run_name (str): The name of the run with a format of [model_name]_lrf,
             which defines the run name inside MLflow.
         seed (int): The number of the random seed to ensure the reproducibility.

     Returns:
         None: The plot of the lrfinder will be saved.

     """

    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info(f'Running learning rate finder with {exp_name} on model '
                f'{model_name}.')

    df_path = Path('../boneage.csv')
    im_path = Path(f'{str(Path.home())}/boneage/')

    df_split = pd.read_csv('../boneage_idx_split.csv',
                           index_col=0)

    custom_tranform = get_tranform(mean=(0.183, 0.183, 0.183),
                                   std=(0.166, 0.166, 0.166),
                                   bbox=False)

    train_loader, _, valid_loader, test_loader = \
        get_augm_data_loaders_in_file(df_path, im_path, df_split,
                                      train_batch_size=batch_size,
                                      valid_batch_size=128,
                                      custom_tranform=custom_tranform,
                                      datasetclass=BoneAge,
                                      fname_col='id',
                                      n_fold=fold_idx, augm=False,
                                      target_col='label')

    encoder, _ = get_pretrained_models(model_name='resnet', pretrain=pretrain)
    conv_config = {'kernel_size': 4, 'stride': 4, 'padding': 0}
    decoder = InvResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                        n_input_features=512, conv_config=conv_config,
                        normalize=False)

    model = ConvolutionalAutoencoder(encoder, decoder, unpool_scale=8)
    model = model.to(device)

    # start from small learning rates
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6,
                                 weight_decay=alpha)

    prepare_batch_ba = partial(prepare_batch_cae, key='image')
    trainer = create_supervised_trainer(model, optimizer,
                                        nn.MSELoss(),
                                        device=device,
                                        prepare_batch=prepare_batch_ba)

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, output_transform=lambda x: {'batch loss': x})

    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name=run_name):
        mlflow_logger = MLflowLogger()
        mlflow_logger.log_params({
            'seed': seed,
            'batch_size': batch_size,
            'model': model_name,
            'weight_decay': alpha,
            'fold_index': fold_idx,
            'pytorch version': torch.__version__,
            'ignite version': ignite.__version__,
            'cuda version': torch.version.cuda,
            'device name': torch.cuda.get_device_name(0)
        })

        lf = FastaiLRFinder()
        to_save = {'model': model, 'optimizer': optimizer}
        with lf.attach(trainer,
                       to_save,
                       end_lr=end_lr,
                       diverge_th=diverge_th) as trainer_with_lr_finder:
            trainer_with_lr_finder.run(train_loader)

        lf_log = pd.DataFrame(lf.get_results())

        lf_log.to_csv(f'./temp/lf_log_{exp_name}.csv', index=False)
        mlflow_logger.log_artifact(f'./temp/lf_log_{exp_name}.csv')

        fig, ax = plt.subplots()

        ax.plot(lf_log.lr[:-1], lf_log.loss[:-1])
        ax.set_xscale('log')
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('Loss')
        ax.set_title(f'Suggestion from finder: {lf.lr_suggestion()}')
        fig.savefig(f'./temp/lr_finder_{exp_name}.png', dpi=600)
        plt.close(fig)

        mlflow_logger.log_artifact(f'./temp/lr_finder_{exp_name}.png')


def run_cae(batch_size=64, lr=3e-4, alpha=1e-3, cae_loss=nn.MSELoss(),
            model_name='resnet', epoch=50, pretrain=True, fold_idx=0,
            device=torch.device('cpu'), exp_name='dataset_xxx',
            run_name='model_xxx', seed=42):
    """Run the experiment with a given setting.

    Args:
        batch_size (int): Batch size.
        lr (float): The value of the learning rate, possibly from lrfinder.
        alpha (float): The value of weight decay (a.k.a. regularization).
        cae_loss (nn.Module): The loss used to train the CAE.
        model_name (str): The name of the backbone.
        epoch (int): The number of training epochs.
        pretrain (bool): Whether load the weights in pretrained models.
        fold_idx (int): The index of the training/validation set.
        device (torch.device or str): The device to load the models.
        exp_name (str): The name of the experiments with a format of
            dataset+xxx, which defines the experiment name inside MLflow.
        run_name (str): The name of the run with a format of
            [model_name]_cae, which defines the run name inside
            MLflow.
        seed (int): The number of the random seed to ensure the reproducibility.

    Returns:
        (nn.Module, int): a sota cnn backbone pretrained with metric learning
            and the number of features of the backbone.

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

    train_loader, _, valid_loader, test_loader = \
        get_augm_data_loaders_in_file(df_path, im_path, df_split,
                                      train_batch_size=batch_size,
                                      valid_batch_size=128,
                                      custom_tranform=custom_tranform,
                                      datasetclass=BoneAge,
                                      fname_col='id',
                                      n_fold=fold_idx, augm=False,
                                      target_col='label')

    encoder, n_feature = get_pretrained_models(model_name='resnet',
                                               pretrain=pretrain)

    conv_config = {'kernel_size': 4, 'stride': 4, 'padding': 0}
    decoder = InvResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                        n_input_features=512, conv_config=conv_config,
                        normalize=False)

    model = ConvolutionalAutoencoder(encoder, decoder, unpool_scale=8)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=alpha)

    # step_scheduler = StepLR(optimizer, step_size=int(epoch/2), gamma=0.1)
    # scheduler = LRScheduler(step_scheduler)

    def train_output_transform(x, y, y_pred, loss):
        return {'y': y, 'y_pred': y_pred, 'loss': loss.item()}

    prepare_batch_ba = partial(prepare_batch_cae, key='image')

    trainer = create_supervised_trainer(model, optimizer,
                                        cae_loss,
                                        device=device,
                                        prepare_batch=prepare_batch_ba,
                                        output_transform=train_output_transform
                                        )

    # trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer,
                output_transform=lambda out: {'batch loss': out['loss']})

    mse = MeanSquaredError() / (256 * 256 * 3)
    # evaluators
    val_metrics = {'mse': mse,
                   }

    for name, metric in val_metrics.items():
        metric.attach(trainer, name)

    evaluator = create_supervised_evaluator(model,
                                            metrics=val_metrics,
                                            device=device,
                                            prepare_batch=prepare_batch_ba
                                            )
    pbar.attach(evaluator)

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
                             )
            log_metrics = {f'validation {k}': v for k, v in metrics.items()}
            mlflow_logger.log_metrics(log_metrics, step=engine.state.epoch)

        def score_function(engine):
            return -engine.state.metrics['mse']

        handler = EarlyStopping(patience=15, score_function=score_function,
                                trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, handler)

        to_save = {'model': model}

        save_handler = Checkpoint(to_save=to_save,
                                  save_handler=DiskSaver('./temp/cae_models',
                                                         create_dir=True,
                                                         require_empty=False),
                                  n_saved=1,
                                  filename_prefix='best',
                                  score_function=score_function,
                                  score_name="val_mse",
                                  global_step_transform=global_step_from_engine(trainer)
                                  )

        evaluator.add_event_handler(Events.COMPLETED, save_handler)

        def load_best_model(engine):
            to_load = {'model': model}
            last_checkpoint_fp = f'./temp/cae_models/{save_handler.last_checkpoint}'
            print(last_checkpoint_fp)
            checkpoint = torch.load(last_checkpoint_fp, map_location=device)
            Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)
            logger.info('The best model on validation is reloaded.')

            ###
            n_images = 6

            fig, axes = plt.subplots(nrows=2, ncols=n_images,
                                     sharex=True, sharey=True,
                                     figsize=(15, 2.5))
            img_examples = next(iter(test_loader))['image'][:n_images]
            img_examples = img_examples.to(device, dtype=torch.float)
            orig_images = img_examples
            model.eval()
            with torch.no_grad():
                decoded_images = model(img_examples)

            # un-normalize images
            orig_images.mul_(0.166).add_(0.183)
            decoded_images.mul_(0.166).add_(0.183)

            for i in range(n_images):
                for ax, img in zip(axes, [orig_images, decoded_images]):
                    curr_img = img[i].detach().to(torch.device('cpu'))
                    ax[i].imshow(curr_img.permute(1, 2, 0))
                    # avoid horizontal/vertical white lines shown in images
                    ax[i].set_axis_off()

            save_fp = f'./temp/cae_models/' \
                      f'{save_handler.last_checkpoint[:-3]}.pdf'
            fig.savefig(save_fp, dpi=600)
            plt.close(fig)
            mlflow_logger.log_artifact(save_fp)

        trainer.add_event_handler(Events.COMPLETED, load_best_model)

        # handlers for trainer
        @trainer.on(Events.EPOCH_COMPLETED(every=10))
        def log_training_results(engine):
            metrics = engine.state.metrics
            pbar.log_message(f"Training Set "
                             f"- Epoch: {engine.state.epoch} "
                             f"- Mean Square Error: {metrics['mse']:.4f} "
                             )

            #######################
            n_images = 6

            fig, axes = plt.subplots(nrows=2, ncols=n_images,
                                     sharex=True, sharey=True,
                                     figsize=(15, 2.5))
            img_examples = next(iter(test_loader))['image'][:n_images]
            img_examples = img_examples.to(device, dtype=torch.float)
            orig_images = img_examples
            model.eval()
            with torch.no_grad():
                decoded_images = model(img_examples)

            # un-normalize images
            orig_images.mul_(0.166).add_(0.183)
            decoded_images.mul_(0.166).add_(0.183)

            for i in range(n_images):
                for ax, img in zip(axes, [orig_images, decoded_images]):
                    curr_img = img[i].detach().to(torch.device('cpu'))
                    ax[i].imshow(curr_img.permute(1, 2, 0))
                    # avoid horizontal/vertical white lines shown in images
                    ax[i].set_axis_off()

            save_fp = f'./temp/cae_models/epoch_{engine.state.epoch}.pdf'
            fig.savefig(save_fp, dpi=600)
            plt.close(fig)
            mlflow_logger.log_artifact(save_fp)
            ########################

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
                                            metric_names=['mse',]
                                            )

        # Attach the logger to the trainer to log optimizer's parameters,
        # e.g. learning rate at each iteration
        mlflow_logger.attach_opt_params_handler(trainer,
                                                event_name=Events.ITERATION_STARTED,
                                                optimizer=optimizer,
                                                param_name='lr'
                                                )

        _ = trainer.run(train_loader, max_epochs=epoch)
        
        return model, n_feature


if __name__ == '__main__':
    sd = 42
    dc = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    fold = 0
    bs = 128
    n_epoch = 200
    pt = True
    exp = 'boneage_final'
    m_name = 'resnet'
    #
    # r_name = f'{m_name}_cae_lrf'
    # lr_finder(batch_size=bs, alpha=1e-4, end_lr=10, diverge_th=2,
    #           model_name=m_name, pretrain=pt, fold_idx=fold, device=dc,
    #           exp_name=exp, run_name=r_name, seed=sd)

    lrt = 3e-4  # this is from lr finder

    r_name = f'{m_name}_cae'

    for a in [1e-5, ]:
        run_cae(batch_size=bs, lr=lrt, alpha=a, model_name=m_name,
                epoch=n_epoch, pretrain=pt, fold_idx=fold, device=dc,
                exp_name=exp, run_name=r_name, seed=sd)
        gc.collect()
