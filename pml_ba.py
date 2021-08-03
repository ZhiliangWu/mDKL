#
# mDKL
#
# Copyright (c) Siemens AG, 2021
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

from pathlib import Path

import faiss
import numpy as np
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import ignite
from ignite.contrib.handlers import FastaiLRFinder, ProgressBar
from ignite.contrib.handlers.mlflow_logger import MLflowLogger
from ignite.metrics import Average
from ignite.engine import Events
from ignite.handlers import EarlyStopping

from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from data_utils import get_tranform, get_augm_data_loaders_in_file, BoneAge, \
    prepare_batch
from model_utils import get_pretrained_models, create_metric_trainer


def lr_finder(batch_size=32, alpha=1e-4, end_lr=10, diverge_th=5,
              model_name='resnet', pretrain=True, fold_idx=0, device='cpu',
              exp_name='dataset_xxx', run_name='xmodel_metric_lrf', seed=42):
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

    df_path = Path('../boneage.csv')
    im_path = Path(f'{str(Path.home())}/boneage/')

    df_split = pd.read_csv('../boneage_idx_split.csv',
                           index_col=0)

    custom_tranform = get_tranform(mean=(0.183, 0.183, 0.183),
                                   std=(0.166, 0.166, 0.166),
                                   bbox=False)

    train_loader, _, _, _ = \
        get_augm_data_loaders_in_file(df_path, im_path, df_split,
                                      train_batch_size=batch_size,
                                      valid_batch_size=128,
                                      custom_tranform=custom_tranform,
                                      datasetclass=BoneAge,
                                      fname_col='id',
                                      n_fold=fold_idx, augm=False,
                                      target_col='label')

    model, n_feature = get_pretrained_models(model_name=model_name,
                                             pretrain=pretrain)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6,
                                 weight_decay=alpha)

    # pytorch-metric-learning stuff
    distance = distances.LpDistance()
    reducer = reducers.MeanReducer()
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance,
                                         reducer=reducer)
    mining_func = miners.TripletMarginMiner(margin=0.2, distance=distance,
                                            type_of_triplets="semihard")
    # pytorch-metric-learning stuff

    train_output_transform = lambda x, y, embeddings, num_triplets, loss: loss.item()
    trainer = create_metric_trainer(model, optimizer,
                                    loss_func, mining_func,
                                    device=device,
                                    prepare_batch=prepare_batch,
                                    output_transform=train_output_transform
                                    )

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer,
                output_transform=lambda out: {'batch loss': out})

    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=run_name):
        mlflow_logger = MLflowLogger()
        mlflow_logger.log_params({
            'seed': seed,
            'batch_size': batch_size,
            'model': model_name,
            # 'gp_input_dim': dim_gp_in,
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


def run_metric_learning(batch_size=32, lr=1e-3, alpha=1e-4,
                        model_name='resnet', epoch=1, pretrain=True,
                        fold_idx=0, device='cpu',
                        exp_name='dataset_xxx', run_name='model_metric',
                        seed=42):
    """Run the metric learning with a given setting.

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
            [model_name]_metric, which defines the run name inside
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
                                      valid_batch_size=64,
                                      custom_tranform=custom_tranform,
                                      datasetclass=BoneAge,
                                      fname_col='id',
                                      n_fold=fold_idx, augm=False,
                                      target_col='label')

    model, n_feature = get_pretrained_models(model_name=model_name,
                                             pretrain=pretrain)
    # model = LinearModel(feature_extractor=backbone, num_features=n_feature,
    #                     output_dim=dim_gp_in)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=alpha)

    # pytorch-metric-learning stuff
    distance = distances.LpDistance()
    reducer = reducers.MeanReducer()
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance,
                                         reducer=reducer)
    mining_func = miners.TripletMarginMiner(margin=0.2, distance=distance,
                                            type_of_triplets='semihard')
    accuracy_calculator = AccuracyCalculator(include=("mean_average_precision_at_r",), k=10)
    # pytorch-metric-learning stuff

    def train_output_transform(x, y, embeddings, num_triplets, loss):
        return {'n_triplets': num_triplets, 'loss': loss.item()}

    # prepare_batch_reshape = partial(prepare_batch, new_shape=(-1, 1))
    trainer = create_metric_trainer(model, optimizer,
                                    loss_func, mining_func,
                                    device=device,
                                    prepare_batch=prepare_batch,
                                    output_transform=train_output_transform
                                    )

    train_metrics = {'loss': Average(output_transform=lambda x: x['loss']),
                     'n_triplets': Average(output_transform=lambda x: x['n_triplets']),
                     }

    for name, metric in train_metrics.items():
        metric.attach(trainer, name)

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer,
                output_transform=lambda out: {'batch loss': out['loss']})

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

        # handlers for trainer
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            metrics = engine.state.metrics
            pbar.log_message(f"Training Set "
                             f"- Epoch: {engine.state.epoch} "
                             f"- average loss: {metrics['loss']:.4f} "
                             f"- average n_triplets: {metrics['n_triplets']:.2f}"
                             )

        @trainer.on(Events.EPOCH_COMPLETED)
        def evaluate_model(engine):
            data_and_label_getter = lambda x: (x['image'], x['target'])
            tester = testers.BaseTester(reference_set="compared_to_self",
                                        normalize_embeddings=True,
                                        dataloader_num_workers=4,
                                        data_device=device,
                                        data_and_label_getter=data_and_label_getter)

            model.eval()
            embeddings, labels = tester.compute_all_embeddings(valid_loader,
                                                               model,
                                                               nn.Identity())
            embeddings = tester.maybe_normalize(embeddings)

            accuracies = accuracy_calculator.get_accuracy(embeddings,
                                                          embeddings,
                                                          np.squeeze(labels),
                                                          np.squeeze(labels),
                                                          True)

            print(f"Valid MAP10 = {accuracies['mean_average_precision_at_r']}")
            engine.state.metrics['MAP10'] = accuracies['mean_average_precision_at_r']
            mlflow.log_metric("validation MAP10",
                              accuracies['mean_average_precision_at_r'],
                              step=engine.state.epoch
                              )

        def score_function(engine):
            val_metric = engine.state.metrics['MAP10']
            return val_metric

        handler = EarlyStopping(patience=5, score_function=score_function,
                                trainer=trainer)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

        mlflow_logger.attach_output_handler(trainer,
                                            event_name=Events.EPOCH_COMPLETED,
                                            tag='training',
                                            metric_names=['loss',
                                                          'n_triplets',
                                                          ]
                                            )

        mlflow_logger.attach_opt_params_handler(trainer,
                                                event_name=Events.ITERATION_STARTED,
                                                optimizer=optimizer,
                                                param_name='lr'
                                                )

        _ = trainer.run(train_loader, max_epochs=epoch)

        return model, n_feature


if __name__ == '__main__':
    sd = 42
    dc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    f = 0
    bs = 128
    n_epoch = 100
    pt = True
    exp = 'boneage_augm'
    m_name = 'resnet'

    a = 1e-4

    # r_name = f'{m_name}_metric_lrf'
    # lr_finder(batch_size=bs, alpha=a, end_lr=10, diverge_th=2,
    #          model_name=m_name, pretrain=pt, fold_idx=f, exp_name=exp,
    #          run_name=r_name, device=dc, seed=sd)

    lrt = 1e-4
    r_name = f'{m_name}_metric'

    _ = run_metric_learning(batch_size=bs, lr=lrt, alpha=a, model_name=m_name,
                            epoch=n_epoch, pretrain=pt, fold_idx=f,
                            exp_name=exp, run_name=r_name, device=dc, seed=sd)
