#
# mDKL
#
# Copyright (c) Siemens AG, 2021
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import re
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, BasicBlock

# always use the one from torchvision except for mcdropout evlauation
from torchvision.models.densenet import DenseNet
# from densenet import DenseNet
from pytorch_metric_learning import losses, miners

import ignite.distributed as idist
from ignite.engine.deterministic import DeterministicEngine
from ignite.engine.engine import Engine, Events
from ignite.handlers import Checkpoint
from ignite.metrics import Metric
from ignite.utils import convert_tensor

if idist.has_xla_support:
    import torch_xla.core.xla_model as xm


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'densenet': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
}


class FeatureResNet(ResNet):
    """The backbone of a ResNet."""
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x


class FeatureResNetMCDropOut(ResNet):
    """The backbone of a ResNet with dropout."""
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = torch.nn.functional.dropout(x, p=0.2)
        x = self.layer2(x)
        x = torch.nn.functional.dropout(x, p=0.2)
        x = self.layer3(x)
        x = torch.nn.functional.dropout(x, p=0.2)
        x = self.layer4(x)
        x = torch.nn.functional.dropout(x, p=0.2)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x


class FeatureDenseNet(DenseNet):
    """The backbone of a densenet"""
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        # out = self.classifier(out)
        return out


def get_pretrained_models(model_name='resnet', pretrain=True, dropout=False):
    """Get a sota CNN backbone, whose weights are pretrained on ImageNet.

    Args:
        model_name (str): The name of a sota model, either resnet or dense.
        pretrain (boolean): Whether load weights of pretrained models.
        dropout (boolean): Whether activate dropout in the model.

    Returns:
        (nn.Module, int):  A sota CNN backbone, the number of features of the
            backbone.

    """

    if model_name == 'resnet':
        if dropout:
            model = FeatureResNetMCDropOut(BasicBlock, [2, 2, 2, 2])
        else:
            model = FeatureResNet(BasicBlock, [2, 2, 2, 2])

        if pretrain:
            state_dict = load_state_dict_from_url(model_urls['resnet18'])
            model.load_state_dict(state_dict)
        num_features = model.fc.in_features

    elif model_name == 'dense':
        if dropout:
            model = FeatureDenseNet(32, (6, 12, 24, 16), 64, drop_rate=0.2)
        else:
            model = FeatureDenseNet(32, (6, 12, 24, 16), 64)

        # from https://github.com/pytorch/vision/blob/a75fdd4180683f7953d97ebbcc92d24682690f96/torchvision/models/densenet.py#L200

        if pretrain:
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

            state_dict = load_state_dict_from_url(model_urls['densenet'])

            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

            model.load_state_dict(state_dict)

        num_features = model.classifier.in_features

    else:
        raise ValueError("Input model not defined")

    try:
        del model.classifier
    except AttributeError:
        # resnet has not classifier attribute
        del model.fc

    return model, num_features


class LinearModel(nn.Module):
    """Add a linear layer after the backbone."""
    def __init__(self, feature_extractor, num_features, output_dim=2):
        """

        Args:
            feature_extractor (nn.Module): The backbone of the model, used as a
                feature extractor.
            num_features (int): The number of features from the backbone.
            output_dim (int): The output dimension of the new model.
        """
        super(LinearModel, self).__init__()
        self.features = feature_extractor
        self.fc = nn.Linear(num_features, output_dim)

    def forward(self, x):
        """Defines the forward pass of the new model."""
        features = self.features(x)
        out = self.fc(features)

        return out


def get_model(name='resnet', pretrain=True, output_dim=1, dropout=False):
    """Get a sota CNN model with custmized number of outputs.

    Args:
        name (str): The name of a sota model, either resnet or dense.
        pretrain (boolean): Whether load weights of pretrained models.
        output_dim (int): The number of target variables,.
        dropout (boolean): Whether activate dropout in the model.

    Returns:
        nn.Module: A sota CNN model.

    """
    m, n_feat = get_pretrained_models(model_name=name, pretrain=pretrain,
                                      dropout=dropout)
    model = LinearModel(feature_extractor=m, num_features=n_feat,
                        output_dim=output_dim)

    return model


class DKLModel(nn.Module):
    """Defines a DKL model with deep networks as a feature extractor and a
    GP based output layer for prediction."""
    def __init__(self, feature_extractor, gp_layer):
        """

        Args:
            feature_extractor (nn.Module): A sota feature extractor.
            gp_layer (nn.Module): A sota GP based output layer.
        """
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = gp_layer

    def forward(self, x):
        features = self.feature_extractor(x)
        res = self.gp_layer(features)

        return res


class Mock(BaseEstimator):
    """Mock a BaseEstimator with defined prediction values."""

    _estimator_type = "regressor"    # Tell yellowbrick this is a regressor

    def __init__(self, y_pred_train, y_pred_test):
        self.y_pred_train = y_pred_train
        self.y_pred_test = y_pred_test

    def predict(self, is_train=True):
        """X indicates whether prediction on train or not
        """
        if is_train:
            output = self.y_pred_train
        else:
            output = self.y_pred_test

        return output

    def score(self, X, y, sample_weight=None):

        y_pred = self.predict(X)

        return r2_score(y, y_pred, sample_weight=sample_weight)


class EpochOutputStore(object):
    """EpochOutputStore handler to save output prediction and target history
    after every epoch."""

    def __init__(self, output_transform=lambda x: x):
        """

        Args:
            output_transform (Callable): Transform the process_function's
            output_transform (Callable): Transform the process_function's
                output , e.g., lambda x: x[0].
        """
        self.predictions = None
        self.targets = None
        self.output_transform = output_transform

    def reset(self):
        self.predictions = []
        self.targets = []

    def update(self, engine):
        y_pred, y = self.output_transform(engine.state.output)
        self.predictions.append(y_pred)
        self.targets.append(y)

    def attach(self, engine):
        engine.add_event_handler(Events.EPOCH_STARTED, self.reset)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.update)

    def get_output(self, to_numpy=False):
        """Get the total output in torch.tensor or np.array."""
        prediction_tensor = torch.cat(self.predictions, dim=0)
        target_tensor = torch.cat(self.targets, dim=0)

        if to_numpy:
            prediction_tensor = prediction_tensor.cpu().detach().numpy().flatten()
            target_tensor = target_tensor.cpu().detach().numpy().flatten()

        return prediction_tensor, target_tensor


class CheckPointAfter(Checkpoint):
    """Save the model after a defined epoch."""
    def __init__(self, start_epoch, *args, **kwargs):
        self.start_save_epoch = start_epoch
        print(f'start saving after {self.start_save_epoch}')
        super(CheckPointAfter, self).__init__(*args, **kwargs)

    def __call__(self, engine):
        global_step = self.global_step_transform(engine, engine.last_event_name)
        if global_step > self.start_save_epoch:
            super(CheckPointAfter, self).__call__(engine)
        else:
            print('skipping checkpoints...')


def get_initial_inducing_points(feature_extractor, train_loader, device,
                                num_inducing=5):
    """Generate initial inducing points using a backbone

    Args:
        feature_extractor (nn.Module): A backbone to generate features.
        train_loader (DataLoader): Dataloader of the
            training set.
        device (torch.device or str): Device to load the backbone and data.
        num_inducing (int): The multiple of batch size.
            The number of inducing points is (num_inducing x batch_size).

    Returns:
        torch.Tensor: The initial inducing points

    """
    feature_extractor.eval()
    inducing_points_list = []

    for i in range(num_inducing):
        with torch.no_grad():
            current_batch = next(iter(train_loader))['image'].to(device)
            inducing_points = feature_extractor(current_batch)
            inducing_points_list.append(inducing_points)

    initial_inducing_points = torch.cat(inducing_points_list, dim=0)

    return initial_inducing_points


################################################################################
"""Following are modified functions from ignite to facilitate the DKL 
training."""


def _prepare_batch(batch: Sequence[torch.Tensor],
                   device: Optional[Union[str, torch.device]] = None,
                   non_blocking: bool = False):
    """Prepare batch for training: pass to a device with options."""

    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


def create_dkl_trainer(
    model: torch.nn.Module,
    likelihood: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    mll: Union[Callable, torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred, loss: loss.item(),
    deterministic: bool = False,
) -> Engine:

    device_type = device.type if isinstance(device, torch.device) else device
    on_tpu = "xla" in device_type if device_type is not None else False

    if on_tpu and not idist.has_xla_support:
        raise RuntimeError("In order to run on TPU, please install PyTorch XLA")

    def _update(engine: Engine,
                batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        likelihood.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        output = model(x)
        y_pred = output.mean.detach()
        loss = -mll(output, y)
        # loss = loss_fn(y_pred, y)
        loss.backward()

        if on_tpu:
            xm.optimizer_step(optimizer, barrier=True)
        else:
            optimizer.step()

        return output_transform(x, y, y_pred, loss)

    trainer = Engine(_update) if not deterministic else DeterministicEngine(_update)

    return trainer


def create_dkl_evaluator(
    model: torch.nn.Module,
    likelihood: torch.nn.Module,
    metrics: Optional[Dict[str, Metric]] = None,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred: (y_pred, y),
) -> Engine:

    metrics = metrics or {}

    def _inference(engine: Engine,
                   batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            output = model(x)
            y_pred = output.mean
            return output_transform(x, y, y_pred)

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def create_dkl_cae_trainer(
    model: torch.nn.Module,
    likelihood: torch.nn.Module,
    cae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    mll: Union[Callable, torch.nn.Module],
    cae_loss: Union[Callable, torch.nn.Module],
    cae_coeff: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred, loss_gp, loss_cae:
    loss_gp.item(),
    deterministic: bool = False,
) -> Engine:

    device_type = device.type if isinstance(device, torch.device) else device
    on_tpu = "xla" in device_type if device_type is not None else False

    if on_tpu and not idist.has_xla_support:
        raise RuntimeError("In order to run on TPU, please install PyTorch XLA")

    def _update(engine: Engine,
                batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        likelihood.train()
        cae.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        output = model(x)
        y_pred = output.mean.detach()
        loss_gp = -mll(output, y)
        recon = cae(x)
        loss_cae = cae_loss(recon, x)
        loss = loss_gp + cae_coeff * loss_cae
        loss.backward()

        if on_tpu:
            xm.optimizer_step(optimizer, barrier=True)
        else:
            optimizer.step()

        return output_transform(x, y, y_pred, loss_gp, loss_cae)

    trainer = Engine(_update) if not deterministic else DeterministicEngine(_update)

    return trainer


def create_dkl_cae_evaluator(
    model: torch.nn.Module,
    likelihood: torch.nn.Module,
    cae: torch.nn.Module,
    metrics: Optional[Dict[str, Metric]] = None,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred, recon: (y, y_pred),
) -> Engine:

    metrics = metrics or {}

    def _inference(engine: Engine,
                   batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        likelihood.eval()
        cae.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            output = model(x)
            y_pred = output.mean
            recon = cae(x)
            return output_transform(x, y, y_pred, recon)

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def create_metric_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    mining_function: miners.BaseMiner = None,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred, loss: loss.item(),
    deterministic: bool = False,
) -> Engine:

    device_type = device.type if isinstance(device, torch.device) else device
    on_tpu = "xla" in device_type if device_type is not None else False

    if on_tpu and not idist.has_xla_support:
        raise RuntimeError("In order to run on TPU, please install PyTorch XLA")

    def _update(engine: Engine,
                batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        embeddings = model(x)
        indices_tuple = mining_function(embeddings, y)
        loss = loss_fn(embeddings, y, indices_tuple)
        loss.backward()

        if on_tpu:
            xm.optimizer_step(optimizer, barrier=True)
        else:
            optimizer.step()

        return output_transform(x, y, embeddings,
                                mining_function.num_triplets,
                                loss)

    trainer = Engine(_update) if not deterministic else DeterministicEngine(
        _update)

    return trainer


if __name__ == '__main__':
    pass


