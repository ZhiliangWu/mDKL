#
# mDKL
#
# Copyright (c) Siemens AG, 2021
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

from typing import Callable

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
from sklearn.model_selection import train_test_split, KFold

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler

from logging_conf import logger


class BoneAge(Dataset):
    """The BoneAge dataset."""

    def __init__(self, csv_file, root_dir, transform=None,
                 fname_col='id', target_col='age_normalize'):
        """
        Args:
            csv_file (str): Path to the csv file with annotations.
            root_dir (pathlib.Path): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            fname_col (str): Column name for the image file location.
            target_col (list): Columns containing the targets for prediction
        """
        self.df = pd.read_csv(csv_file, index_col=0)
        self.root_dir = root_dir
        self.transform = transform
        self.fname_col = fname_col
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        img_fpath = self.root_dir / f'{self.df.iloc[idx][self.fname_col]}.png'
        image = cv2.imread(str(img_fpath))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target = self.df.iloc[idx][[self.target_col]].values
        target = target.astype('float').item()

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        sample = {'image': image, 'target': target}

        return sample


class AugmBBoxDatasetAll(Dataset):
    """An augmented bounding box dataset using albumentations."""

    def __init__(self, csv_file, root_dir, transform=None,
                 fname_col='image_name', albu_format=False,
                 target_col=None):
        """
        Args:
            csv_file (str): Path to the csv file with annotations.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            fname_col (str): Column name for the image file location
            albu_format (bool): Whether use the albumentations format.
            target_col (str): Discrete label, used for metric learning only
        """
        self.bbox_df = pd.read_csv(csv_file, index_col=0)
        self.root_dir = root_dir
        self.transform = transform
        self.fname_col = fname_col
        self.albu_format = albu_format
        self.target_col = target_col

    def __len__(self):
        return len(self.bbox_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        img_fpath = self.root_dir / self.bbox_df.iloc[idx][self.fname_col]
        image = cv2.imread(str(img_fpath))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        yolo_columns = ['x1n', 'y1n', 'x2n', 'y2n']
        bbox = self.bbox_df.iloc[idx][yolo_columns].tolist()
        # fix some strange negative values in data
        bbox = [v if v > 0 else 0 for v in bbox]
        class_label = ['lesion', ]

        transformed = self.transform(image=image, bboxes=[bbox],
                                     class_labels=class_label)

        # repeat transformation until there is bbox inside
        while not transformed['bboxes']:
            transformed = self.transform(image=image, bboxes=[bbox],
                                         class_labels=class_label)

        x1, y1, x2, y2 = transformed['bboxes'][0]

        if self.albu_format:
            sample = {'image': transformed['image'],
                      'target': torch.FloatTensor([x1, y1, x2, y2])
                      }
        else:
            # otherwise use the yolo format
            x_c = (x1 + x2) / 2
            y_c = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            sample = {'image': transformed['image'],
                      'target': torch.FloatTensor([x_c, y_c, w, h])
                      }

        if self.target_col:
            # new idea, load the kmeans and compute the cluster online
            kmeans = load('../kmeans_4_origin_targets_dl.joblib')
            label = kmeans.predict(np.array([x1, y1, x2, y2]).reshape(1, -1))
            sample = {'image': transformed['image'],
                      'target': label.item()
                      }

        return sample


def do_train_valid_test_split(dataframe, id_col='uid', test_ratio=0.1,
                              cross=False):
    """

    Args:
        dataframe (pd.DataFrame): The DataFrame with an column of sample ids.
        id_col (str): The column name of the id.
        test_ratio (int): The ratio of the test set
        cross (bool): Whether do five-fold cross-validation split.

    Returns:
        pd.DataFrame: A split df with multiple columns indicating the label
            of each sample (train, valid, test).

    Example:
        >>> fn = './DL_lung_0.5.csv'
        >>> df = pd.read_csv(fn, index_col=0)
        >>> # sanity check
        >>> # df = pd.DataFrame(data=np.arange(100), columns=['uid'])
        >>> df_split = do_train_valid_test_split(df)
        >>> df_split.to_csv(f'{fn[:-4]}_idx_split.csv')
    """

    data_df = dataframe.reset_index()
    num_samples = len(data_df)
    indices = np.arange(num_samples)
    train_idx, test_idx = train_test_split(indices, test_size=test_ratio,
                                           random_state=42)

    df_split = data_df.loc[:, [id_col]]
    if cross:
        kf = KFold(n_splits=5, shuffle=False)

        for i, (t, v) in enumerate(kf.split(train_idx)):
            train = train_idx[t]
            valid = train_idx[v]
            fold_name = f'fold_{i}'
            df_split[fold_name] = 0
            df_split.loc[train, fold_name] = 1
            df_split.loc[valid, fold_name] = 2
            df_split.loc[test_idx, fold_name] = 3
    else:
        for i in range(5):
            train, valid = train_test_split(train_idx, test_size=test_ratio,
                                            random_state=i)
            fold_name = f'fold_{i}'
            df_split[fold_name] = 0
            df_split.loc[train, fold_name] = 1
            df_split.loc[valid, fold_name] = 2
            df_split.loc[test_idx, fold_name] = 3

    return df_split


def get_tranform(mean=(0.389, 0.389, 0.389),
                 std=(0.240, 0.240, 0.240),
                 bbox=True):
    """Defines a customized data transformation including augmentation.

    Args:
        mean (tuple): The mean values of each channel.
        std (tuple): The standard deviations of each channel.
        bbox (bool): Whether for bbox dataset.

    Returns:
        Callable: customized transformation on images.

    """

    def bbox_tranform():
        tr_ct = A.Compose([
            A.RandomCrop(460, 460, p=0.2),
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ],
            bbox_params=A.BboxParams(format='albumentations',
                                     label_fields=['class_labels']))

        val_ct = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ],
            bbox_params=A.BboxParams(format='albumentations',
                                     label_fields=['class_labels']))

        return tr_ct, val_ct

    def augm_tranform():
        tr_ct = A.Compose([A.Rotate(25, p=0.2),
                           A.Resize(256, 256),
                           A.HorizontalFlip(p=0.5),
                           A.Normalize(mean=mean, std=std),
                           ToTensorV2()
                           ])

        val_ct = A.Compose([A.Resize(256, 256),
                            A.Normalize(mean=mean, std=std),
                            ToTensorV2()
                            ])

        return tr_ct, val_ct

    if bbox:
        return bbox_tranform
    else:
        return augm_tranform


def get_augm_data_loaders_in_file(df_path, im_path, split_df,
                                  train_batch_size=64, valid_batch_size=128,
                                  custom_tranform=None,
                                  datasetclass=AugmBBoxDatasetAll,
                                  fname_col='File_name',
                                  n_fold=0, augm=True, fix=False, **kwargs):
    """Gets dataloaders for training / inference on image-based datasets.

    Args:
        df_path (str): The file path of the csv file.
        im_path (str): The file path of the image folder.
        split_df (pd.DataFrame): The DataFrame for the train-valid-test splits.
        train_batch_size (int): The batch size used for training.
        valid_batch_size (int): The batch size used for validation and testing.
        custom_tranform (Callable): The customized data tranformation Callable.
        datasetclass (torch.utils.data.Dataset): The dataset of interests.
        fname_col (str): Column name for the image file location.
        n_fold (int): The index of the training and validation set, from 1 to 5.
        augm (bool): Whether include image augumentaion approaches.
        fix (bool): Whether do the shuffle during the training/inference.
        **kwargs (dict): Other kwargs for defining the Dataset object.

    Returns:
        (DataLoader, DataLoader, DataLoader, DataLoader): Dataloaders for
            training, training/training_evaluation, validation and testing.

    """
    tr_ct, val_ct = custom_tranform()

    if augm:
        dataset = datasetclass(df_path, im_path, transform=tr_ct,
                               fname_col=fname_col, **kwargs)
    else:
        dataset = datasetclass(df_path, im_path, transform=val_ct,
                               fname_col=fname_col, **kwargs)

    dataset_no_augm = datasetclass(df_path, im_path, transform=val_ct,
                                   fname_col=fname_col, **kwargs)

    train_idx = split_df[split_df[f'fold_{n_fold}'] == 1].index.to_list()
    valid_idx = split_df[split_df[f'fold_{n_fold}'] == 2].index.to_list()
    test_idx = split_df[split_df[f'fold_{n_fold}'] == 3].index.to_list()

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    logger.info(f'Size of the training: {len(train_idx)}.')
    logger.info(f'Size of the validation: {len(valid_idx)}.')
    logger.info(f'Size of the testing: {len(test_idx)}.')

    train_loader = DataLoader(dataset, batch_size=train_batch_size,
                              sampler=train_sampler, num_workers=4)
    valid_loader = DataLoader(dataset_no_augm, batch_size=valid_batch_size,
                              sampler=valid_sampler, num_workers=4)
    test_loader = DataLoader(dataset_no_augm, batch_size=valid_batch_size,
                             sampler=test_sampler, num_workers=4)

    if fix:
        subset = Subset(dataset_no_augm, indices=test_idx)
        test_loader = DataLoader(subset, batch_size=valid_batch_size,
                                 shuffle=False)
    if augm:
        # a seperated evaluater is necessary for augmented dataset
        train_evaluator_loader = DataLoader(dataset_no_augm,
                                            batch_size=valid_batch_size,
                                            sampler=train_sampler,
                                            num_workers=4)

        return train_loader, train_evaluator_loader, valid_loader, test_loader
    else:
        return train_loader, train_loader, valid_loader, test_loader


def prepare_batch(batch, device, non_blocking, new_shape=None):
    """Prepare the batch data for training/inference, move data to GPU, reshape
    the target if necessary.

    Args:
        batch (torch.Tensor): A batch of data.
        device (torch.device or str): Device to load the backbone and data.
        non_blocking (bool): Whether tries to convert asynchronously with
            respect to the host if possible.
            https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to
        new_shape (tuple): The new shape of the target variable, sometimes
            necessary for certain API calls.

    Returns:
        (torch.Tensor, torch.Tensor)

    """

    x = batch['image'].to(device, dtype=torch.float, non_blocking=non_blocking)

    y = batch['target'].to(device, dtype=torch.float, non_blocking=non_blocking)

    if new_shape:
        y = y.view(*new_shape)

    return x, y


def prepare_batch_cae(batch, device, non_blocking, key=None):
    """prepare batch for CAE training, the kwarg key is used for the
    compatibility with MNIST dataset"""

    if key:
        x = batch[key]
    else:
        x = batch[0]
    x = x.to(device, dtype=torch.float, non_blocking=non_blocking)

    return x, x


def bbox_plot_aug(image, x1n, y1n, x2n, y2n, des=None, idx=None, ax=None):
    """Visualize the image with bbox information.
    x1n, y1n, x2n, y2n follows
    https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#albumentations

    Args:
        image (np.array): The image to show.
        x1n (float): The bbox information for x1n.
        y1n (float): The bbox information for y1n.
        x2n (float): The bbox information for x2n
        y2n (float): The bbox information for y2n.
        des (str): An optional description of the image.
        idx (int): The index number of the figure.
        ax (matplotlib.axes.Axes): The axes for plotting the figure.

    Returns:
        None.

    """

    if not ax:
        fig, ax = plt.subplots(figsize=[16, 9])

    ax.imshow(image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # note, the reading by cv2 results in a format as following
    # if the reading is done by pillow, then height and width should be inverted
    height, width, _ = image.shape
    x = x1n * width
    y = y1n * height
    w = (x2n - x1n) * width
    h = (y2n - y1n) * height

    patch = ax.add_patch(patches.Rectangle((x, y), w, h,
                                           fill=False, edgecolor='red',
                                           linewidth=2))
    patch.set_path_effects([patheffects.Stroke(
        linewidth=3, foreground='white'), patheffects.Normal()])
    if des:
        txt = ax.text(x, y, des, verticalalignment='top', color='white',
                      fontsize=14, weight='bold')
        txt.set_path_effects([patheffects.Stroke(
            linewidth=1, foreground='white'), patheffects.Normal()])
    if idx is not None:
        ax.set_title('Sample #{}'.format(idx))


if __name__ == '__main__':
    pass
