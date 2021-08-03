#
# mDKL
#
# Copyright (c) Siemens AG, 2021
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from yellowbrick.features import JointPlotVisualizer
from yellowbrick.regressor import ResidualsPlot, PredictionError

from sklearn.metrics import mean_squared_error, r2_score, \
    mean_absolute_error,  median_absolute_error

from logging_conf import logger
from model_utils import Mock


def joint_plot_xy(x_t, y_t, x_te, y_te, dp, n_epoch, label):
    """A joint plot for predicted location (x, y).

    Args:
        x_t (np.ndarray): The values of the predicted x values in the
            training set.
        y_t (np.ndarray): The values of the predicted y values in the
            training set.
        x_te (np.ndarray): The values of the predicted x values in the
            validatoin/test set.
        y_te (np.ndarray): The values of the predicted y values in the
            validation/test set.
        dp (str): Data path of the generated plot.
        n_epoch (int): The number of training epoch.
        label (str): The name of the plot in a format of pred_[valid/test].

    Returns:
        None: The joint-plot is saved to the given data path.

    """
    fig, ax = plt.subplots()
    viz = JointPlotVisualizer(ax)
    viz.fit_transform(X=x_t, y=y_t)
    viz.fit_transform(X=x_te, y=y_te)
    viz.finalize()
    save_fp = f'./{dp}/epoch_{n_epoch}_{label}_joint.pdf'
    fig.savefig(save_fp, dpi=600)
    plt.close(fig)


def residual_plot(y_t_true, y_t_p, y_te_true, y_te_p, dp, n_epoch, label):
    """Generate the residual plot for the predictions in training and testing.

    Args:
        y_t_true (np.ndarray): True values of the targets in the training set.
        y_t_p (np.ndarray): Predicted values of the targets in the training
            set.
        y_te_true (np.ndarray): True values of the targets in the
            validation/teset set.
        y_te_p (np.ndarray): Predicted values of the targets in the
            validation/teset set.
        dp (str): Data path of the generated plot.
        n_epoch (int): The number of training epoch.
        label (str): The name of the plot in a format of pred_[valid/test].

    Returns:
         None: The residual plot is saved to the given data path.

    """

    mock = Mock(y_t_p, y_te_p)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[20, 8])

    visualizer1 = ResidualsPlot(mock, ax1, is_fitted=True)

    visualizer1.score(True, y_t_true, train=True)
    visualizer1.score(False, y_te_true, train=False)
    visualizer1.finalize()

    visualizer2 = PredictionError(mock, ax2, is_fitted=True)

    visualizer2.score(True, y_t_true)
    visualizer2.score(False, y_te_true)
    visualizer2.finalize()

    save_fp = f'./{dp}/epoch_{n_epoch}_residual_{label}.pdf'
    fig.savefig(save_fp, dpi=600)
    plt.close(fig)


def prepare_pq_plot_equal(y_true, y_pred, y_std, n_quantiles=10):
    """Prepare metrics table for QP plot, each chunk have the same number of
    samples.

    Args:
        y_true (np.ndarray): True labels of the shape (n_samples, [n_tasks]).
        y_pred (np.ndarray): Prediction of the labels. 
        y_std (np.ndarray): Stadard deviation of the predictions.
        n_quantiles (int): How many chunks we want to have.

    Returns:
        pd.DataFrame: Metrics table with corresponding quantile split.
    """

    if y_std.ndim > 1:
        y_std_avg = np.mean(y_std, axis=1)

        # argsort return indices from small to large
        y_true_sorted = y_true[np.argsort(y_std_avg), :]
        y_pred_sorted = y_pred[np.argsort(y_std_avg), :]

    else:
        y_true_sorted = y_true[np.argsort(y_std)]
        y_pred_sorted = y_pred[np.argsort(y_std)]

    n_samples = y_true.shape[0]
    n_samples_per_quantile = n_samples // n_quantiles

    logger.info('Equal mode')
    logger.info(f'In each chunk, there are {n_samples_per_quantile} samples.')

    y_true_list = [y_true_sorted[i: i+n_samples_per_quantile] for i in range(
        0, n_samples, n_samples_per_quantile)]
    y_pred_list = [y_pred_sorted[i: i+n_samples_per_quantile] for i in range(
        0, n_samples, n_samples_per_quantile)]

    quantile_list = []
    rmse_list = []
    mae_list = []
    r2score_list = []
    mad_list = []  # median absolute error

    for i, (y_t, y_p) in enumerate(zip(y_true_list[:n_quantiles],
                                       y_pred_list[:n_quantiles])):

        rmse_quantile = mean_squared_error(y_t, y_p,
                                           multioutput='uniform_average',
                                           squared=False)
        mae_quantile = mean_absolute_error(y_t, y_p,
                                           multioutput='uniform_average')
        r2s_quantile = r2_score(y_t, y_p, multioutput='uniform_average')
        mad_quantile = median_absolute_error(y_t, y_p,
                                             multioutput='uniform_average')
        quantile_list.append((i + 1) / n_quantiles)
        rmse_list.append(rmse_quantile)
        mae_list.append(mae_quantile)
        r2score_list.append(r2s_quantile)
        mad_list.append(mad_quantile)

    array_pq = np.stack([quantile_list, rmse_list, mae_list,
                         r2score_list, mad_list],
                        axis=1)
    df_pq = pd.DataFrame(data=array_pq,
                         columns=['quantile', 'rmse', 'mae',  'r2s', 'mad']
                         )

    return df_pq


def prepare_pq_plot_accu(y_true, y_pred, y_std, n_quantiles=10):
    """Prepare metrics table for qp plot, the samples are accumulated.

    Args:
        y_true (np.ndarray): True labels of shape (n_samples, [n_tasks]).
        y_pred (np.ndarray): Prediction of the labels.
        y_std (np.ndarray): Stadard deviation of the predictions.
        n_quantiles (int): How many chunks we want to have.

    Returns:
        pd.DataFrame: Metrics table with accumulated quantile splits.
    """

    if y_std.ndim > 1:
        y_std_avg = np.mean(y_std, axis=1)

        y_true_sorted = y_true[np.argsort(y_std_avg), :]
        y_pred_sorted = y_pred[np.argsort(y_std_avg), :]

    else:
        y_true_sorted = y_true[np.argsort(y_std)]
        y_pred_sorted = y_pred[np.argsort(y_std)]

    n_samples = y_true.shape[0]
    n_samples_per_quantile = n_samples // n_quantiles

    y_true_list = [y_true_sorted[: i+n_samples_per_quantile] for i in range(
        0, n_samples, n_samples_per_quantile)]
    y_pred_list = [y_pred_sorted[: i+n_samples_per_quantile] for i in range(
        0, n_samples, n_samples_per_quantile)]

    if n_quantiles < len(y_true_list):
        # to include all samples in the last chunk
        y_true_list[-2] = y_true_sorted[:]
        y_pred_list[-2] = y_pred_sorted[:]
        logger.info('The extra chunk is merged into the last chunk!')

    quantile_list = []
    rmse_list = []
    mae_list = []
    r2score_list = []
    mad_list = []  # median absolute error

    logger.info('Accumulation mode...')

    for i, (y_t, y_p) in enumerate(zip(y_true_list[:n_quantiles],
                                       y_pred_list[:n_quantiles])):
        logger.info(f'In chunk {i}, there are {y_t.shape[0]} samples.')

        rmse_quantile = mean_squared_error(y_t, y_p,
                                           multioutput='uniform_average',
                                           squared=False)
        mae_quantile = mean_absolute_error(y_t, y_p,
                                           multioutput='uniform_average')
        r2s_quantile = r2_score(y_t, y_p, multioutput='uniform_average')
        mad_quantile = median_absolute_error(y_t, y_p,
                                             multioutput='uniform_average')
        quantile_list.append((i + 1) / n_quantiles)
        rmse_list.append(rmse_quantile)
        mae_list.append(mae_quantile)
        r2score_list.append(r2s_quantile)
        mad_list.append(mad_quantile)

    array_pq = np.stack([quantile_list, rmse_list, mae_list,
                         r2score_list, mad_list],
                        axis=1)
    df_pq = pd.DataFrame(data=array_pq,
                         columns=['quantile', 'rmse', 'mae', 'r2s', 'mad'])

    return df_pq


def plot_pq(df_pq, df_pq_std=None, columns=('mae', 'r2s'),
            title='Performance-Quantile'):
    """Plot the quantile performance plot from the prepared metrics table.

    Args:
        df_pq (pd.DataFrame): The QP table information with mean values.
        df_pq_std (pd.DataFrame): The QP table information with std values.
        columns (tuple): Which column of the qp table to be plotted, limited
            to 2 items.
        title (str): An optional name of the figure.

    Returns:
        plt.Figure: A figure of the resulting QP plot.

    """

    fig, ax1 = plt.subplots(figsize=(16, 9))

    if len(columns) == 1:
        ax1.plot(df_pq['quantile'], df_pq[columns[0]], 'r', label=columns[0])
        ax1.set_ylabel(columns[0].upper())
        ax1.legend(loc=1)

        if df_pq_std is not None:
            ax1.fill_between(df_pq['quantile'],
                             df_pq[columns[0]] - df_pq_std[columns[0]],
                             df_pq[columns[0]] + df_pq_std[columns[0]],
                             color='r',
                             alpha=0.5
                             )

    elif len(columns) == 2:
        _ = ax1.plot(df_pq['quantile'], df_pq[columns[0]], 'r',
                     label=columns[0])
        ax1.set_ylabel(columns[0].upper())
        ax2 = ax1.twinx()
        _ = ax2.plot(df_pq['quantile'], df_pq[columns[1]], 'g',
                     label=columns[1])
        ax2.set_ylabel(columns[1].upper())
        ax1.legend(loc=1)
        ax2.legend(loc=4)

        if df_pq_std is not None:
            ax1.fill_between(df_pq['quantile'],
                             df_pq[columns[0]] - df_pq_std[columns[0]],
                             df_pq[columns[0]] + df_pq_std[columns[0]],
                             color='r',
                             alpha=0.5
                             )

            ax2.fill_between(df_pq['quantile'],
                             df_pq[columns[1]] - df_pq_std[columns[1]],
                             df_pq[columns[1]] + df_pq_std[columns[1]],
                             color='g',
                             alpha=0.5
                             )

    else:
        raise ValueError('Too many columns. Currently only two are allowed.')

    ax1.set_xlabel('Quantile')
    ax1.set_title(title)
    plt.show()

    return fig


def plot_pq_all(mean_arr, std_arr, label_list, title='QP_accu',
                metric='rmse', ax=None, colors=None, alpha=0.5):
    """Generate a summary QP plots with different methods.

    Args:
        mean_arr (np.ndarray): Mean performance of each method at a certain
            quantile, (#methods, #quantiles).
        std_arr  (np.ndarray): Standard deviations of each method at a certain
            quantile, (#methods, #quantiles).
        label_list (list): The name list of each methods, used for legends.
        title (str): An optional name of the figure.
        metric (str): The performance metric, used for the y-axis labe.
        ax (matplotlib.axes.Axes): An optional existing ax to plot the figure.
        colors (list): The color list for each methods.
        alpha (float): The value of opacity.

    Returns:
        plt.Figure: A figure of the summary QP plot.

    """
    if ax:
        fig = None
    else:
        fig, ax = plt.subplots(figsize=(16, 9))

    n_quantile = mean_arr.shape[1]
    x_values = (np.arange(n_quantile) + 1) / n_quantile

    for i, l in enumerate(label_list):
        if colors:
            ax.plot(x_values, mean_arr[i, :], label=l, linewidth=0.8,
                    color=colors[i])
            ax.fill_between(x_values,
                            mean_arr[i, :] - std_arr[i, :],
                            mean_arr[i, :] + std_arr[i, :],
                            alpha=alpha, color=colors[i])
        else:
            ax.plot(x_values, mean_arr[i, :], label=l, linewidth=0.8)

            ax.fill_between(x_values,
                            mean_arr[i, :] - std_arr[i, :],
                            mean_arr[i, :] + std_arr[i, :],
                            alpha=alpha)

    ax.legend(loc=4, frameon=True, borderaxespad=0.1)
    ax.set_xlabel('Quantile of the predictive variance')
    ax.set_ylabel(metric.upper())
    ax.grid(True)
    if title:
        ax.set_title(title)
    plt.show()

    return fig
