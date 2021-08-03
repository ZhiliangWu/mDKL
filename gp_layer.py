#
# mDKL
#
# Copyright (c) Siemens AG, 2021
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

import torch
import gpytorch

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import IndependentMultitaskVariationalStrategy
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel


class SVGPLayer(ApproximateGP):
    """The SVGP output layer with an RBF kernel."""
    def __init__(self, inducing_points):
        """

        Args:
            inducing_points (torch.Tensor): The initial inducing points.
        """
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points,
                                                   variational_distribution,
                                                   learn_inducing_locations=True
                                                   )
        super(SVGPLayer, self).__init__(variational_strategy)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class IMTSVGPLayer(ApproximateGP):
    """The independent multi-task SVGP output layer with RBF kernels."""
    def __init__(self, inducing_points, num_tasks=2):
        """

        Args:
            inducing_points (torch.Tensor): the initial inducing points.
            num_tasks (int): The number of tasks (outputs in our case).
        """
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
        )

        variational_strategy = IndependentMultitaskVariationalStrategy(
            VariationalStrategy(self, inducing_points, variational_distribution,
                                learn_inducing_locations=True),
            num_tasks=num_tasks, task_dim=-1)

        super(IMTSVGPLayer, self).__init__(variational_strategy)

        self.mean_module = ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


if __name__ == '__main__':
    pass
