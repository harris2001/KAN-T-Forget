################################################################################
# Copyright (c) 2024 Harris Hadjiantonis.                                      #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-08-2024                                                              #
# Author(s): Harris Hadjiantonis                                               #
# E-mail: harrishadjiantonis@gmail.com                                         #
# Website: https://github.com/harris2001/                                      #
################################################################################

import torch.nn as nn

from avalanche.models.dynamic_modules import (
    MultiTaskModule,
    MultiHeadClassifier,
)
from avalanche.models.base_model import BaseModel


class optimised_KAN(nn.Module, BaseModel):
    """
    Performance optimised Kolmogorov-Arnold Neural Network (KAN) with custom parameters.

    **Example**::

        >>> from avalanche.models import OptimisedKAN
        >>> n_classes = 10 # e.g. MNIST
        >>> model = OptimisedKAN(num_classes=n_classes, normalize_layers=True)
        >>> print(model) # View model details
    """

    def __init__(
        self,
        num_classes=10,
        input_size=28 * 28,
        hidden_size=32,
        hidden_layers=1,
        grids=8,
        normalize_layers=True,
        device='cpu',
        
    ):
        """
        :param num_classes: output size
        :param input_size: input size
        :param hidden_size: hidden layer size
        :param hidden_layers: number of hidden layers
        :param drop_rate: dropout rate. 0 to disable
        """
        super().__init__()

        layers = nn.Sequential(
            *(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop_rate),
            )
        )
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}",
                nn.Sequential(
                    *(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=drop_rate),
                    )
                ),
            )

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self._input_size = input_size

    def forward(self, x, normalize_layers=True):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        return x



__all__ = ["OptimisedKAN"]
