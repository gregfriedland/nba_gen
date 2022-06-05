from math import ceil, log2

import torch.nn as nn
import torch.nn.functional as F

from torchgan.models import Discriminator, Generator


class PBPGenerator(Generator):
    def __init__(
        self,
        encoding_dims=100,
        internal_size=32,
        num_internal_layers=2,
        event_size=10,
        label_type="none",
    ):
        super().__init__(encoding_dims, label_type)
        use_bias = False
        nl = nn.LeakyReLU(0.2)
        last_nl = nn.Tanh()

        model = []
        model.append(
            nn.Sequential(
                nn.Linear(self.encoding_dims, internal_size),
                nn.BatchNorm1d(internal_size),
                nl,
            )
        )
        for i in range(num_internal_layers):
            model.append(
                nn.Sequential(
                    nn.Linear(internal_size, internal_size, bias=use_bias),
                    nn.BatchNorm1d(internal_size),
                    nl,
                )
            )
        model.append(
            nn.Sequential(
                nn.Linear(internal_size, event_size, bias=True), last_nl
            )
        )
        self.model = nn.Sequential(*model)
        self._weight_initializer()

    def forward(self, x, feature_matching=False):
        return self.model(x)


class PBPDiscriminator(Discriminator):
    def __init__(
        self,
        event_size=10,
        internal_size=32,
        num_internal_layers=2,
        label_type="none",
    ):
        super().__init__(event_size, label_type)
        use_bias = False
        nl = nn.LeakyReLU(0.2)

        model = [
            nn.Sequential(
                nn.Linear(event_size, internal_size, bias=True), nl
            )
        ]
        for i in range(num_internal_layers):
            model.append(
                nn.Sequential(
                    nn.Linear(internal_size, internal_size),
                    nn.BatchNorm1d(internal_size),
                    nl,
                )
            )
        model.append(nn.Sequential(
            nn.Linear(internal_size, 1, bias=use_bias), nl
        ))
        self.model = nn.Sequential(*model)
        self._weight_initializer()

    def forward(self, x, feature_matching=False):
        r"""Calculates the output tensor on passing the image ``x`` through the Discriminator.

        Args:
            x (torch.Tensor): A 4D torch tensor of the image.
            feature_matching (bool, optional): Returns the activation from a predefined intermediate
                layer.

        Returns:
            A 1D torch.Tensor of the probability of each image being real.
        """
        x = self.model(x)
        return x
        # if feature_matching is True:
        #     return x
        # else:
        #     x = self.disc(x)
        #     return x.view(x.size(0))