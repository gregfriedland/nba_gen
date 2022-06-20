from math import ceil, log2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgan.losses import GeneratorLoss, DiscriminatorLoss

from torchgan.models import Discriminator, Generator


class PBPGenerator(Generator):
    def __init__(
        self,
        encoding_dims=100,
        internal_size=32,
        event_size=10,
    ):
        super().__init__(encoding_dims, "none")

        self.model = nn.Sequential(
            nn.Linear(self.encoding_dims, internal_size),
            nn.ReLU(),
            nn.Linear(internal_size, event_size),
        )
        self._weight_initializer()

    def forward(self, x):
        return self.model(x)


class PBPDiscriminator(Discriminator):
    def __init__(
        self,
        event_size=10,
        internal_size=32,
    ):
        super().__init__(event_size, "none")

        self.model = nn.Sequential(
            nn.Linear(event_size, internal_size),
            nn.ReLU(),
            nn.Linear(internal_size, 1),
            # nn.Sigmoid()  # use binary_cross_entropy_with_logits instead
        )
        self._weight_initializer()

    def forward(self, x):
        return self.model(x)


class PBPGeneratorLoss(GeneratorLoss):
    def __init__(self, reduction="mean"):
        super().__init__(reduction)

    def forward(self, dgz):
        target = torch.ones_like(dgz)
        return F.binary_cross_entropy_with_logits(dgz, target, reduction=self.reduction)


class PBPDiscriminatorLoss(DiscriminatorLoss):
    def __init__(self, reduction="mean"):
        super().__init__(reduction)

    def forward(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels, reduction=self.reduction)
