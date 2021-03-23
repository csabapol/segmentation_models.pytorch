import torch.nn as nn

from . import base
from . import functional as F
from .base import Activation


class JaccardLoss(base.Loss):
    def __init__(self, eps=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class CategoricalFocalLoss(base.Loss):
    def __init__(
        self,
        eps=1e-7,
        alpha=0.25,
        gamma=2.0,
        activation=None,
        ignore_channels=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.categorical_focal_loss(
            y_pr,
            y_gt,
            eps=self.eps,
            alpha=self.alpha,
            gamma=self.gamma,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class WeightedDiceLoss(base.Loss):
    def __init__(
        self,
        class_weights,
        eps=1.0,
        beta=1.0,
        activation=None,
        ignore_channels=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.class_weights = class_weights

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.weighted_f_score(
            y_pr,
            y_gt,
            self.class_weights,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DiceLoss(base.Loss):
    def __init__(
        self, eps=1.0, beta=1.0, activation=None, ignore_channels=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass
