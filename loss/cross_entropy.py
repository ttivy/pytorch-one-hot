"""
Cross Entropy Loss implementation for one-hot target.

Refs:
    https://pytorch.org/docs/stable/nn.html#crossentropyloss
"""

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss as _CrossEntropyLoss


def cross_entropy(input, target, weight=None, dim=1, ignore_index=None, reduction='mean'):
    """Cross Entropy Loss for one-hot target.

    Args:
        input (Tensor): input
        target (Tensor): one-hot target
        weight (Tensor, optional): weight
        dim (int, optional): dim
    """
    assert isinstance(input, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert isinstance(weight, torch.Tensor) or weight is None

    loss = F.log_softmax(input, dim).mul_(target)

    if weight is not None:
        loss.mul_(weight).div_(weight.mean())

    loss = loss.sum(dim=1).neg_()

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()

    raise ValueError('{} is not a valid value for reduction'.format(reduction))


class CrossEntropyLoss(_CrossEntropyLoss):
    """Cross Entropy Loss for one-hot target.
    Attributes:
        dim (int): dim
    """

    def __init__(self, weight=None, dim=1, size_average=None,
                 ignore_index=-100, reduce=None, reduction='mean'):
        """
        Args:
            weight (Tensor, optional): weight
            dim (int, optional): dim
        """
        super().__init__(weight, size_average, ignore_index, reduce, reduction)
        self.dim = dim

    def forward(self, input, target):
        """
        Args:
            input (Tensor): input
            target (Tensor): one-hot target
        """
        return cross_entropy(input, target, dim=self.dim, weight=self.weight,
                             ignore_index=self.ignore_index, reduction=self.reduction)


if __name__ == '__main__':
    w = None#torch.randn(3)
    x = torch.randn(3, 3)
    y = torch.FloatTensor([[0, 0, 0]]*3)
    y[0, 0] = y[1, 1] = y[2, 2] = 1
    _y = torch.LongTensor([0, 1, 2])

    # Both API retuns same value
    #print(x, _y, w)
    print('diff:', (cross_entropy(x, y, w) - F.cross_entropy(x, _y, w)).item())
    assert (cross_entropy(x, y, w) - F.cross_entropy(x, _y, w)).abs() < 1e-6
    assert (CrossEntropyLoss(w)(x, y) - _CrossEntropyLoss(w)(x, _y)).abs() < 1e-6
    print('asserted')
