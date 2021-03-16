import torch
from recurrence_matrix import recurrence_matrix


class RecurrenceFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return recurrence_matrix(x)


class RecurrenceLayer(torch.nn.Module):
    _MAX_DIV = 3

    @classmethod
    def _min_max_scaling(cls, x):
        shape = x.shape
        x = x.view(x.size(0), -1)
        x -= x.min(1, keepdim=True)[0]
        x /= x.max(1, keepdim=True)[0]
        x = x.view(*shape)
        return x

    @classmethod
    def forward(cls, x):
        d = RecurrenceFunction.apply(x)
        # d[d > cls._MAX_DIV] = cls._MAX_DIV
        # d = cls._min_max_scaling(d)
        return d
