"""
Reference:  Iterative Normalization: Beyond Standardization towards Efficient Whitening, CVPR 2019

- Paper:
- Code: https://github.com/huangleiBuaa/IterNorm
"""
import torch.nn
from torch.nn import Parameter

__all__ = ['IterNormSigmaSingle', 'IterNormSigma']


class IterativeNormalization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        X, running_mean, running_wmat, nc, ctx.t, eps, momentum, training, by_instance = args
        if by_instance:
            # change NxCxHxW to (NxD)x(HxW), i.e., d*m
            x = X.view(X.size(0)*nc, -1)
        else:
            # change NxCxHxW to Dx(NxHxW), i.e., d*m
            x = X.transpose(0, 1).contiguous().view(nc, -1)
        d, m = x.size()
        # calculate centered activation by subtracted mini-batch mean
        mean = x.mean(-1, keepdim=True) if training else running_mean
        xc = x - mean
        if training:
            running_mean.copy_(momentum * mean + (1. - momentum) * running_mean)
            # calculate covariance matrix
            sigma = torch.addmm(eps, torch.eye(d).to(X), 1. / m, xc, xc.transpose(0, 1))
            running_wmat.copy_(momentum * sigma + (1. - momentum) * running_wmat)
        else:
            sigma = running_wmat
        # reciprocal of trace of Sigma: shape [g, 1, 1]
        p = [None] * (ctx.t + 1)
        p[0] = torch.eye(d).to(X)
        r_t_r = (sigma * p[0]).sum((0, 1), keepdim=True).reciprocal_()
        sigma_n = sigma * r_t_r
        for k in range(ctx.t):
            p[k+1] = torch.addmm(1.5, p[k], -0.5, torch.matrix_power(p[k], 3), sigma_n)

        wm = p[ctx.t].mul_(r_t_r.sqrt())  # whiten matrix: the matrix inverse of Sigma, i.e., Sigma^{-1/2}
        xn = wm.mm(xc)
        if by_instance:
            # change (NxD)x(HxW) to NxCxHxW
            Xn = xn.view(X.size(0), X.size(1), *X.size()[2:])
        else:
            Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
        return Xn


class IterNormSigmaSingle(torch.nn.Module):
    def __init__(self, num_features, t=5, dim=4, eps=1e-5, momentum=0.1, affine=True, by_instance=False,
                 *args, **kwargs):
        super(IterNormSigmaSingle, self).__init__()
        # assert dim == 4, 'IterNormSigma is not support 2D'
        self.t = t
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        self.by_instance = by_instance
        shape = [1] * dim
        shape[1] = self.num_features

        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        # running whiten matrix
        self.register_buffer('running_wm', torch.eye(num_features))

    def forward(self, inputs: torch.Tensor):
        inputs_hat = IterativeNormalization.apply(inputs,
                                                  self.running_mean,
                                                  self.running_wm,
                                                  self.num_features,
                                                  self.t,
                                                  self.eps,
                                                  self.momentum,
                                                  self.training,
                                                  self.by_instance)
        return inputs_hat


class IterNormSigma(torch.nn.Module):
    def __init__(self, num_features, num_channels=None,
                 t=5, dim=4, eps=1e-5, momentum=0.1,
                 affine=True, by_instance=False,
                 *args, **kwargs):
        super(IterNormSigma, self).__init__()
        # assert dim == 4, 'IterNormSigma is not support 2D'
        self.t = t
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.num_channels = num_channels
        self.num_groups = (self.num_features-1) // self.num_channels + 1
        self.iter_norm_groups = torch.nn.ModuleList(
            [IterNormSigmaSingle(num_features=self.num_channels, eps=eps, momentum=momentum, t=t) for _ in range(self.num_groups - 1)]
        )
        num_channels_last = self.num_features - self.num_channels * (self.num_groups-1)
        self.iter_norm_groups.append(IterNormSigmaSingle(num_features=num_channels_last,
                                                         eps=eps,
                                                         momentum=momentum,
                                                         t=t,
                                                         by_instance=by_instance))
         
        self.affine = affine
        self.dim = dim
        shape = [1] * dim
        shape[1] = self.num_features
        if self.affine:
            self.weight = Parameter(torch.Tensor(*shape))
            self.bias = Parameter(torch.Tensor(*shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor):
        inputs_splits = torch.split(inputs, self.num_channels, dim=1)
        inputs_hat_splits = []
        for i in range(self.num_groups):
            inputs_hat_tmp = self.iter_norm_groups[i](inputs_splits[i])
            inputs_hat_splits.append(inputs_hat_tmp)
        inputs_hat = torch.cat(inputs_hat_splits, dim=1)
        # affine
        if self.affine:
            return inputs_hat * self.weight + self.bias
        else:
            return inputs_hat

    def extra_repr(self):
        return '{num_features}, num_channels={num_channels}, T={T}, eps={eps}, ' \
               'momentum={momentum}, affine={affine}'.format(**self.__dict__)


if __name__ == '__main__':
    ItN = IterNormSigma(8, num_channels=4, T=10, momentum=1, affine=False)
    print(ItN)
    ItN.train()
    #x = torch.randn(32, 64, 14, 14)
    x = torch.randn(32, 8)
    x.requires_grad_()
    y = ItN(x)
    z = y.transpose(0, 1).contiguous().view(x.size(1), -1)
    print(z.matmul(z.t()) / z.size(1))

    y.sum().backward()
    print('x grad', x.grad.size())

    ItN.eval()
    y = ItN(x)
    z = y.transpose(0, 1).contiguous().view(x.size(1), -1)
    print(z.matmul(z.t()) / z.size(1))
