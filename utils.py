import numpy as np
import scipy
import time
import torch
import torch.nn.functional as F
from texttable import Texttable


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def args_print(args):
    _dict = vars(args)
    t = Texttable()
    t.add_row(["Parameter", "Value"])
    for k in _dict:
        t.add_row([k, _dict[k]])
    print(t.draw())


def gen_features(rows, cols, v, m, n, y, p=False):
    """
    rows: adj row index
    cols: adj col index
    v:    adj value
    m, n: adj shape(m ,n)
    y: feature
    return: adj@y  ndarray
    """
    s = time.time()
    print("start:--------------------")

    x = scipy.sparse.coo_matrix((v, (rows, cols)), (m, n))
    print('x.shape', x.shape)
    if p: print(x.toarray())
    norm = x.sum(axis=1)
    print('norm.shape', norm.shape)
    if p: print(norm)
    x_norm = x.multiply(1 / (norm + 0.00001))
    print('x_norm.shape', x_norm.shape)
    if p: print(x_norm.toarray())
    out = x_norm.dot(y)
    print('out.shape', out.shape)

    print(f"time: {time.time() - s:.4f}s---------" )
    return out


class MsgNorm(torch.nn.Module):
    def __init__(self, learn_msg_scale=False):
        super(MsgNorm, self).__init__()

        self.msg_scale = torch.nn.Parameter(torch.Tensor([1.0]),
                                            requires_grad=learn_msg_scale)
        self.reset_parameters()

    def forward(self, x, msg, p=2):
        msg = F.normalize(msg, p=p, dim=1)
        x_norm = x.norm(p=p, dim=1, keepdim=True)
        msg = msg * x_norm * self.msg_scale
        return msg

    def reset_parameters(self):
        torch.nn.init.ones_(self.msg_scale)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.delta:
            self.counter += 1
            print('EarlyStopping counter: {0} out of {1}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

