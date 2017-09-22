#
# some useful function used in running procedure
#

from __future__ import absolute_import

__all__ = ['adjust_learning_rate', 'AverageMeter']


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    lr = lr * gamma ** (epoch / schedule)
    for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        self.avg = None
        self.count = None
        self.sum = None
        self.val = None

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
