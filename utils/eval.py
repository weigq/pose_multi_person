#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

__mtime__ = '17-9-22'

__all__ = ['get_accuracy']

import torch


def get_preds(scores):
    """
    :param scores: scoremap
    :return: coords of joints with type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    scores = scores.contiguous()
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:,:,0] = (preds[:,:,0] - 1) % scores.size(3) + 1
    preds[:,:,1] = torch.floor((preds[:,:,1] - 1) / scores.size(2)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def calc_dists(preds, target, normalize):
    """
    :param preds:
    :param target:
    :param normalize:
    :return:
    """
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n,c,0] > 1 and target[n, c, 1] > 1:
                dists[c, n] = torch.dist(preds[n,c,:], target[n,c,:])/normalize[n]
            else:
                dists[c, n] = -1
    return dists


def get_accuracy(output, target, idxs, thr=0.5):
    """
    :param output: scoremap
    :param target: groudtruth
    :param idxs: ids of joints to calaulate acc
    :param thr: threshold
    :return: accuray of scoremap
    """
    preds   = get_preds(output)
    gts     = get_preds(target)
    norm    = torch.ones(preds.size(0))*output.size(3)/10
    dists   = calc_dists(preds, gts, norm)

    acc = torch.zeros(len(idxs)+1)
    avg_acc = 0
    cnt = 0

    for i in range(len(idxs)):
        acc[i+1] = dist_acc(dists[idxs[i]-1])
        if acc[i+1] >= 0:
            avg_acc = avg_acc + acc[i+1]
            cnt += 1

    if cnt != 0:
        acc[0] = avg_acc / cnt
    return acc
