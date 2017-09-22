import torch

import numpy as np
import math


def tag_loss(tagmap, trans_pts, npeople):
    ''' calculate the tag_loss of tags scoremap '''
    sigmma = 0.1
    res = tagmap.size(-1)
    loss = 0
    # trans_pts = trans_pts.cpu().numpy()
    pts = []

    for b in range(trans_pts.size(0)):
        num_pp = torch.zeros(npeople[b])  # store the h_mean of each people in each batch image
        loss1 = 0
        loss2 = 0
        num_pts_all = []
        # each people in batch
        for n in range(num_pp.size(0)):
            num_pts = []  # store the index of tagmap which caontains detected points
            # each joint/16
            for j in range(16):
                joint = trans_pts[b][n][j]
                if int(joint[0]) < 0 or int(joint[1]) > res - 1 or int(joint[1]) < 0 or int(joint[0]) > res - 1:
                    continue
                else:
                    num_pts.append(j)
                    num_pp[n] += tagmap.data[b][j][int(joint[1])][int(joint[0])]
            num_pp[n] = num_pp[n] * 1.0 / len(num_pts)
            num_pts_all.append(num_pts)
        for n in range(num_pp.size(0)):
            for j in range(len(num_pts_all[n])):
                inj = num_pts_all[n][j]
                loss1 += (tagmap.data[b][inj][int(trans_pts[b][n][inj][1])][int(trans_pts[b][n][inj][0])] - num_pp[n]) ** 2
            for n_ in range(num_pp.size(0)):
                loss2 += math.exp(-1.0 * (num_pp[n] - num_pp[n_]) ** 2 / 2 / sigmma ** 2)
        loss1 /= trans_pts.size(0)
        loss2 /= trans_pts.size(0) ** 2
    loss = (loss1 + loss2) / trans_pts.size(0)

    return loss
