import os
import numpy as np
import scipy
import scipy.spatial

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import random

import nets as models
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def set_train(multi_models):
    for v in range(len(multi_models)):
        multi_models[v].train()

def set_eval(multi_models):
    for v in range(len(multi_models)):
        multi_models[v].eval()

def calculate_coup(opt, matrix1, matrix2, prob1, prob2, targets, return_mat=False):
    target_arange = torch.arange(len(targets[0])).cuda(opt.device)
    matrix1 = matrix1.unsqueeze(0).repeat(len(target_arange), 1, 1)
    matrix2 = matrix2.unsqueeze(0).repeat(len(target_arange), 1, 1)
    matrix = matrix1[target_arange, :, targets[0]].unsqueeze(2).bmm(matrix2[target_arange, :, targets[0]].unsqueeze(1))
    matrix_diag = (matrix*(torch.eye(opt.class_num, device=opt.device).unsqueeze(0).repeat(len(target_arange), 1, 1))).sum(1)
    matrix_norm = (matrix1.bmm(matrix2.transpose(1, 2)) * (torch.eye(opt.class_num, device=opt.device).unsqueeze(0).repeat(len(target_arange), 1, 1))).sum(1)
    prob = torch.pow(prob1 * prob2, opt.gamma_1)
    prob_norm = prob.sum(1)
    s_coup = (prob / prob_norm.unsqueeze(1)) * ((matrix_diag / matrix_norm))
    if return_mat:
        s_coup_mat = [torch.diag(s_coup[i]).unsqueeze(0) for i in range(len(s_coup))]
        s_coup_mat = torch.cat(tuple(s_coup_mat), dim=0)
        return s_coup.sum(1), s_coup_mat
    return s_coup.sum(1)

def calculate_decoup(opt, matrix1, matrix2, prob1, prob2, targets, return_mat=False):
    target_arange = torch.arange(len(targets[0])).cuda(opt.device)
    matrix1 = matrix1.unsqueeze(0).repeat(len(target_arange), 1, 1)
    matrix2 = matrix2.unsqueeze(0).repeat(len(target_arange), 1, 1)
    matrix = matrix1[target_arange, :, targets[0]].unsqueeze(2).bmm(matrix2[target_arange, :, targets[0]].unsqueeze(1))
    matrix_norm = matrix1.bmm(matrix2.transpose(1, 2))
    prob = torch.pow((prob1.unsqueeze(2)).bmm(prob2.unsqueeze(1)), opt.gamma_2)
    prob_norm = 1 - torch.pow((prob1 * prob2), opt.gamma_2).sum(1)
    s_decoup = (prob / prob_norm.unsqueeze(1).unsqueeze(1)) * (matrix / matrix_norm)
    if return_mat:
        s_decoup_mat = s_decoup - (s_decoup*(torch.eye(opt.class_num, device=opt.device).unsqueeze(0).repeat(len(target_arange), 1, 1)))
        s_decoup = s_decoup.sum(1).sum(1) - (s_decoup*(torch.eye(opt.class_num, device=opt.device).unsqueeze(0).repeat(len(target_arange), 1, 1))).sum(1).sum(1)
        return s_decoup, s_decoup_mat
    s_decoup = s_decoup.sum(1).sum(1) - (s_decoup*(torch.eye(opt.class_num, device=opt.device).unsqueeze(0).repeat(len(target_arange), 1, 1))).sum(1).sum(1)
    return s_decoup

def cross_modal_contrastive_criterion(fea, lab, tau=1., reduction='mean'):
    img_fea = fea[0]
    txt_fea = fea[1].t()
    sim = img_fea.mm(txt_fea)
    sim_t = sim.t()

    sim = (sim / tau).exp()
    diag1 = sim.diag()
    loss1 = diag1 / sim.sum(1)

    sim_t = (sim_t / tau).exp()
    diag2 = sim_t.diag()
    loss2 = diag2 / sim_t.sum(1)
    if reduction == 'mean':
        return loss1.log().mean() + loss2.log().mean()
    return loss1.log() + loss2.log()

def RDH_loss(fea, neg):
    img_fea = fea[0]
    txt_fea = fea[1].t()
    scores = img_fea.mm(txt_fea)
    margin = 0.2
    diagonal = scores.diag().view(scores.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    cost_s = (margin + scores - d1).clamp(min=0)
    cost_im = (margin + scores - d2).clamp(min=0)

    # clear diagonals
    mask = torch.eye(scores.size(0)) > .5
    mask = mask.to(cost_s.device)
    cost_s, cost_im = cost_s.masked_fill_(mask, 0), cost_im.masked_fill_(mask, 0)

    top_neg_row = torch.topk(cost_s, k=neg, dim=1).values
    top_neg_column = torch.topk(cost_im.t(), k=neg, dim=1).values
    return -1 * ((top_neg_row.sum(dim=1) + top_neg_column.sum(dim=1)) / neg)  # (K,1)

def i2t(npts, sims, per_captions=1, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (per_captions * N, max_n_word, d) matrix of captions
    CapLens: (per_captions * N) array of caption lengths
    sims: (N, per_captions * N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    top5 = np.zeros((npts, 5), dtype=int)
    retreivaled_index = []
    for index in range(npts):
        inds = np.argsort(sims[index])
        retreivaled_index.append(inds)
        # Score
        rank = 1e20
        for i in range(per_captions * index, per_captions * index + per_captions, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]
        top5[index] = inds[0:5]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5, retreivaled_index)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(npts, sims, per_captions=1, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (per_captions * N, max_n_word, d) matrix of captions
    CapLens: (per_captions * N) array of caption lengths
    sims: (N, per_captions * N) matrix of similarity im-cap
    """
    ranks = np.zeros(per_captions * npts)
    top1 = np.zeros(per_captions * npts)
    top5 = np.zeros((per_captions * npts, 5), dtype=int)

    # --> (per_captions * N(caption), N(image))
    sims = sims.T
    retreivaled_index = []
    for index in range(npts):
        for i in range(per_captions):
            inds = np.argsort(sims[per_captions * index + i])
            retreivaled_index.append(inds)
            ranks[per_captions * index + i] = np.where(inds == index)[0][0]
            top1[per_captions * index + i] = inds[0]
            top5[per_captions * index + i] = inds[0:5]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5, retreivaled_index)
    else:
        return (r1, r5, r10, medr, meanr)

def fx_calc_map_label(train, train_labels, test, test_label, k=0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(test, train, metric)

    ord = dist.argsort(1)

    numcases = train_labels.shape[0]
    if k == 0:
        k = numcases
    if k == -1:
        ks = [50, numcases]
    else:
        ks = [k]

    def calMAP(_k):
        _res = []
        for i in range(len(test_label)):
            order = ord[i]
            p = 0.0
            r = 0.0
            for j in range(_k):
                if test_label[i] == train_labels[order[j]]:
                    r += 1
                    p += (r / (j + 1))
            if r > 0:
                _res += [p / r]
            else:
                _res += [0]
        return np.mean(_res)

    res = []
    for k in ks:
        res.append(calMAP(k))
    return res

class AverageMeter(object):
    def __init__(self):
        self.reset()

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

def rank(opts, img_embeds, rec_embeds, names):
    st = random.getstate()
    random.seed(opts.seed)
    idxs = np.argsort(names)
    names = names[idxs]
    idxs = range(opts.medr)

    glob_rank = []
    glob_recall = {1:0.0,5:0.0,10:0.0}
    for i in range(10):
        ids = random.sample(range(0,len(names)), opts.medr)
        img_sub = img_embeds[ids,:]
        rec_sub = rec_embeds[ids,:]

        if opts.embtype == 'image':
            sims = np.dot(img_sub,rec_sub.T)# im2recipe
        else:
            sims = np.dot(rec_sub,img_sub.T)# recipe2im

        med_rank = []
        recall = {1:0.0,5:0.0,10:0.0}
        for ii in idxs:
            # sort indices in descending order
            sorting = np.argsort(sims[ii,:])[::-1].tolist()

            # find where the index of the pair sample ended up in the sorting
            pos = sorting.index(ii)

            if (pos+1) == 1:
                recall[1] += 1
            if (pos+1) <= 5:
                recall[5] += 1
            if (pos+1) <= 10:
                recall[10] += 1

            med_rank.append(pos+1)

        for i in recall.keys():
            recall[i] = recall[i]/opts.medr

        med = np.median(med_rank)
        for i in recall.keys():
            glob_recall[i] += recall[i]
        glob_rank.append(med)

    for i in glob_recall.keys():
        glob_recall[i] = glob_recall[i]/10
    random.setstate(st)

    return np.average(glob_rank), glob_recall

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(
        self,
        scores,
        hard_negative=True,
        labels=None,
        soft_margin="linear",
        mode="train",
    ):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        if labels is None:
            margin = self.margin
        elif soft_margin == "linear":
            margin = self.margin * labels
        elif soft_margin == "exponential":
            s = (torch.pow(10, labels) - 1) / 9
            margin = self.margin * s
        elif soft_margin == "sin":
            s = torch.sin(math.pi * labels - math.pi / 2) / 2 + 1 / 2
            margin = self.margin * s

        # compare every diagonal score to scores in its column: caption retrieval
        cost_s = (margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row: image retrieval
        cost_im = (margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > 0.5
        mask = mask.to(cost_s.device)
        cost_s, cost_im = cost_s.masked_fill_(mask, 0), cost_im.masked_fill_(mask, 0)

        # maximum and mean
        cost_s_max, cost_im_max = cost_s.max(1)[0], cost_im.max(0)[0]
        cost_s_mean, cost_im_mean = cost_s.mean(1), cost_im.mean(0)

        if mode == "predict":
            p = margin - (cost_s_mean + cost_im_mean) / 2
            p = p.clamp(min=0, max=margin)
            idx = torch.argsort(p)
            ratio = scores.size(0) // 10 + 1
            p = p / torch.mean(p[idx[-ratio:]])
            return p

        elif mode == "warmup":
            return cost_s_mean.sum() + cost_im_mean.sum()
        elif mode == "train":
            if hard_negative:
                return cost_s_max + cost_im_max
            else:
                return cost_s_mean + cost_im_mean

        elif mode == "eval_loss":
            return cost_s_mean + cost_im_mean