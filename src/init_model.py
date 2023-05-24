import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

import nets as models

def init_model(args, train_dataset, mode='train'):
    print('===> Building Models & Optimizers ..')
    multi_models = []
    n_view = len(train_dataset.train_data)
    for v in range(n_view):
        if v == args.views.index('Img'):
            multi_models.append(models.__dict__['ImageNet'](input_dim=train_dataset.train_data[v].shape[1], output_dim=args.output_dim).to(args.device))
        elif v == args.views.index('Txt'):
            multi_models.append(models.__dict__['TextNet'](input_dim=train_dataset.train_data[v].shape[1], output_dim=args.output_dim).to(args.device))
        else:
            multi_models.append(models.__dict__['ImageNet'](input_dim=train_dataset.train_data[v].shape[1], output_dim=args.output_dim).to(args.device))
    
    C1 = nn.Linear(args.output_dim, args.class_num).to(args.device)
    C2 = nn.Linear(args.output_dim, args.class_num).to(args.device)

    parameters1 = list(C1.parameters())
    parameters2 = list(C2.parameters())
    parameters1 += list(multi_models[0].parameters())
    parameters2 += list(multi_models[1].parameters())

    if mode == 'warmup':
        lr = args.warmup_lr
        ms = args.warmup_ms
        wd1 = args.warmup_wd1
        wd2 = args.warmup_wd2
        gamma = args.warmup_gamma
    else:
        lr = args.train_lr
        ms = args.train_ms
        wd1 = wd2 = args.train_wd
        gamma = args.train_gamma

    if args.optimizer == 'SGD':
        optimizer1 = torch.optim.SGD(parameters1, lr=lr, momentum=0.9, weight_decay=wd1)
        optimizer2 = torch.optim.SGD(parameters2, lr=lr, momentum=0.9, weight_decay=wd2)
    elif args.optimizer == 'Adam':
        optimizer1 = optim.Adam(parameters1, lr=lr, betas=[0.5, 0.999], weight_decay=wd1)
        optimizer2 = optim.Adam(parameters2, lr=lr, betas=[0.5, 0.999], weight_decay=wd2)

    lr_schedu1 = optim.lr_scheduler.MultiStepLR(optimizer1, ms, gamma=gamma)
    lr_schedu2 = optim.lr_scheduler.MultiStepLR(optimizer2, ms, gamma=gamma)

    return multi_models, [C1, C2], [optimizer1, optimizer2], [lr_schedu1, lr_schedu2]

def init_model_recipe(args, mode='train', model=None, preds=None):
    if mode == 'warmup':
        lr1 = args.warmup_lr1
        lr2 = args.warmup_lr2
        ms = args.warmup_ms
        wd1 = args.warmup_wd1
        wd2 = args.warmup_wd2
        gamma = args.warmup_gamma
    else:
        lr1 = lr2 = args.train_lr
        ms = args.train_ms
        wd1 = wd2 = args.train_wd
        gamma = args.train_gamma

    print('===> Building Models & Optimizers ..')
    if model == None:
        multi_models = []
        for v in range(2):
            if v == args.views.index('Img'):
                multi_models.append(models.FoodSpaceImageEncoder(args).to(args.device))
            elif v == args.views.index('Txt'):
                multi_models.append(models.FoodSpaceTextEncoder(args).to(args.device))
    else:
        multi_models = model

    if preds == None:
        # C1 = nn.Linear(args.embDim, args.class_num).to(args.device)
        # C2 = nn.Linear(args.embDim, args.class_num).to(args.device)
        C1 = nn.Sequential(
            nn.Linear(args.embDim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, args.class_num)
        ).to(args.device)
        C2 = nn.Sequential(
            nn.Linear(args.embDim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, args.class_num)
        ).to(args.device)
    else:
        C1, C2 = preds

    C1_params = list(C1.parameters())
    C2_params = list(C2.parameters())

    heads_params1 = [kv[1] for kv in multi_models[0].named_parameters() if (kv[0].split('.')[0] not in ['visionMLP', 'textMLP'])]
    vision_params = [kv[1] for kv in multi_models[0].named_parameters() if kv[0].split('.')[0] in ['visionMLP']]
    optimizer1 = torch.optim.Adam([
                {'params': heads_params1}, # these correspond to the text and vision "heads" (layer before foodSpace and after resnet and text embedder)
                {'params': vision_params, 'lr': lr1*args.freeVision, 'weight_decay': wd1 },  # resnet embeddings
                {'params': C1_params, 'lr': lr1, 'weight_decay': wd1 }
        ], lr=lr1*args.freeHeads, weight_decay=wd1)

    heads_params2 = [kv[1] for kv in multi_models[1].named_parameters() if (kv[0].split('.')[0] not in ['visionMLP', 'textMLP'])]
    wordEmb_params = [kv[1] for kv in multi_models[1].named_parameters() if 'textMLP.mBERT.embeddings' in kv[0] or 'word_embeddings' in kv[0]]
    text_params = [kv[1] for kv in multi_models[1].named_parameters() if ('textMLP' in kv[0]) and ('embeddings' not in kv[0])]
    optimizer2 = torch.optim.Adam([
                {'params': heads_params2}, # these correspond to the text and vision "heads" (layer before foodSpace and after resnet and text embedder)
                {'params': wordEmb_params, 'lr': lr2*args.freeWordEmb, 'weight_decay': wd2 },# word embeddings
                {'params': text_params, 'lr': lr2*args.freeText, 'weight_decay': wd2 }, # text embedder params except word embeddings
                {'params': C2_params, 'lr': lr2, 'weight_decay': wd2 }
        ], lr=lr2*args.freeHeads, weight_decay=wd2)    

    lr_schedu1 = optim.lr_scheduler.MultiStepLR(optimizer1, ms, gamma=gamma)
    lr_schedu2 = optim.lr_scheduler.MultiStepLR(optimizer2, ms, gamma=gamma)

    return multi_models, [C1, C2], [optimizer1, optimizer2], [lr_schedu1, lr_schedu2]