import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import nets as models
from config.config_recipe import args
# from config.config_cc import args
# from config.config_mscoco import args
if args.data_name == 'recipe':
    from dataset.dataset_recipe1m import foodSpaceLoader, error_catching_loader, define_transform
    from src.init_model import init_model_recipe
elif args.data_name == 'cc':
    from dataset.dataset_cc import ConceptualCaptions, error_catching_loader, define_transform
    from src.init_model import init_model_recipe
elif args.data_name == 'mscoco':
    from dataset.dataset_mscoco import MSCOCOLoader, error_catching_loader, define_transform
    from src.init_model import init_model_recipe
from src.warmup_stage import Warmup_Stage
from src.train_stage import Train_Stage


def main():
    args.device = torch.device('cuda:' + str(args.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
    args.log_dir = os.path.join(args.root_dir, 'logs_' + args.data_name, args.log_name)
    args.ckpt_dir = os.path.join(args.root_dir, 'ckpt_' + args.data_name, args.log_name)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    summary_writer = SummaryWriter(log_dir=args.log_dir)

    print('===> Preparing data ..')
    if args.data_name == 'recipe':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = define_transform(normalize, 'train')
        train_dataset = foodSpaceLoader(train_transform, error_catching_loader, 'train', args)
        valid_transform = define_transform(normalize, 'val')
        valid_dataset = foodSpaceLoader(valid_transform, error_catching_loader, 'val', args)
        test_transform = define_transform(normalize, 'test')
        test_dataset = foodSpaceLoader(test_transform, error_catching_loader, 'test', args)
        args.len_traindata = len(train_dataset)
    elif args.data_name == 'cc':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = define_transform(normalize, 'train')
        train_dataset = ConceptualCaptions(train_transform, error_catching_loader, 'train', args)
        valid_transform = define_transform(normalize, 'val')
        valid_dataset = ConceptualCaptions(valid_transform, error_catching_loader, 'val', args)
        test_transform = define_transform(normalize, 'test')
        test_dataset = ConceptualCaptions(test_transform, error_catching_loader, 'test', args)
        args.len_traindata = len(train_dataset)
    elif args.data_name == 'mscoco':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = define_transform(normalize, 'train')
        train_dataset = MSCOCOLoader(train_transform, error_catching_loader, 'train', args)
        valid_transform = define_transform(normalize, 'val')
        valid_dataset = MSCOCOLoader(valid_transform, error_catching_loader, 'val', args)
        test_transform = define_transform(normalize, 'test')
        test_dataset = MSCOCOLoader(test_transform, error_catching_loader, 'test', args)
        args.len_traindata = len(train_dataset)
    
    print('===> Preparing dataloader ..')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )

    # Preparing Models & Optimizers & Loss Functions
    warmup_models, warmup_preds, warmup_optimizers, warmup_schedulers = init_model_recipe(args, 'warmup')
    
    print('===> Warmup Stage ..')
    warmup_stage = Warmup_Stage(args, summary_writer, [train_loader, valid_loader, test_loader], warmup_models, warmup_preds, warmup_optimizers, warmup_schedulers)
    if args.warmup_epochs > 0:
        if args.load_warmup_models != -1:
            epoch = args.load_warmup_models
            warmup_stage.load_checkpoint(args.load_warmup_path, args.load_warmup_models)
            warmup_stage.weight_stage(epoch)
            
            warmup_stage.warmup_test(train_loader, epoch, 'train')
            warmup_stage.warmup_test(valid_loader, epoch, 'valid')
            warmup_stage.warmup_test(test_loader, epoch, 'test')
        for epoch in range(args.warmup_epochs):
            print('Epoch: ' + str(epoch + 1) + ' / ' + str(args.warmup_epochs))
            if epoch == args.e_epoch1 - 1:
                warmup_stage.E_step(0)
            if epoch == args.e_epoch2 - 1:
                warmup_stage.E_step(1)
            warmup_stage.M_step(epoch + 1)

            warmup_stage.warmup_test(train_loader, epoch + 1, 'train')
            warmup_stage.warmup_test(valid_loader, epoch + 1, 'valid')
            warmup_stage.warmup_test(test_loader, epoch + 1, 'test')
            warmup_stage.weight_stage(epoch + 1)
            warmup_stage.save_checkpoint(epoch + 1)
    matrix1, matrix2, coup_list, decoup_list = warmup_stage.get_return()

    print('===> Training Stage ..')
    train_models, train_preds, train_optimizers, train_schedulers = init_model_recipe(args, 'train', model=warmup_models, preds=warmup_preds)
    train_stage = Train_Stage(args, summary_writer, [train_loader, valid_loader, test_loader], train_models, train_preds, train_optimizers, train_schedulers, [matrix1, matrix2], coup_list, decoup_list)
    for epoch in range(args.train_epochs):
        train_stage.train(epoch + 1)
        train_stage.test(epoch + 1)
        if (epoch + 1) % 5 == 0:
            for epoch_pred in range(5):
                train_stage.train_pred(epoch_pred + epoch - 4)
        
def seed_torch(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed_torch(args.seed)
    main()

        
def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed_torch()
    main()