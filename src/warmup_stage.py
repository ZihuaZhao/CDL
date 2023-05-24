import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from tqdm import tqdm

from src.methods import *

class Warmup_Stage():
    def __init__(self, args, summary_writer, loaders, multi_models, preds, optimizers, schedulers):
        self.args = args
        self.summary_writer = summary_writer
        self.train_loader, self.valid_loader, self.test_loader = loaders
        self.multi_models = multi_models
        self.C1, self.C2 = preds
        self.optimizer1, self.optimizer2 = optimizers
        self.scheduler1, self.scheduler2 = schedulers

        self.ce_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.nll_criterion = nn.NLLLoss(reduction='none')

        self.matrix1 = torch.zeros(args.class_num, args.class_num).to(args.device)
        self.matrix2 = torch.zeros(args.class_num, args.class_num).to(args.device)
        for i in range(args.class_num):
            self.matrix1[i, i] = 1
            self.matrix2[i, i] = 1

        self.coup_list = torch.ones(len(self.train_loader.dataset)).to(args.device)
        self.decoup_list = torch.zeros(len(self.train_loader.dataset)).to(args.device)

    def get_return(self):
        return self.matrix1, self.matrix2, self.coup_list, self.decoup_list

    def reset_matrix(self):
        self.matrix1 = torch.zeros(self.args.class_num, self.args.class_num).to(self.args.device)
        self.matrix2 = torch.zeros(self.args.class_num, self.args.class_num).to(self.args.device)
        for i in range(self.args.class_num):
            self.matrix1[i, i] = 1
            self.matrix2[i, i] = 1

    def M_step(self, epoch):
        set_train(self.multi_models)
        for batch_idx, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            batches, targets, index = batch[0], batch[1], batch[2]
            batches, targets = batches[0].to(self.args.device), targets[0].to(self.args.device)
            model = self.multi_models[0]
            model.zero_grad()
            self.optimizer1.zero_grad()

            outputs = model(batches)
            preds = self.C1(outputs)
            preds = torch.softmax(preds, dim=1)
            preds = preds.mm(self.matrix1)

            loss = self.nll_criterion(torch.log(preds), targets)
            loss = loss.mean()
            loss.backward()
            self.optimizer1.step()

        for batch_idx, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            batches, targets, index = batch[0], batch[1], batch[2]
            batches, targets = batches[1].to(self.args.device), targets[1].to(self.args.device)
            model = self.multi_models[1]
            model.zero_grad()
            self.optimizer2.zero_grad()

            outputs = model(batches)
            preds = self.C2(outputs)
            preds = torch.softmax(preds, dim=1)
            preds = preds.mm(self.matrix2)

            loss = self.nll_criterion(torch.log(preds), targets)
            loss = loss.mean()
            loss.backward()
            self.optimizer2.step()
            
        self.scheduler1.step()
        self.scheduler2.step()

    def E_step(self, mode):
        C = self.C1 if mode == 0 else self.C2
        matrix = self.matrix1 if mode == 0 else self.matrix2

        set_eval(self.multi_models)
        with torch.no_grad():
            preds_list = []
            targets_list = []
            for batch_idx, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                batches, targets, index = batch[0], batch[1], batch[2]
                batches, targets = batches[mode].to(self.args.device), targets[mode].to(self.args.device)
                outputs = self.multi_models[mode](batches)

                preds = C(outputs)
                preds = torch.softmax(preds, dim=1)
                preds_list.append(preds)
                targets_list.append(targets)
            preds_list = torch.cat(preds_list, dim=0)
            targets_list = torch.cat(targets_list, dim=0)

            for i in range(self.args.class_num):
                for j in range(self.args.class_num):
                    noisy_idx = (targets_list == j)
                    if preds_list[:, i].sum() == 0:
                        matrix[i, j] = 0
                    else:
                        matrix[i, j] = preds_list[noisy_idx][:, i].sum()
                        matrix[i, j] /= preds_list[:, i].sum()

        if mode == 0:
            self.matrix1 = matrix
        else:
            self.matrix2 = matrix

    def warmup_test(self, data_loader, epoch, mode):
        set_eval(self.multi_models)
        test_loss, loss_list, correct_list, total_list = 0., [0.] * 2, [0.] * 2, [0.] * 2
        length = len(data_loader.dataset)
        cossim_list = np.zeros(length)
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
                batches, targets, index, corr = batch[0], batch[1], batch[2], batch[3]
                batches, targets, corr = [batches[v].to(self.args.device) for v in range(2)], [targets[v].to(self.args.device) for v in range(2)], corr.to(self.args.device)
                outputs = [self.multi_models[v](batches[v]) for v in range(2)]
                preds1 = self.C1(outputs[0])
                preds2 = self.C2(outputs[1])
                cossim_list[index] = np.array(torch.cosine_similarity(preds1, preds2, dim=1).cpu())
                pred = [preds1, preds2]
                losses = []
                for v in range(2):
                    losses.append(self.ce_criterion(pred[v], targets[v]).mean())
                    loss_list[v] += losses[v]
                    _, predicted = pred[v].max(1)
                    total_list[v] += targets[v].size(0)
                    acc = predicted.eq(targets[v]).sum().item()
                    correct_list[v] += acc
                loss = sum(losses)
                test_loss += loss.item()
        acc1 = correct_list[0] / total_list[0]
        acc2 = correct_list[1] / total_list[1]

        self.summary_writer.add_scalar(os.path.join('Image', 'Acc', mode), acc1, epoch)
        self.summary_writer.add_scalar(os.path.join('Text', 'Acc', mode), acc2, epoch)
        self.summary_writer.add_scalar(os.path.join('Loss', mode), test_loss, epoch)
        # if epoch == self.args.warmup_epochs - 1:
        print('Img Acc: ' + str(acc1) + ' Txt Acc: ' + str(acc2))

    def weight_stage(self, epoch):
        set_eval(self.multi_models)
        cdc_clean_weights = AverageMeter()
        cdc_noise_weights = AverageMeter()
        coup_clean_weights = AverageMeter()
        coup_noise_weights = AverageMeter()
        decoup_clean_weights = AverageMeter()
        decoup_noise_weights = AverageMeter()
        map_clean = []
        map_noise = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.train_loader):
                batches, targets, index, clean_corr = batch[0], batch[1], batch[2], batch[3]
                batches, targets, index, clean_corr = [batches[v].to(self.args.device) for v in range(2)], [targets[v].to(self.args.device) for v in range(2)], index.to(self.args.device), clean_corr.to(self.args.device)
                outputs = [self.multi_models[v](batches[v]) for v in range(2)]
                preds1 = self.C1(outputs[0])
                preds2 = self.C2(outputs[1])

                probs1 = torch.softmax(preds1, dim=1)
                probs2 = torch.softmax(preds2, dim=1)

                s_coup, s_coup_mat = calculate_coup(self.args, self.matrix1, self.matrix2, probs1, probs2, targets, True)
                s_decoup, s_decoup_mat = calculate_decoup(self.args, self.matrix1, self.matrix2, probs1, probs2, targets, True)

                weight_map = s_coup_mat + s_decoup_mat
                clean_idx = torch.nonzero(clean_corr).squeeze(0)
                noise_idx = torch.nonzero(1 - clean_corr).squeeze(0)
                weight_map_clean = weight_map[clean_idx].squeeze(1).detach().cpu().tolist()
                weight_map_noise = weight_map[noise_idx].squeeze(1).detach().cpu().tolist()
                map_clean = map_clean + weight_map_clean
                map_noise = map_noise + weight_map_noise

                self.coup_list[index] = s_coup
                self.decoup_list[index] = s_decoup

                noise_corr = 1 - clean_corr
                clean_weight = (clean_corr * (s_coup - s_decoup)).sum() / clean_corr.sum()
                noise_weight = (noise_corr * (s_coup - s_decoup)).sum() / noise_corr.sum()
                coup_clean = (clean_corr * s_coup).sum() / clean_corr.sum()
                coup_noise = (noise_corr * s_coup).sum() / noise_corr.sum()
                decoup_clean = (clean_corr * s_decoup).sum() / clean_corr.sum()
                decoup_noise = (noise_corr * s_decoup).sum() / noise_corr.sum()

                cdc_clean_weights.update(clean_weight.item(), outputs[0].size(0))
                cdc_noise_weights.update(noise_weight.item(), outputs[0].size(0))
                coup_clean_weights.update(coup_clean.item(), outputs[0].size(0))
                coup_noise_weights.update(coup_noise.item(), outputs[0].size(0))
                decoup_clean_weights.update(decoup_clean.item(), outputs[0].size(0))
                decoup_noise_weights.update(decoup_noise.item(), outputs[0].size(0))

        map_clean = np.array(map_clean)
        map_noise = np.array(map_noise)
        map_clean = map_clean.sum(0)
        map_noise = map_noise.sum(0)

        self.summary_writer.add_scalar('Warmup_CDC_Weights/Clean', cdc_clean_weights.avg, epoch)
        self.summary_writer.add_scalar('Warmup_CDC_Weights/Noise', cdc_noise_weights.avg, epoch)
        self.summary_writer.add_scalar('Warmup_CDC_Weights/Dist', cdc_clean_weights.avg - cdc_noise_weights.avg, epoch)
        self.summary_writer.add_scalar('Warmup_CDC_Coup/Clean', coup_clean_weights.avg, epoch)
        self.summary_writer.add_scalar('Warmup_CDC_Coup/Noise', coup_noise_weights.avg, epoch)
        self.summary_writer.add_scalar('Warmup_CDC_Coup/Dist', coup_clean_weights.avg - coup_noise_weights.avg, epoch)
        self.summary_writer.add_scalar('Warmup_CDC_Decoup/Clean', decoup_clean_weights.avg, epoch)
        self.summary_writer.add_scalar('Warmup_CDC_Decoup/Noise', decoup_noise_weights.avg, epoch)
        self.summary_writer.add_scalar('Warmup_CDC_Decoup/Dist', decoup_clean_weights.avg - decoup_noise_weights.avg, epoch)
    
    def save_checkpoint(self, epoch):
        save_path = os.path.join(self.args.ckpt_dir, str(epoch))
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.multi_models[0].state_dict(), os.path.join(save_path, 'multi_model_0.pkl'))
        torch.save(self.multi_models[1].state_dict(), os.path.join(save_path, 'multi_model_1.pkl'))
        torch.save(self.C1, os.path.join(save_path, 'C1.pkl'))
        torch.save(self.C2, os.path.join(save_path, 'C2.pkl'))
        torch.save(self.coup_list, os.path.join(save_path, 'coup_list.pkl'))
        torch.save(self.decoup_list, os.path.join(save_path, 'decoup_list.pkl'))
        torch.save(self.matrix1, os.path.join(save_path, 'matrix1.pkl'))
        torch.save(self.matrix2, os.path.join(save_path, 'matrix2.pkl'))

    def load_checkpoint(self, path, epoch):
        save_path = os.path.join(path, str(epoch))
        self.multi_models[0].load_state_dict(torch.load(os.path.join(save_path, 'multi_model_0.pkl'), map_location=self.args.device))
        self.multi_models[1].load_state_dict(torch.load(os.path.join(save_path, 'multi_model_1.pkl'), map_location=self.args.device))
        self.C1 = torch.load(os.path.join(save_path, 'C1.pkl'), map_location=self.args.device)
        self.C2 = torch.load(os.path.join(save_path, 'C2.pkl'), map_location=self.args.device)
        self.coup_list = torch.load(os.path.join(save_path, 'coup_list.pkl'), map_location=self.args.device)
        self.decoup_list = torch.load(os.path.join(save_path, 'decoup_list.pkl'), map_location=self.args.device)
        self.matrix1 = torch.load(os.path.join(save_path, 'matrix1.pkl'), map_location=self.args.device)
        self.matrix2 = torch.load(os.path.join(save_path, 'matrix2.pkl'), map_location=self.args.device)