import os
import numpy as np
import scipy
import scipy.spatial
import random
import json

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from tqdm import tqdm

from src.methods import *
from src.bar_show import progress_bar

class Train_Stage():
    def __init__(self, args, summary_writer, loaders, multi_models, preds, optimizers, schedulers, matrixs, coup_list, decoup_list):
        self.args = args
        self.summary_writer = summary_writer
        self.train_loader, self.valid_loader, self.test_loader = loaders
        self.multi_models = multi_models
        self.C1, self.C2 = preds
        self.optimizer1, self.optimizer2 = optimizers
        self.lr_scheduler1, self.lr_scheduler2 = schedulers
        self.matrix1, self.matrix2 = matrixs
        self.coup_list, self.decoup_list = coup_list, decoup_list

        self.nll_criterion = nn.NLLLoss(reduction='none')
        self.ce_criterion = torch.nn.CrossEntropyLoss()
        self.cl_criterion = ContrastiveLoss(0.2)   

        self.best_recall = 0
    
    def train(self, epoch):
        print('\nEpoch: %d / %d' % (epoch, self.args.train_epochs))
        set_train(self.multi_models)
        train_loss, loss_list, total_list = 0., [0.] * 2, [0.] * 2,

        for batch_idx, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            batches, targets, index, clean_corr = batch[0], batch[1], batch[2], batch[3]
            batches, targets, index, clean_corr = [batches[v].to(self.args.device) for v in range(2)], [targets[v].to(self.args.device) for v in range(2)], index.to(self.args.device), clean_corr.to(self.args.device)

            for v in range(2):
                self.multi_models[v].zero_grad()
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()

            outputs = [self.multi_models[v](batches[v]) for v in range(2)]
            sims = outputs[0].mm(outputs[1].t())
            loss_corr = self.cl_criterion(sims, hard_negative=False, mode='train')

            if self.args.label_modeling == 'cdl':
                loss_corr = loss_corr * self.coup_list[index]

            loss = loss_corr.mean()

            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()
            train_loss += loss.item()

            losses = [loss, loss]
            for v in range(2):
                loss_list[v] += losses[v]
                total_list[v] += targets[v].size(0)
            progress_bar(batch_idx, len(self.train_loader), 'Loss: %.3f | LR: %g'
                         % (train_loss / (batch_idx + 1), self.optimizer1.param_groups[0]['lr']))

        self.lr_scheduler1.step()
        self.lr_scheduler2.step()
        train_dict = {('view_%d_loss' % v): loss_list[v] / len(self.train_loader) for v in range(2)}
        train_dict['sum_loss'] = train_loss / len(self.train_loader)
        self.summary_writer.add_scalars('Loss/train', train_dict, epoch)

    def train_pred(self, epoch):
        print('\nEpoch: %d / %d' % (epoch, self.args.train_epochs))
        set_train(self.multi_models)
        train_loss, loss_list, total_list, correct_list = 0., [0.] * 2, [0.] * 2, [0.] * 2

        cdc_clean_weights = AverageMeter()
        cdc_noise_weights = AverageMeter()
        coup_clean_weights = AverageMeter()
        coup_noise_weights = AverageMeter()
        decoup_clean_weights = AverageMeter()
        decoup_noise_weights = AverageMeter()
        tau_clean_weights = AverageMeter()
        tau_noise_weights = AverageMeter()
        for batch_idx, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            batches, targets, index, clean_corr = batch[0], batch[1], batch[2], batch[3]
            batches, targets, index, clean_corr = [batches[v].to(self.args.device) for v in range(2)], [targets[v].to(self.args.device) for v in range(2)], index.to(self.args.device), clean_corr.to(self.args.device)

            for v in range(2):
                self.multi_models[v].zero_grad()
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()

            with torch.no_grad():
                outputs = [self.multi_models[v](batches[v]) for v in range(2)]

            pred1 = self.C1(outputs[0])
            pred2 = self.C2(outputs[1])
            prob1 = torch.softmax(pred1, dim=1)
            prob2 = torch.softmax(pred2, dim=1)
        
            s_coup, s_coup_mat = calculate_coup(self.args, self.matrix1, self.matrix2, prob1, prob2, targets, True)
            s_decoup, s_decoup_mat = calculate_decoup(self.args, self.matrix1, self.matrix2, prob1, prob2, targets, True)

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
            
            if epoch >= self.args.weight_epoch:
                self.coup_list = s_coup.detach()
                self.decoup_list = s_decoup.detach()

            loss = self.nll_criterion(prob1.mm(self.matrix1), targets[0]) + self.nll_criterion(prob2.mm(self.matrix2), targets[1])
            loss = loss.mean()

            preds = [pred1, pred2]
            for v in range(2):
                _, predicted = preds[v].max(1)
                total_list[v] += targets[v].size(0)
                acc = predicted.eq(targets[v]).sum().item()
                correct_list[v] += acc

            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()
        
        self.summary_writer.add_scalar('Train_CDC_Weights/Clean', cdc_clean_weights.avg, epoch)
        self.summary_writer.add_scalar('Train_CDC_Weights/Noise', cdc_noise_weights.avg, epoch)
        self.summary_writer.add_scalar('Train_CDC_Weights/Dist', cdc_clean_weights.avg - cdc_noise_weights.avg, epoch)
        self.summary_writer.add_scalar('Train_CDC_Weights/Rate', cdc_clean_weights.avg / cdc_noise_weights.avg, epoch)
        self.summary_writer.add_scalar('Train_CDC_Coup/Clean', coup_clean_weights.avg, epoch)
        self.summary_writer.add_scalar('Train_CDC_Coup/Noise', coup_noise_weights.avg, epoch)
        self.summary_writer.add_scalar('Train_CDC_Coup/Dist', coup_clean_weights.avg - coup_noise_weights.avg, epoch)
        self.summary_writer.add_scalar('Train_CDC_Coup/Rate', coup_clean_weights.avg / coup_noise_weights.avg, epoch)
        self.summary_writer.add_scalar('Train_CDC_Decoup/Clean', decoup_clean_weights.avg, epoch)
        self.summary_writer.add_scalar('Train_CDC_Decoup/Noise', decoup_noise_weights.avg, epoch)
        self.summary_writer.add_scalar('Train_CDC_Decoup/Dist', decoup_clean_weights.avg - decoup_noise_weights.avg, epoch)
        self.summary_writer.add_scalar('Train_CDC_Decoup/Rate', decoup_clean_weights.avg / decoup_noise_weights.avg, epoch)
        self.summary_writer.add_scalar('Train_CDC_Tau/Clean', tau_clean_weights.avg, epoch)
        self.summary_writer.add_scalar('Train_CDC_Tau/Noise', tau_noise_weights.avg, epoch)
        self.summary_writer.add_scalar('Train_CDC_Tau/Dist', tau_clean_weights.avg - tau_noise_weights.avg, epoch)
        self.summary_writer.add_scalar('Train_Acc_image', torch.tensor(correct_list[0]/total_list[0]), epoch)
        self.summary_writer.add_scalar('Train_Acc_text', torch.tensor(correct_list[1]/total_list[1]), epoch)

    def eval_dataset(self, data_loader, epoch, mode='test'):
        fea, lab, ind = [[] for _ in range(2)], [[] for _ in range(2)], [[] for _ in range(2)]
        test_loss, loss_list, correct_list, total_list = 0., [0.] * 2, [0.] * 2, [0.] * 2
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
                batches, targets, index = batch[0], batch[1], batch[2]
                batches, targets = [batches[v].to(self.args.device) for v in range(2)], [targets[v].to(self.args.device) for v in range(2)]
                outputs = [self.multi_models[v](batches[v]) for v in range(2)]
                pred, losses = [], []
                for v in range(2):
                    C = self.C1 if v == 0 else self.C2
                    fea[v].append(outputs[v])
                    lab[v].append(targets[v])
                    ind[v].append(index)
                    pred.append(C(outputs[v]))
                    losses.append(self.ce_criterion(pred[v], targets[v]))
                    loss_list[v] += losses[v]
                    _, predicted = pred[v].max(1)
                    total_list[v] += targets[v].size(0)
                    acc = predicted.eq(targets[v]).sum().item()
                    correct_list[v] += acc
                loss = sum(losses)
                test_loss += loss.item()

            fea = [torch.cat(fea[v]).cpu().detach().numpy() for v in range(2)]
            lab = [torch.cat(lab[v]).cpu().detach().numpy() for v in range(2)]
            ind = [torch.cat(ind[v]).cpu().detach().numpy() for v in range(2)]
        test_dict = {('view_%d_loss' % v): loss_list[v] / len(data_loader) for v in range(2)}
        test_dict['sum_loss'] = test_loss / len(data_loader)
        self.summary_writer.add_scalars('Loss/' + mode, test_dict, epoch)
        self.summary_writer.add_scalars('Accuracy/' + mode, {('view_%d_acc' % v): correct_list[v] / total_list[v] for v in range(2)}, epoch)
        return fea, lab, ind

    def test_recipe(self, epoch, data_loader, mode):
        losses = AverageMeter()
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            batches, targets, index, rec_ids = batch[0], batch[1], batch[2], batch[5]
            batches, targets = [batches[v].to(self.args.device) for v in range(2)], [targets[v].to(self.args.device) for v in range(2)]

            with torch.no_grad():
                output = [self.multi_models[v](batches[v]) for v in range(2)]

                if i == 0:
                    data0 = output[0].data.cpu().numpy()
                    data1 = output[1].data.cpu().numpy()
                    data2 = rec_ids
                else:
                    data0 = np.concatenate((data0, output[0].data.cpu().numpy()), axis=0)
                    data1 = np.concatenate((data1, output[1].data.cpu().numpy()), axis=0)
                    data2 = np.concatenate((data2, rec_ids), axis=0)

                sims = output[0].mm(output[1].t())
                loss = self.cl_criterion(sims, hard_negative=False, mode='warmup')

                losses.update(loss.item(), output[0].size(0))

        mode_tmp = mode
        mode = mode + '/img2txt'
        medR, recall = rank(self.args, data0, data1, data2)
        print('\t* Val medR {medR:.4f}\tRecall {recall}'.format(medR=medR, recall=recall))
        self.summary_writer.add_scalar(mode + "/val_loss", losses.avg, epoch)
        self.summary_writer.add_scalar(mode + "/recall_1", recall[1], epoch)
        self.summary_writer.add_scalar(mode + "/recall_5", recall[5], epoch)
        self.summary_writer.add_scalar(mode + "/recall_10", recall[10], epoch)
        self.summary_writer.add_scalar(mode + "/recall_1-5-10", recall[10]+recall[5]+recall[1], epoch)

        mode = mode_tmp + '/txt2img'
        medR, recall = rank(self.args, data1, data0, data2)
        print('\t* Val medR {medR:.4f}\tRecall {recall}'.format(medR=medR, recall=recall))
        self.summary_writer.add_scalar(mode + "/val_loss", losses.avg, epoch)
        self.summary_writer.add_scalar(mode + "/recall_1", recall[1], epoch)
        self.summary_writer.add_scalar(mode + "/recall_5", recall[5], epoch)
        self.summary_writer.add_scalar(mode + "/recall_10", recall[10], epoch)
        self.summary_writer.add_scalar(mode + "/recall_1-5-10", recall[10]+recall[5]+recall[1], epoch)

        return medR, recall
    
    def test_mscoco(self, epoch, data_loader, mode):
        img_embs = None
        txt_embs = None
        img_ids = None
        txt_ids = None
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            batches, targets, index, rec_ids = batch[0], batch[1], batch[2], batch[5]
            batches, targets = [batches[v].to(self.args.device) for v in range(2)], [targets[v].to(self.args.device) for v in range(2)]

            with torch.no_grad():
                output = [self.multi_models[v](batches[v]) for v in range(2)]
                img_emb, txt_emb = output
                if img_embs is None:
                    img_embs = torch.zeros((len(data_loader.dataset), img_emb.size(1))).cuda(self.args.device)
                    txt_embs = torch.zeros((len(data_loader.dataset), txt_emb.size(1))).cuda(self.args.device)
                img_embs[index] = img_emb.data
                txt_embs[index] = txt_emb.data
                if img_ids is None:
                    img_ids = index
                    txt_ids = index
                else:
                    img_ids = torch.cat(tuple([img_ids, index]), dim=0)
                    txt_ids = torch.cat(tuple([txt_ids, index]), dim=0)
        
        iids = img_ids
        tiids = txt_ids

        if mode == 'train':
            img_embs = img_embs[: 1000]
            txt_embs = txt_embs[: 1000]
            iids = iids[: 1000]
            tiids = tiids[: 1000]

        scores = img_embs @ txt_embs.t()

        print("scores: {}".format(scores.size()))
        print("iids: {}".format(iids.size()))
        print("tiids: {}".format(tiids.size()))

        topk10 = scores.topk(10, dim=1)
        topk5 = scores.topk(5, dim=1)
        topk1 = scores.topk(1, dim=1)
        
        topk10_iids = tiids[topk10.indices]
        topk5_iids = tiids[topk5.indices]
        topk1_iids = tiids[topk1.indices]

        tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
        tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
        tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

        topk10 = scores.topk(10, dim=0)
        topk5 = scores.topk(5, dim=0)
        topk1 = scores.topk(1, dim=0)
        topk10_iids = iids[topk10.indices]
        topk5_iids = iids[topk5.indices]
        topk1_iids = iids[topk1.indices]

        ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
        ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
        ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

        eval_result = {
            "tr_r10": tr_r10.item() * 100.0, 
            "tr_r5": tr_r5.item() * 100.0, 
            "tr_r1": tr_r1.item() * 100.0, 
            "ir_r10": ir_r10.item() * 100.0, 
            "ir_r5": ir_r5.item() * 100.0, 
            "ir_r1": ir_r1.item() * 100.0, 
            "average_score": 100.0 * (tr_r1 + tr_r5 + tr_r10 + ir_r1 + ir_r5 + ir_r10).item() / 6.0, 
        }

        self.summary_writer.add_scalar(mode + '/tr_r10', tr_r10, epoch)
        self.summary_writer.add_scalar(mode + '/tr_r5', tr_r5, epoch)
        self.summary_writer.add_scalar(mode + '/tr_r1', tr_r1, epoch)
        self.summary_writer.add_scalar(mode + '/ir_r10', ir_r10, epoch)
        self.summary_writer.add_scalar(mode + '/ir_r5', ir_r5, epoch)
        self.summary_writer.add_scalar(mode + '/ir_r1', ir_r1, epoch)
        self.summary_writer.add_scalar(mode + '/average_score', (tr_r1 + tr_r5 + tr_r10 + ir_r1 + ir_r5 + ir_r10) / 6.0, epoch)

        print('* Eval result = %s' % json.dumps(eval_result))
        return

    def test(self, epoch):
        set_eval(self.multi_models)
        
        if self.args.data_name == 'recipe' or self.args.data_name == 'cc':
            self.test_recipe(epoch, self.train_loader, 'train')
            self.test_recipe(epoch, self.valid_loader, 'valid')
            self.test_recipe(epoch, self.test_loader, 'test')
        elif self.args.data_name == 'mscoco':
            self.test_mscoco(epoch, self.train_loader, 'train')
            self.test_mscoco(epoch, self.valid_loader, 'valid')
            self.test_mscoco(epoch, self.test_loader, 'test')
        else:
            self.test_dataset(epoch, self.train_loader, 'train')
            self.test_dataset(epoch, self.valid_loader, 'valid')
            self.test_dataset(epoch, self.test_loader, 'test')

    def multiview_test(self, fea, lab):
        sim = scipy.spatial.distance.cdist(fea[1], fea[0], metric='cosine')
        (r1, r5, r10, medr, meanr) = i2t(len(fea[0]), sim)
        print("Image to text: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(r1, r5, r10, medr, meanr))

        (r1i, r5i, r10i, medri, meanri) = t2i(len(fea[0]), sim)
        print("Text to image: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(r1i, r5i, r10i, medri, meanri))

        MAPs = np.zeros([2, 2])
        val_dict = {}
        print_str = ''
        for i in range(2):
            for j in range(2):
                if i == j:
                    continue
                MAPs[i, j] = fx_calc_map_label(fea[j], lab[j], fea[i], lab[i], k=0, metric='cosine')[0]
                key = '%s2%s' % (self.args.views[i], self.args.views[j])
                val_dict[key] = MAPs[i, j]
                print_str = print_str + key + ': %.3f\t' % val_dict[key]
        return val_dict, print_str

    def end_train(self, epoch):
        print('Evaluation on Last Epoch:')
        fea, lab, _ = self.eval_dataset(self.test_loader, epoch, 'test')
        test_dict, print_str = self.multiview_test(fea, lab)
        print(print_str)

        print('Evaluation on Best Validation:')
        [self.multi_models[v].load_state_dict(self.multi_model_state_dict[v]) for v in range(2)]
        fea, lab, _ = self.eval_dataset(self.test_loader, epoch, 'test')
        test_dict, print_str = self.multiview_test(fea, lab)
        print(print_str)
        import scipy.io as sio
        save_dict = dict(**{self.args.views[v]: fea[v] for v in range(2)}, **{self.args.views[v] + '_lab': lab[v] for v in range(2)})
        sio.savemat('features/%s_%g.mat' % (self.args.data_name, self.args.r_label), save_dict)