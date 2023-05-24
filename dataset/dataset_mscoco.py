import torch.utils.data as data
import os
import json
import pickle
import numpy as np
import lmdb
import random
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from PIL import Image, ExifTags
from torch.utils.data.dataloader import default_collate
from src.utils_recipe import PadToSquareResize, AverageMeter, SubsetSequentialSampler, cosine_distance, worker_init_fn, get_variable
from src.utils_recipe import get_list_of_files
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer

def my_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = list(filter (lambda x:x is not None, batch))
    return default_collate(batch)

def default_loader(path):
    im = Image.open(path).convert('RGB')
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(im._getexif().items())

        if exif[orientation] == 3:
            im = im.rotate(180, expand=True)
        elif exif[orientation] == 6:
            im = im.rotate(270, expand=True)
        elif exif[orientation] == 8:
            im = im.rotate(90, expand=True)
    except:
        pass
    return im

def error_catching_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(im._getexif().items())

            if exif[orientation] == 3:
                im = im.rotate(180, expand=True)
            elif exif[orientation] == 6:
                im = im.rotate(270, expand=True)
            elif exif[orientation] == 8:
                im = im.rotate(90, expand=True)
        except:
            pass

        return im
    except:
        # print('bad image: '+path, end =" ")#print(file=sys.stderr)
        return Image.new('RGB', (224, 224), 'white')

def define_transform(normalize, mode='train'):
    if mode == 'train':
        transform=transforms.Compose([transforms.RandomChoice([
                                        PadToSquareResize(resize=256, padding_mode='random'),
                                        transforms.Resize((256, 256))]),
                                    transforms.RandomRotation(10),
                                    transforms.RandomCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    else:
        transform=transforms.Compose([
                            PadToSquareResize(resize=256, padding_mode='reflect'),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize])
    return transform


class MSCOCOLoader(data.Dataset):
    def __init__(self, transform=None, loader=default_loader, partition=None, opts=None, aug='', language=0, pred=[], prob=[], mode='train'):
        
        self.data = json.load(open(os.path.join(opts.data_dir, 'dataset_with_label/dataset_coco_with_label.' + partition + '.json'), 'r'))

        self.transform = transform
        self.loader = loader
        self.opts = opts
        self.aug = aug
        self.language = language
        self.partition = partition
        self.pred = pred
        self.prob = prob
        self.mode = mode
        self.class_num = opts.class_num

        self.img_labels = []
        self.txt_labels = []
        self.img_list = []
        self.cap_list = []
        self.labels = self.img_labels
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        for i in range(len(self.data)):
            self.img_labels.append(self.data[i]['label'])
            self.txt_labels.append(self.data[i]['label'])
            self.img_list.append(os.path.join(self.data[i]['filepath'], self.data[i]['filename']))
            self.cap_list.append(self.data[i]['sentences'])

        if partition == 'train':
            length = len(self.img_list)
            opts.noise_corr_path = os.path.join(opts.root_dir, 'noisy_labels', opts.data_name, 'noise_corr_%g' % (opts.r_corr))
            opts.noise_label_path = os.path.join(opts.root_dir, 'noisy_labels', opts.data_name, 'noise_label_%g' % (opts.r_label))
            if not os.path.exists(opts.noise_corr_path):
                os.mkdir(opts.noise_corr_path)
            if not os.path.exists(opts.noise_label_path):
                os.mkdir(opts.noise_label_path)
            if opts.use_noise_file:
                self.img_corr = np.load(os.path.join(opts.noise_corr_path, 'noise_img_corr.npy'))
                self.txt_corr = np.load(os.path.join(opts.noise_corr_path, 'noise_txt_corr.npy'))
                self.clean_corr = (self.img_corr == np.arange(length)).astype(int) * (self.txt_corr == np.arange(length)).astype(int)

                self.noise_labels = np.load(os.path.join(opts.noise_label_path, 'noise_labels_' + self.opts.noise_mode + '.npy'))
                self.labels = self.noise_labels
            else:
                self.img_corr = np.arange(length)
                self.txt_corr = np.arange(length)
                img_idx = np.arange(length)
                txt_idx = np.arange(length)
                np.random.shuffle(img_idx)
                np.random.shuffle(txt_idx)
                noise_length = int(opts.r_corr * length / 2)

                shuf_img_idx = self.img_corr[img_idx[:noise_length]]
                shuf_txt_idx = self.txt_corr[txt_idx[:noise_length]]
                np.random.shuffle(shuf_img_idx)
                np.random.shuffle(shuf_txt_idx)
                self.img_corr[img_idx[:noise_length]] = shuf_img_idx
                self.txt_corr[txt_idx[:noise_length]] = shuf_txt_idx
                self.clean_corr = (self.img_corr == np.arange(length)).astype(int) * (self.txt_corr == np.arange(length)).astype(int)

                np.save(os.path.join(opts.noise_corr_path, 'noise_img_corr.npy'), self.img_corr)
                np.save(os.path.join(opts.noise_corr_path, 'noise_txt_corr.npy'), self.txt_corr)

                range_idx = np.arange(length)
                np.random.shuffle(range_idx)
                noise_length = int(opts.r_label * length)

                range_class = np.arange(opts.class_num)
                self.transition = {i: i for i in range(opts.class_num)}
                half_num = int(opts.class_num // 2)
                for i in range(half_num):
                    self.transition[range_class[i]] = int(range_class[half_num + i])

                noise_idx = range_idx[: noise_length]
                self.noise_labels = []
                for idx in range(length):
                    if idx in noise_idx:
                        if self.opts.noise_mode == 'sym':
                            noise_label = int(random.randint(0, self.opts.class_num))
                            self.noise_labels.append(noise_label)
                        elif self.opts.noise_mode == 'asym':
                            noise_label = self.transition[self.img_labels[idx]]
                            self.noise_labels.append(noise_label)
                    else:
                        self.noise_labels.append(self.img_labels[idx])
                
                np.save(os.path.join(opts.noise_label_path, 'noise_labels_' + self.opts.noise_mode + '.npy'), self.noise_labels)
                self.labels = self.noise_labels
        else:
            length = len(self.img_labels)
            self.img_corr = np.arange(length)
            self.txt_corr = np.arange(length)
            self.clean_corr = (self.img_corr == np.arange(length)).astype(int) * (self.txt_corr == np.arange(length)).astype(int)
            self.semi_list = np.ones(length)

    def __getitem__(self, index):
        image_id = self.img_corr[index] if self.partition == 'train' else index

        image = Image.open(os.path.join(self.opts.data_dir, self.img_list[image_id])).convert('RGB')
        image = self.transform(image)

        cap_id = self.txt_corr[index] if self.partition == 'train' else index
        if self.partition == 'train':
            raw_caption = self.cap_list[cap_id][random.randint(0, 4)]['raw']
        else:
            raw_caption = self.cap_list[cap_id][0]['raw']
        caption = self.tokenizer(raw_caption)['input_ids']
        caption += [0 for i in range(128 - len(caption))]
        caption = torch.tensor(caption, dtype=torch.int64)

        img_label = self.img_labels[self.img_corr[index]]
        txt_label = self.txt_labels[self.txt_corr[index]]
        label = self.labels[index]
        
        if self.opts.clean_label:
            return [image, caption], [img_label, txt_label], index, self.clean_corr[index], [img_label, txt_label], index
        else:
            return [image, caption], [label, label], index, self.clean_corr[index], [img_label, txt_label], index

    def __len__(self):
        return len(self.img_labels)