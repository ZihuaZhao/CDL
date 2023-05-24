import torch.utils.data as data
import os
import pickle
import numpy as np
import lmdb
import random
import pandas as pd
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

class ConceptualCaptions(data.Dataset):
    def __init__(self, transform=None, loader=default_loader, partition=None, opts=None, aug='', language=0, pred=[], prob=[], mode='train'):
        self.opts = opts
        self.transform = transform
        self.loader = loader
        self.aug = aug
        self.language = language
        self.partition = partition
        self.pred = pred
        self.prob = prob
        self.mode = mode
        self.class_num = opts.class_num
        
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        self.ids = np.load(os.path.join(opts.data_dir, 'keys_' + partition + '.npy'))
        
        self.map_list = np.load(os.path.join(opts.data_dir, 'map_all.npy'))

        self.label_list = np.load(os.path.join(opts.data_dir, 'label_list_all.npy'))

        self.img_corr = np.arange(len(self.ids))
        self.txt_corr = np.arange(len(self.ids))

        with open(os.path.join(opts.data_dir, 'cap_list_all.txt'), 'r') as f:
            self.cap_list = f.readlines()
        for i in range(len(self.cap_list)):
            self.cap_list[i] = self.cap_list[i].strip('\n')

        self.img_labels = []
        self.txt_labels = []
        self.labels = self.img_labels

    def __getitem__(self, index):
        image_id = self.img_corr[index] if self.partition == 'train' else index
        image_id = self.map_list[image_id]

        image = Image.open(os.path.join(self.opts.data_dir, 'images', 'image_' + str(image_id) + '.jpg')).convert('RGB')
        image = self.transform(image)

        txt_id = self.txt_corr[index] if self.partition == 'train' else index
        raw_caption = self.cap_list[txt_id]
        caption = self.tokenizer(raw_caption)['input_ids']
        caption += [0 for i in range(128 - len(caption))]
        caption = torch.tensor(caption, dtype=torch.int64)

        label = self.label_list[index]

        return [image, caption], [label, label], index, 1, index
        
    def __len__(self):
        return len(self.ids)