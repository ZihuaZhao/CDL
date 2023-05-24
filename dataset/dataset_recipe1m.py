import torch.utils.data as data
import os
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


class foodSpaceLoader(data.Dataset):
    def __init__(self, transform=None, loader=default_loader, partition=None, opts=None, aug='', language=0, pred=[], prob=[], mode='train'):
        
        self.env = lmdb.open(os.path.join(opts.data_dir, partition + '_lmdb'), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)

        with open(os.path.join(opts.data_dir, 'keys', opts.data_scale, partition + '_keys.pkl'), 'rb') as f:
            self.ids = pickle.load(f)

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
        self.labels = self.img_labels

        if partition == 'train':
            self.img_labels = np.load(os.path.join(opts.data_dir, 'noisy_labels', opts.data_scale, 'img_labels.npy'))
            self.txt_labels = np.load(os.path.join(opts.data_dir, 'noisy_labels', opts.data_scale, 'txt_labels.npy'))
            
            length = len(self.ids)
            opts.noise_corr_path = os.path.join(opts.data_dir, 'noisy_labels', opts.data_scale, 'noise_corr_%g' % (opts.r_corr))
            opts.noise_label_path = os.path.join(opts.data_dir, 'noisy_labels', opts.data_scale, 'noise_label_%g' % (opts.r_label))
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
            length = len(self.ids)
            self.img_corr = np.arange(length)
            self.txt_corr = np.arange(length)
            self.clean_corr = (self.img_corr == np.arange(length)).astype(int) * (self.txt_corr == np.arange(length)).astype(int)

    def __getitem__(self, index):
        # read lmdb
        with self.env.begin(write=False) as txn:
            # serialized_sample = txn.get(self.ids[index].encode())
            serialized_smp = txn.get(self.ids[index].encode())
            serialized_img = txn.get(self.ids[self.img_corr[index]].encode())
            serialized_txt = txn.get(self.ids[self.txt_corr[index]].encode())
        img_sample = pickle.loads(serialized_img, encoding='latin1')
        txt_sample = pickle.loads(serialized_txt, encoding='latin1')
        ori_sample = pickle.loads(serialized_smp, encoding='latin1')

        # image
        img_path = self.opts.img_path
        imgs = img_sample['imgs']
        if self.partition == 'train':
            imgIdx = np.random.choice(range(min(self.opts.maxImgs, len(imgs))))
        else:
            imgIdx = 0
        loader_path = [imgs[imgIdx]['id'][i] for i in range(4)]
        loader_path = os.path.join(*loader_path)
        path = os.path.join(img_path, self.partition, loader_path, imgs[imgIdx]['id'])
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        if len(self.aug)>0:
            aug = 0
            if 'english' in self.aug: # random choice between original text ("0"), en-de-en back translation ("1") or en-ru-en back translation ("2")
                tmp = np.random.choice([1,2])
                aug = np.random.choice([aug,tmp]) # 50% chance using original, 25% en-de-en and 25% en-ru-en
            lang = []
            if 'de' in self.aug: lang.append(3)
            if 'ru' in self.aug: lang.append(4)
            if 'fr' in self.aug: lang.append(5)
            if len(lang)>0:
                tmp = np.random.choice(lang)
                aug = np.random.choice([aug,tmp]) # 50% chance using english (orig or back translation) and 50% chance of using different laguage (ko, de, ru, fr)
            # ingrs = torch.tensor(sample['recipe'][aug])
            tmp = torch.tensor(txt_sample['recipe'][aug])
            part_of_recipe = np.array(txt_sample['part_of_recipe'])
            ingrs = torch.tensor([tmp[0]],dtype=int)
            if 'textinput'in self.aug:
                textinputs = np.random.choice(['title','ingr','inst','title,ingr','title,inst','ingr,inst','title,ingr,inst'])
            else:
                textinputs = self.opts.textinputs
            if 'title' in textinputs:
                ingrs = torch.cat((ingrs,tmp[part_of_recipe[aug]==1]))
            if 'ingr' in textinputs:
                ingrs = torch.cat((ingrs,tmp[part_of_recipe[aug]==2]))
            if 'inst' in textinputs:
                ingrs = torch.cat((ingrs,tmp[part_of_recipe[aug]==3]))
            tmp = torch.zeros_like(tmp)
            tmp[:len(ingrs)] = ingrs
            ingrs = tmp
            if 'mask' in self.aug:
                # select randomly up to 25% of words to mask ( [MASK] = 103 )
                inds = np.random.choice((ingrs>0).sum().item(), np.random.choice(int((ingrs>0).sum().item()*0.25)))
                ingrs[inds] = 103
        else:
            tmp = torch.tensor(txt_sample['recipe'][self.language])
            part_of_recipe = np.array(txt_sample['part_of_recipe'])
            ingrs = torch.tensor([tmp[0]],dtype=int)
            if 'title' in self.opts.textinputs:
                ingrs = torch.cat((ingrs,tmp[part_of_recipe[int(self.language)]==1]))
            if 'ingr' in self.opts.textinputs:
                ingrs = torch.cat((ingrs,tmp[part_of_recipe[int(self.language)]==2]))
            if 'inst' in self.opts.textinputs:
                ingrs = torch.cat((ingrs,tmp[part_of_recipe[int(self.language)]==3]))
            if len(ingrs)==512:
                ingrs[-1] = 102
            else:
                ingrs = torch.cat((ingrs,torch.tensor([102])))
            tmp = torch.zeros_like(tmp)
            tmp[:len(ingrs)] = ingrs
            ingrs = tmp
        
        rec_id = self.ids[index]

        img_label = self.img_labels[self.img_corr[index]] if self.partition == 'train' else img_sample['classes']-2
        txt_label = self.txt_labels[self.txt_corr[index]] if self.partition == 'train' else txt_sample['classes']-2
        label = self.labels[index] if self.partition == 'train' else ori_sample['classes']-2
        
        if self.opts.clean_label:
            return [img, ingrs], [img_label, txt_label], index, self.clean_corr[index], [img_label, txt_label], rec_id
        else:
            return [img, ingrs], [label, label], index, self.clean_corr[index], [img_label, txt_label], rec_id

    def __len__(self):
        return len(self.ids)