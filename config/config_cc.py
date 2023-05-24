import os
import argparse
from config.config_utils import str2bool

parser = argparse.ArgumentParser(description='dorefa-net implementation')

# DATASET PARAMETERS
parser.add_argument("--data_name", type=str, default="cc")
parser.add_argument('--data_dir', type=str, default='/path/to/data')
parser.add_argument('--data_scale', type=str, default='toy')
parser.add_argument('--class_num', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=16)

# BACKBONE PARAMETERS
parser.add_argument('--views', nargs='+', help='<Required> Quantization bits', default=['Img', 'Txt', 'Audio', '3D', 'Video'])
parser.add_argument('--embDim', default=1024, type=int)          # Dim of FoodSpace

## TEXT BACKBONE
parser.add_argument("--w2vInit", type=str2bool, nargs='?', const=True, default=True, help="Initialize word embeddings with w2v model?")
parser.add_argument('--maxSeqlen', default=20, type=int)         # Used when building LMDB
parser.add_argument('--maxInsts', default=20, type=int)          # Used when building LMDB
parser.add_argument('--maxImgs', default=5, type=int)            # Used when building LMDB
parser.add_argument('--textmodel', default='mBERT_fulltxt', type=str)  
parser.add_argument('--textinputs', default='title,ingr,inst', type=str)  # Pieces of recipe to use. Only valid if textmodel='AWE'
parser.add_argument("--textAug", type=str, default='', help="Use text augmentation: 'english', 'de', 'ru' and/or 'fr'. 'english' uses back-translation from 'de' and 'ru'")
parser.add_argument('--BERT_layers', default=2, type=int)
parser.add_argument('--BERT_heads', default=2, type=int) 

## IMAGE BACKBONE
parser.add_argument('--img_path', default='/path/to/image')
parser.add_argument('--valfreq', default=1,type=int)
parser.add_argument("--w2vTrain", type=str2bool, nargs='?', const=True, default=True, help="Allow word embeddings to be trained?")
parser.add_argument("--freeVision", type=str2bool, nargs='?', const=True, default=True, help="Train vision parameters?")
parser.add_argument("--freeHeads", type=str2bool, nargs='?', const=True, default=True, help="Train model embedding heads?")
parser.add_argument("--freeWordEmb", type=str2bool, nargs='?', const=True, default=True, help="Train word embedding parameters?")
parser.add_argument("--freeText", type=str2bool, nargs='?', const=True, default=True, help="Train text encoder parameters?")

# TRAINING PARAMETERS
parser.add_argument('--root_dir', type=str, default='/path/to/root')
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--eval_batch_size', type=int, default=128)
parser.add_argument('--tau', type=float, default=1.)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--medr', default=1000, type=int)
parser.add_argument('--embtype', default='image', type=str) # [image|recipe] query type

## TWO STAGE
### WARMUP PARAMETERS
parser.add_argument('--load_warmup_models', type=int, default=-1) # if -1 -> False else -> load epoch
parser.add_argument('--load_warmup_path', type=str, default='') 
parser.add_argument('--warmup_epochs', type=int, default=0) # 20
parser.add_argument('--e_epoch1', type=int, default=20) # 3
parser.add_argument('--e_epoch2', type=int, default=20) # 4
parser.add_argument('--warmup_ms', type=list, default=[10])
parser.add_argument('--warmup_lr1', type=float, default=1e-4) # 5e-5
parser.add_argument('--warmup_lr2', type=float, default=5e-5) # 5e-5
parser.add_argument('--warmup_wd1', type=float, default=0) # 0.0005
parser.add_argument('--warmup_wd2', type=float, default=0)
parser.add_argument('--warmup_gamma', type=float, default=0.2)
parser.add_argument('--logit_adjust', type=bool, default=False)

### TRAIN PARAMETERS
parser.add_argument('--train_epochs', type=int, default=100) # 20
parser.add_argument('--train_ms', type=list, default=[10, 20])
parser.add_argument('--train_lr', type=float, default=5e-5)
parser.add_argument('--train_wd', type=float, default=0)
parser.add_argument('--train_gamma', type=float, default=0.1)
parser.add_argument('--weight_epoch', type=int, default=20) # 5

### EVALUATION PARAMETERS
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--model_epoch', type=int, default=0)

parser.add_argument('--label_denoise', type=str, default='transition_matrix') # elr/ transition_matrix
parser.add_argument('--use_cdc_reweight', type=bool, default=False)
parser.add_argument('--cdc_weight_dir', type=str, default='')

parser.add_argument('--label_modeling', type=str, default='cdl')

parser.add_argument('--gamma_1', default=2, type=float)
parser.add_argument('--gamma_2', default=2, type=float)

# RUNNING PARAMETERS
parser.add_argument('--gpu_ids', type=list, default=[4]) # Use data parallel when including MULTIPLE gpu_ids and set the first as the MAIN gpu
parser.add_argument('--seed', type=int, default=1234) 
parser.add_argument('--log_name', type=str, default='')

args = parser.parse_args()

print(args)

