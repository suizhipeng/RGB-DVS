import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import My_dataset
from utils import create_dataset
from torch.utils.data import DataLoader
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('Options for training SqueezeNet in pytorch')
parser.add_argument('--batch_size', type=int, default=5, metavar='N', help='batch size of train')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train for')
parser.add_argument('--learning_rate', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--num_classes', type=int, default=2, help="how many classes training for")
parser.add_argument('--img_w', type=int, default=224, help="image width or height")
parser.add_argument('--channel', type=int, default=3, help="image channel number")
parser.add_argument('--device', type=str, default="cuda:0", help="gpu device no.")
parser.add_argument('--num_workers', type=int, default=0, help="gpu device no.")
parser.add_argument('--datasets_path', type=str, default="/media/HDD1/szp/dual_modality_datasets/05_13/demo_dataset", help="path to datasets")
parser.add_argument('--datasets_name', type=str, default="", help="name of datasets")
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--model_name', type=str, default="", help='path to model dict')
parser.add_argument('--log_dir', type=str, default="", help='path to logs')
parser.add_argument('--data', type=str, default="rgb_label", help='Which type of data to return')

args = parser.parse_args()

data_train, data_val, data_test = create_dataset(args.datasets_path, args.datasets_name, run_type="train", data=args.data)
print(len(data_train), len(data_val), len(data_test))  # 80 10 10
data_loader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
data_loader_val = DataLoader(data_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
data_loader_test = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
print(len(data_loader_train), len(data_loader_val), len(data_loader_test))  # 16 2 2

for img, label in data_loader_test:
    print(img.shape)  # torch.Size([5, 1080, 1440, 3])
    print(type(label))  # <class 'dict'>
    print(label['person_01'].shape)  # torch.Size([5, 1, 1, 1080, 1440])