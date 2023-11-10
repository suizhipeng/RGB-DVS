from operator import add
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import glob
import torch


class My_dataset(Dataset):
    def __init__(self, datasets_path, datasets_name, data):
        super().__init__()
        self.datasets_path = datasets_path
        self.datasets_name = datasets_name
        self.data = data

        self.events = []
        self.images = []
        self.ev_labels = []
        self.im_labels = []

        if self.data == 'event':
            print()
        
        elif self.data == 'rgb':
            self.images.extend(glob.glob(os.path.join(datasets_path, datasets_name, 'raw_frame/*.png')))
            self.images.sort()
            self.length = len(self.images)
        
        elif self.data == 'event_label':
            print()
        
        elif self.data == 'rgb_label':
            self.images.extend(glob.glob(os.path.join(datasets_path, datasets_name, 'raw_frame/*.png')))
            self.images.sort()
            self.im_labels.extend(glob.glob(os.path.join(datasets_path, datasets_name, 'manual_mask/*.npy')))
            self.im_labels.sort()
            assert len(self.images) == len(self.im_labels)
            self.length = len(self.images)

        elif self.data == 'event_rgb':
            self.events.extend()
            self.events.sort()

            self.images.extend()
            self.images.sort()

            assert len(self.events) == len(self.images)
            self.length = len(self.events)

        elif self.data == 'event_rgb_label':
            self.events.extend()
            self.events.sort()

            self.images.extend()
            self.images.sort()

            self.ev_labels.extend()
            self.ev_labels.sort()

            self.im_labels.extend()
            self.im_labels.sort()

            assert len(self.events) == len(self.images)
            assert len(self.events) == len(self.ev_labels)
            assert len(self.images) == len(self.im_labels)
            self.length = len(self.events)
        
        else:
            print('Wrong data type')
            exit()


    def __getitem__(self, index):
        if self.data == 'event':
            event = np.load(self.events[index])
            return event
        elif self.data == 'rgb':
            image = cv2.imread(self.images[index])
            return image
        elif self.data == 'event_label':
            event = np.load(self.events[index])
            label = np.load(self.ev_labels[index])
            return event, label
        elif self.data == 'rgb_label':
            image = cv2.imread(self.images[index])
            label = np.load(self.im_labels[index], allow_pickle=True).item()
            return image, label
        elif self.data == 'event_rgb':
            event = np.load(self.events[index])
            image = cv2.imread(self.images[index])
            return event, image
        elif self.data == 'event_rgb_label':
            event = np.load(self.events[index])
            image = cv2.imread(self.images[index])
            ev_label = np.load(self.ev_labels[index])
            im_label = np.load(self.im_labels[index])
            return event, image, ev_label, im_label
        else:
            print('Error!')
            exit()

    def __len__(self):
        return self.length


def create_dataset(datasets_path, datasets_name, run_type="train", data='rgb'):
    train_dataset = My_dataset(datasets_path, datasets_name, data)
    if run_type == "test":
        return train_dataset
    elif run_type == 'train':
        val_ratio = 0.1
        test_ratio = 0.1
        val_num = int(len(train_dataset) * val_ratio)
        test_num = int(len(train_dataset) * test_ratio)
        train_num = len(train_dataset) - val_num - test_num
        lengths = [train_num, val_num, test_num]
        return torch.utils.data.dataset.random_split(train_dataset, lengths)
    else:
        print('Wrong run_type!!!')
        exit()
