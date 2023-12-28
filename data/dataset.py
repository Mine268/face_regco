import os
from PIL import Image
from imgaug import  augmenters as iaa
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys
import random


class FaceDataset(object):
    def __init__(self, root_directory, train_ratio=0.9, input_shape=(3, 112, 112)):
        train_dataset = {}
        test_dataset = {}
        self.root_directory = root_directory
        self.train_len, self.test_len = 0, 0
        self.input_shape = input_shape

        # 设置随机数种子，确保结果可重现
        random.seed(42)

        # 遍历根目录下的每个人种文件夹
        for race in os.listdir(root_directory):
            race_path = os.path.join(root_directory, race)
            
            if os.path.isdir(race_path):
                people = [person for person in os.listdir(race_path) if os.path.isdir(os.path.join(race_path, person))]
                
                # 随机打乱人的列表
                random.shuffle(people)

                # 根据指定比例分割人列表
                split_index = int(len(people) * train_ratio)
                train_people = people[:split_index]
                test_people = people[split_index:]

                # 添加人及其图片到相应的字典中
                train_dataset[race] = {person: os.listdir(os.path.join(race_path, person)) for person in train_people}
                test_dataset[race] = {person: os.listdir(os.path.join(race_path, person)) for person in test_people}
        
        # 将字典的元素合并为列表
        self.train_dataset = []
        for race, persons in train_dataset.items():
            for person, images in persons.items():
                self.train_dataset.append([f"{self.root_directory}/{race}/{person}/{image}" for image in images])
        self.test_dataset = []
        for race, persons in test_dataset.items():
            for person, images in persons.items():
                self.test_dataset.append([f"{self.root_directory}/{race}/{person}/{image}" for image in images])
        
        self.train_transform = T.Compose([
                T.Resize(self.input_shape[1:]),
                T.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.test_transform = T.Compose([
                T.Resize(self.input_shape[1:]),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.dataset = None
        self.transform = None
        self.train()
        
    def train(self):
        self.dataset = self.train_dataset
        self.transform = self.train_transform
    
    def test(self):
        self.dataset = self.test_dataset
        self.transform = self.test_transform
                    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        positive_list = self.dataset[idx]
        positive_1, positive_2 = random.sample(positive_list, 2)
        
        n_idx = random.randint(0, len(self.dataset) - 2)
        if n_idx >= idx:
            n_idx += 1
        negative = self.dataset[n_idx]
        negative = random.sample(negative, 1)[0]
        
        positive_1 = self.transform(Image.open(positive_1).convert('RGB'))
        positive_2 = self.transform(Image.open(positive_2).convert('RGB'))
        negative = self.transform(Image.open(negative).convert('RGB'))
        
        return torch.concat([positive_1[None, ...], positive_2[None, ...], negative[None, ...]], dim=0)

# 改进的 dataset
class Dataset_181(object):
    def __init__(self, imgs, train: bool = True, input_shape=(3, 112, 112)):
        self.train = train
        self.input_shape = input_shape
        self.imgs = imgs

        if self.train:
            self.transforms = T.Compose([
                T.Resize(self.input_shape[1:]),
                # T.RandomCrop(self.input_shape[1:]),
                T.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(self.input_shape[1:]),
                # T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split('/')[-1]
        label = int(splits.split('_')[0])
        img_path = sample
        data = Image.open(img_path).convert('RGB')
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def setup_datasets(root: str, ratio: float = 0.9):
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # 按照人进行训练图像划分
        train_dict = {}
        for img_path in imgs:
            splits = img_path.split('/')[-1]
            label = int(splits.split('_')[0])
            train_dict.setdefault(label, []).append(img_path)
        train_imgs = [item for sublist in train_dict.values() for item in sublist]
        random.shuffle(train_imgs)

        train_imgs = random.sample(train_imgs, int(len(train_imgs) * ratio))
        test_imgs = list(set(imgs) - set(train_imgs))

        train_dataset = Dataset_181(imgs=train_imgs, train=True, input_shape=(3, 112, 112))
        test_dataset = Dataset_181(imgs=test_imgs, train=False, input_shape=(3, 112, 112))

        return train_dataset, test_dataset

