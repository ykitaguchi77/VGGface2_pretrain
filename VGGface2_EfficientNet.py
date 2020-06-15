import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset

'''
https://betashort-lab.com/%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B5%E3%82%A4%E3%82%A8%E3%83%B3%E3%82%B9/%E3%83%87%E3%82%A3%E3%83%BC%E3%83%97%E3%83%A9%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0/pytorch%E3%81%AEdatasets%E3%81%A7%E7%94%BB%E5%83%8F%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88%E3%82%92%E4%BD%9C%E3%82%8B/
'''


class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, path, transform1 = None, transform2 = None, train = True):        
        # csvデータの読み出し
        image_id = []
        image_path = []
        with open('C:\\Datasets\\VGGface2\\test.csv', 'r') as f:
            for row in f:
                id = row.split('/')[0]
                path = row
                image_id.append(id)
                image_path.append(path)
    


        dataframe = []
        with open('C:\\Datasets\\VGGface2\\train', 'r') as f:
            for line in f:
                row = line.strip().split(',')
                dataframe.append(row)
        self.dataframe = dataframe

        
        self.transform1 = transform1
        self.transform2 = transform2
        self.train = train

        self.labelset = torchvision.datasets.CIFAR10(root = path, train = self.train, download = True)
        self.dataset = torchvision.datasets.CIFAR10(root = path, train = self.train, download = True)

        self.datanum = len(dataset)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_label = self.labelset[idx][0]
        out_data = self.dataset[idx][0]

        if self.transform1:
            out_label = self.transform1(out_label)

        if self.transform2:
            out_data = self.transform2(out_data)

        return out_data, out_label