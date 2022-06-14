import os
import torch.utils.data as data
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import cv2
import numpy

class TestconditionDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = opt.dataroot
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   
        self.A_size = len(self.A_paths) 
        btoA = self.opt.direction == 'BtoA'
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc       
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc   

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size] 
        A_img = Image.open(A_path).convert('RGB')

        '''Randomly choose target stroke width level'''
        stroke_bank = list(range(1,6))
        stroke_width = numpy.random.choice(stroke_bank, size=1, replace=False)[0]

        '''Appoint certain stroke width level'''
        # stroke_width = 2

        print("target stroke width level:", stroke_width)

        self.transform_A = get_transform(self.opt, grayscale=(self.input_nc == 1))
        A = self.transform_A(A_img)

        return {'A': A, 'A_paths': A_path, 'B': A, 'B_paths': A_path, 'stroke_width':stroke_width}

    def __len__(self):
        return self.A_size
