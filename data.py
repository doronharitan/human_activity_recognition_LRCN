from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import os
from random import sample
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from natsort import natsorted
import time
class UCF101Dataset(Dataset):
    def __init__(self, data_path, num_frames_video, data, mode):
        super(UCF101Dataset, self).__init__()
        self.data_path = data_path + os.sep + mode #Todo: switch and ise this code: os.path.join(data_dir, x), cleaner
        self.mode = mode
        self.num_frames_video = num_frames_video
        self.xs = data[0]
        if mode != 'test':
            self.ys = data[1]
        self.set_transforms()

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        video_frames = []
        # need to open all of the img of the relevant video, and stack them
        sampled_video_name = self.xs[idx].replace('/', ',')
        # get all the images
        sample_video_frames = [image for image in os.listdir(self.data_path + os.sep) if sampled_video_name in image]
        #Todo replace it with globe import glob
        # for name in glob.glob('dir/*'): or for name in glob.glob('dir/*[0-9].*'):
        #     print name
        # random sampling of the frames we take from each video
        sample_video_frames = sample(sample_video_frames, self.num_frames_video)
        sample_video_frames = natsorted(sample_video_frames)
        for image in sample_video_frames:
            img = Image.open(self.data_path + os.sep + image) #todo change is to os.path.join. it looks better
            img = img.convert('RGB')
            img = self.transform(img)
            # Convert image and label to torch tensors
            video_frames.append(img)
        # img_stack = video_frames[1] # erase todo erase
        img_stack = torch.stack(video_frames)
        label = torch.from_numpy(np.asarray(int(self.ys[idx])))
        return img_stack, label

    def set_transforms(self):
        if self.mode =='train':
            self.transform = transforms.Compose([transforms.Resize(256),  # this is set only because we are using Imagenet pretrain model.
                                            transforms.RandomCrop(224),   # ToDo: why we need to do augmentation?
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                 std=(0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([transforms.Resize(224),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                 std=(0.229, 0.224, 0.225))])


class SplitData():
    def __init__(self):
        super(SplitData, self).__init__()
        self.xs, self.xs_test = [], []
        self.video_list_train, self.video_list_test = [], []
        self.ys = []
        # setting the train data files as a list so the not overpower the system

    def split(self, ucf_list_root, seed, smaller_dataset=False):
        self.get_video_list(ucf_list_root)
        self.smaller_dataset = smaller_dataset
        self.get_data('train')
        self.get_data('test')
        if smaller_dataset:
            self.split_to_train_val(seed, smaller_dataset=True)
        train_data, val_data = self.split_to_train_val(seed)
        return train_data, val_data, [self.xs_test], self.label_encoder_dict

    def get_data(self, mode): #איןthink on how to change it, make it better.
        if mode == 'train':
            video_list = self.video_list_train
            list = self.xs
        else:
            video_list = self.video_list_test
            list = self.xs_test
        for video in video_list:
            if mode == 'train':   # Todo ask Alex if it is faster or slower when it is here? or should it be outside (time wize)
                video, label = video.split(' ')
                self.ys.append(int(label.split('\n')[0]))
            list.append(video.split('.')[0])

    def get_video_list(self, ucf_list_root):   # todo- decide if I want to use append or += for a list
        for file in os.listdir(ucf_list_root):
            if 'train' in file:
                with open(ucf_list_root + file) as f:
                    self.video_list_train += f.readlines()
            elif 'classInd' in file:
                with open(ucf_list_root + file) as f:
                    label_encoder = f.readlines()
                self.label_encoder_dict = {x.split(' ')[0] : x.split(' ')[1].rstrip('\n') for x in label_encoder} #Todo use the strip the ending in the correct locations
            else:
                with open(ucf_list_root + file) as f:
                    self.video_list_test += f.readlines()

    def split_to_train_val(self, seed, smaller_dataset=False):
        X_train, X_Val, y_train, y_val =  train_test_split(self.xs, self.ys, test_size = 0.2, random_state = seed)
        if smaller_dataset:
            _, self.xs, _, self.ys = train_test_split(self.xs, self.ys, test_size=0.005, random_state=seed)
            _ , self.xs_test = train_test_split(self.xs_test, test_size=0.1)
        else:
            return [X_train, y_train], [X_Val, y_val]












