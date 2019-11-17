from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import os
from random import sample
import torchvision.transforms as transforms
from natsort import natsorted
import time

class UCF101Dataset(Dataset):
    def __init__(self, data_path, num_frames_video, data, mode):
        super(UCF101Dataset, self).__init__()
        self.mode = mode if mode != 'val' else 'train'
        self.data_path = os.path.join(data_path, self.mode)
        self.num_frames_video = num_frames_video
        self.xs = data[0]
        if mode != 'test':
            self.ys = data[1]
        self.set_transforms()

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        sampled_video_name = self.xs[idx].replace('/', ',')
        video_frames = []
        # get all the video frames
        sample_video_frames = [image for image in os.listdir(self.data_path + os.sep) if sampled_video_name in image]
        # sample a time point from where we will sample 2 continual seconds, in 2.5 FPS it means 5 frames
        sample_start_point = sample(range(len(sample_video_frames) - 5), 1)[0]
        # keep so I can test which work better, sample continual seconds or random ones? or there is no difference
        # sample_video_frames = sample(sample_video_frames, self.num_frames_video)
        sample_video_frames = natsorted(sample_video_frames)
        sample_video_frames = sample_video_frames[sample_start_point : sample_start_point+5]
        for image in sample_video_frames:
            img = Image.open(os.path.join(self.data_path, image))
            img = img.convert('RGB')
            img = self.transform(img)
            video_frames.append(img)
        img_stack = torch.stack(video_frames)
        label = torch.from_numpy(np.asarray(int(self.ys[idx]))).long()
        return img_stack, label

    def set_transforms(self):
        if self.mode =='train':
            self.transform = transforms.Compose([transforms.Resize(256),  # this is set only because we are using Imagenet pretrain model.
                                            transforms.RandomCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.485, 0.456, 0.406),  #todo add a link where this values is taken from
                                                                 std=(0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([transforms.Resize(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                 std=(0.229, 0.224, 0.225))])














