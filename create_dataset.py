from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms
import skvideo
import skvideo.io
from torch.utils.data.sampler import Sampler
from random import sample
from utils_action_recognition import print_dataset_type_error, set_transforms

class UCF101Dataset(Dataset):
    def __init__(self, data_path, data, mode, dataset='UCF101'):
        super(UCF101Dataset, self).__init__()
        self.dataset = dataset
        if self.dataset == 'UCF101':
            self.labels = data[1]
        self.data_path = data_path
        self.images = data[0]
        self.transform = set_transforms(mode)

    # ====== Override to give PyTorch size of dataset ======
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.dataset == 'UCF101':
            sampled_video_name = self.images[idx].split('/')[1] +'.avi'
        elif self.dataset == 'youtube':
            sampled_video_name = self.images[idx]
        else:
           print_dataset_type_error()
        # ====== extract numpy array from the video and sample it so we will have an array with lower FPS rate =======
        video_frames = skvideo.io.vread(os.path.join(self.data_path, sampled_video_name))
        video_frames_array = []
        for image in video_frames:
            img = Image.fromarray(image.astype('uint8'), 'RGB')
            img = self.transform(img)
            video_frames_array.append(img)
        img_stack = torch.stack(video_frames_array)
        if self.dataset == 'UCF101':
            label = torch.from_numpy(np.asarray(int(self.labels[idx]))).long()
            return img_stack, label, idx
        else:
            return img_stack



class UCF101DatasetSampler(Sampler):
    def __init__(self, data, batch_size):
        self.num_samples = len(data)
        self.classes_that_were_sampled = []
        self.data_labels = data.labels
        self.batch_size = batch_size


    def __iter__(self):
        idx_list = []
        for i in range(self.batch_size):
            idx_image_sample = sample(range(self.num_samples), 1)[0]
            label_sample = self.data_labels[idx_image_sample]
            while label_sample in self.classes_that_were_sampled:
                idx_image_sample = sample(range(self.num_samples), 1)[0]
                label_sample = self.data_labels[idx_image_sample]
            self.classes_that_were_sampled += [label_sample]
            idx_list += [idx_image_sample]
        return iter(idx_list)

    def __len__(self):
        return self.num_samples










