from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import os
from random import sample
import torchvision.transforms as transforms
# from natsort import natsorted
import skvideo
skvideo.setFFmpegPath(r"C:\Users\Doron\Documents\ffmpeg-20191106-fc7b6d5-win64-static\bin")
import skvideo.io


class UCF101Dataset(Dataset):
    def __init__(self, data_path,  num_frames_video, data, mode):
        super(UCF101Dataset, self).__init__()
        # self.mode = mode #if mode != 'val' else 'train'
        self.data_path = os.path.join(data_path, mode if mode != 'val' else 'train')
        self.num_frames_video = num_frames_video
        self.xs = data[0]
        if mode != 'test':
            self.ys = data[1]
        self.set_transforms()

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        sampled_video_name = self.xs[idx].split('/')[1] +'.avi'
        video_frames = skvideo.io.vread(os.path.join(self.data_path, sampled_video_name))
        #extract numpy array from the video and sample it so we will have an arrry with lower FPS rate
        video_frames_array = []
        # get all the video frames
        # sample_video_frames = [image for image in os.listdir(self.data_path + os.sep) if sampled_video_name in image]
        # sample a time point from where we will sample 2 continual seconds, in 2.5 FPS it means 5 frames
        # sample_start_point = sample(range(videodata.shape[0]-5), 1)[0]
        # keep so I can test which work better, sample continual seconds or random ones? or there is no difference
        # sample_video_frames = sample(sample_video_frames, self.num_frames_video)
        # sample_video_frames = natsorted(sample_video_frames)
        # sample_video_frames = videodata[sample_start_point : sample_start_point+5]
        for image in video_frames: #do I need to do it on each frame? pr can I do it on all images at once?
            img = Image.fromarray(image.astype('uint8'), 'RGB')
            # img = Image.open(os.path.join(self.data_path, image))
            # img = img.convert('RGB')
            img = self.transform(img)
            video_frames_array.append(img)
        img_stack = torch.stack(video_frames_array)
        label = torch.from_numpy(np.asarray(int(self.ys[idx]))).long()
        return img_stack, label

    def set_transforms(self):
        # if self.mode =='train':
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.485, 0.456, 0.406),  #todo add a link where this values is taken from
                                                                 std=(0.229, 0.224, 0.225))])














