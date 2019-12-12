import argparse
import skvideo #can save the new data as jpeg, can be used once I change it to h5pq
from tqdm import tqdm
skvideo.setFFmpegPath(r"C:\Users\Doron\Documents\ffmpeg-20191106-fc7b6d5-win64-static\bin")
import skvideo.io
import os
import cv2
from random import sample
import torchvision.transforms as transforms
from PIL import Image
import numpy as np



parser = argparse.ArgumentParser(description='UCF101 Action Recognition, LRCN architecture')
parser.add_argument('--root', default=r'C:\Users\Doron\Desktop\ObjectRecognition data\UCF101_row_data/', type=str, help='path to find the UCF101 row data')
parser.add_argument('--ucf_list_root', default=r'C:\Users\Doron\Google Drive\Object detection light\Data_UCF101\UCF101_video_list/', type=str, help='path to find the UCF101 list splitting the data to train and test')
parser.add_argument('--sampling_rate', default=10, type=int, help='how to sample the data')
parser.add_argument('--sampled_data_dir', default=r'C:\Users\Doron\Desktop\ObjectRecognition data\UCF101_sampled_data_video', type=str, help='the default path for the new sampled data')
parser.add_argument('--FPS', default=25, type=int, help='')
parser.add_argument('--num_frames_to_extract', default=10, type=int, help='')


class UCF101Preprocessing():  # todo module it!, make it prettier
    """"
    Create the sampled data,
    input - video,
    function - converted to numpy array, from each raw video we sampled X frames,
    Output: numpy array
    """
    def __init__(self):
        super(UCF101Preprocessing, self).__init__()
        self.root = arg.root
        self.ucf_list_root = arg.ucf_list_root
        self.sampling_rate = arg.sampling_rate
        self.sampled_data_dir = arg.sampled_data_dir
        self.fps = arg.FPS
        self.num_frames_to_extract = arg.num_frames_to_extract
        self.folder_dir = self.sampled_data_dir + '_' + str(self.sampling_rate) + '_' + str(self.num_frames_to_extract)


    def create_sampled_data(self):
        if not os.path.exists(self.folder_dir):
            os.makedirs(self.folder_dir)
            os.makedirs(os.path.join(self.folder_dir, 'test'))
            os.makedirs(os.path.join(self.folder_dir, 'train'))
        for file_name in os.listdir(self.ucf_list_root): #file_name todo also add transform
            if 'test' in file_name:
                transform = self.set_transforms('test')
                save_path = os.path.join(self.folder_dir, 'test')
            elif 'train' in file_name:
                save_path = os.path.join(self.folder_dir, 'train')
                transform = self.set_transforms('train')
            else:
                continue
            with open(self.ucf_list_root + file_name) as f:
                video_list = f.readlines()
            with tqdm(total=len(video_list)) as pbar:
                for video_name in video_list:
                    video_name = video_name.split(' ')[0].rstrip('\n')
                    video = cv2.VideoCapture(self.root + video_name)
                    video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
                    video_length = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    num_frame = int(video_length * self.fps)
                    if video_length < (self.num_frames_to_extract * self.sampling_rate):
                        sample_start_point = 0
                        self.sampling_rate = 2
                    else:
                        sample_start_point = sample(range(num_frame - (self.num_frames_to_extract * self.sampling_rate)), 1)[0] # in the cv2 we need to define the frame number in range 0.0-1.0
                    # seeting the video to start reading from the frame we want
                    image_array = []
                    sample_start_point = sample_start_point + self.sampling_rate
                    for frame in range(self.num_frames_to_extract):
                        # sample_start_point_norm = sample_start_point / num_frame
                        video.set(1, sample_start_point)
                        success, image = video.read()  # todo, see if it change/ have any impact on the data type
                        if not success:
                            print('Error')
                        else:
                            #create a transform function
                            image = Image.fromarray(image.astype('uint8'), 'RGB')
                            image = transform(image)
                            image_array += [np.uint8(image)]
                        sample_start_point = sample_start_point + self.sampling_rate
                    (h, w) = image.size[:2]
                    video.release()
                    save_video_path = os.path.join(save_path, video_name.split('/')[1])
                    output_video = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'MJPG'), 5, (w,h), True)
                    for frame in range(len(image_array)):
                        output_video.write(image_array[frame])
                    output_video.release()
                    # cv2.imwrite(os.path.join(save_path, video_name_save + "_frame%d.jpg" % frame_count), image)
                    cv2.destroyAllWindows()
                    pbar.update(1)

    def set_transforms(self, mode):
        if mode == 'train':
            transform = transforms.Compose(
                [transforms.Resize(256),  # this is set only because we are using Imagenet pretrain model.
                 transforms.RandomCrop(224),
                 transforms.RandomHorizontalFlip()
                 ])
        else:
            transform = transforms.Compose([transforms.Resize((224, 224))])
        return transform


#videodata = skvideo.io.vread("video_file_name")

if __name__ == '__main__':      #todo remove from OOP
    arg = parser.parse_args()
    data_processor = UCF101Preprocessing()
    data_processor.create_sampled_data()
