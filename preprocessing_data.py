
import argparse
import skvideo #can save the new data as jpeg, can be used once I change it to h5pq
import cv2
import torch.utils.data
from tqdm import tqdm

skvideo.setFFmpegPath(r"C:\Users\Doron\Documents\ffmpeg-20191106-fc7b6d5-win64-static\bin")
import skvideo.io
import os
from sys import getsizeof



parser = argparse.ArgumentParser(description='UCF101 Action Recognition, LRCN architecture')
parser.add_argument('--root', default=r'C:\Users\Doron\PycharmProjects\ObjectRecognition\Data_UCF101\UCF101_row_data/', type=str, help='path to find the UCF101 row data')
parser.add_argument('--ucf_list_root', default=r'C:\Users\Doron\PycharmProjects\ObjectRecognition\Data_UCF101\UCF101_video_list/', type=str, help='path to find the UCF101 list splitting the data to train and test')
parser.add_argument('--sampling_rate', default=10, type=str, help='how to sample the data')
parser.add_argument('--sampled_data_dir', default=r'C:\Users\Doron\PycharmProjects\ObjectRecognition\Data_UCF101\UCF101_sampled_data', type=str, help='the default path for the new sampled data')

class UCF101Preprocessing():
    """"
    Create the sampled data,
    input - video,
    function - converted to numpy array, from each raw video we sampled X frames,
    Output: numpy array
    """
    def __init__(self, root, ucf_list_root, sampling_rate, sampled_data_dir):
        super(UCF101Preprocessing, self).__init__()
        self.root = root
        self.ucf_list_root = ucf_list_root
        self.sampling_rate = sampling_rate
        self.sampled_data_dir = sampled_data_dir
        self.folder_dir = self.sampled_data_dir +  '_' + str(self.sampling_rate)


    def create_sampled_data(self):
        if not os.path.exists(self.folder_dir):
            os.makedirs(self.folder_dir)
            os.makedirs(self.folder_dir + os.sep + 'test')
            os.makedirs(self.folder_dir + os.sep + 'train')

        for file in os.listdir(self.ucf_list_root):
            if 'test' in file:
                save_path = self.folder_dir + os.sep + 'test' + os.sep
            elif 'train' in file:
                save_path = self.folder_dir + os.sep + 'train' + os.sep
            else:
                continue
            with open(self.ucf_list_root + file) as f:
                video_list = f.readlines()
            with tqdm(total=len(video_list)) as pbar:
                for video_name in video_list:
                    video_name_save = video_name.split('.')[0].replace('/', ',')
                    video_data = cv2.VideoCapture(self.root + video_name.split(' ')[0])
                    frame_count = 0
                    success, image = video_data.read()
                    while success:
                        if frame_count%self.sampling_rate==0:
                            cv2.imwrite(save_path + video_name_save +"_frame%d.jpg" % frame_count, image)# save frame as JPEG file
                        success, image = video_data.read()
                        frame_count += 1
                    pbar.update(1)

                # videodata = skvideo.io.vread(self.root + video_name, inputdict={'-r': 12})


if __name__ == '__main__':
    arg = parser.parse_args()
    data_processor = UCF101Preprocessing(arg.root, arg.ucf_list_root, arg.sampling_rate, arg.sampled_data_dir)
    data_processor.create_sampled_data()
