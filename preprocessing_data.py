import argparse
import skvideo  # can save the new data as jpeg, can be used once I change it to h5pq
from tqdm import tqdm
from tqdm import tnrange, tqdm_notebook #used when I run in colab/GCloud
from utils_action_recognition import create_folder_dir_if_needed, set_transform_and_save_path, create_new_video,\
    setting_sample_rate
skvideo.setFFmpegPath(r"C:\Users\Doron\Documents\ffmpeg-20191106-fc7b6d5-win64-static\bin")
import skvideo.io
import os
import cv2
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description='UCF101 Action Recognition, LRCN architecture')
parser.add_argument('--root', default=r'C:\Users\Doron\Desktop\ObjectRecognition data\UCF101_row_data/', type=str,
                    help='path to find the UCF101 row data')
parser.add_argument('--ucf_list_root',
                    default=r'C:\Users\Doron\Google Drive\Object detection light\Data_UCF101\UCF101_video_list/',
                    type=str, help='path to find the UCF101 list splitting the data to train and test')
parser.add_argument('--sampling_rate', default=10, type=int, help='how to sample the data')
parser.add_argument('--sampled_data_dir',
                    default=r'C:\Users\Doron\Desktop\ObjectRecognition data\UCF101_sampled_data_video', type=str,
                    help='the default path for the new sampled data')
parser.add_argument('--FPS', default=25, type=int, help='')
parser.add_argument('--num_frames_to_extract', default=10, type=int, help='')


def main():
    """"
       Create the sampled data,
       input - video, full length
       function - converted to numpy array, from each video we sampled X frames,
       Output: videos in length of X frames
       """

    arg = parser.parse_args()
    folder_dir = arg.sampled_data_dir + '_' + str(arg.sampling_rate) + '_' + str(arg.num_frames_to_extract)
    # ====== create the folder dir where the new videos would be save at ======
    create_folder_dir_if_needed(folder_dir, mode='preprocessing_data')
    for file_name in os.listdir(arg.ucf_list_root):
        # ====== set the transform of the image according to test and train data ======
        transform, save_path = set_transform_and_save_path(folder_dir, file_name)
        with open(arg.ucf_list_root + file_name) as f:
            video_list = f.readlines()
        with tqdm(total=len(video_list)) as pbar:
        # with tqdm_notebook(total=len(dataloader)) as pbar:
            for video_name in video_list:
                # ====== extract video name ======
                video_name = video_name.split(' ')[0].rstrip('\n')
                # ====== read video file using CV2  and randomly set the start point where we should start reading the video ======
                video = cv2.VideoCapture(arg.root + video_name)
                sample_start_point, sampling_rate = setting_sample_rate(arg.num_frames_to_extract, arg.sampling_rate, video, arg.fps)
                # ====== setting the video to start reading from the frame we want ======
                image_array = []
                sample_start_point = sample_start_point + sampling_rate
                for frame in range(arg.num_frames_to_extract):
                    video.set(1, sample_start_point)
                    success, image = video.read()
                    if not success:
                        print('Error')
                    else:
                        image = Image.fromarray(image.astype('uint8'), 'RGB')
                        image = transform(image)
                        image_array += [np.uint8(image)]
                    sample_start_point = sample_start_point + sampling_rate
                video.release()
                create_new_video(save_path, video_name, image_array)
                pbar.update(1)


if __name__ == '__main__':
   main()
