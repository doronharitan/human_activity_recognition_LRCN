import argparse
from tqdm import tqdm
from utils_action_recognition import create_folder_dir_if_needed, \
    capture_and_sample_video, save_video_original_size_dict, save_setting_info
from tqdm import tnrange, tqdm_notebook #used when I run in colab/GCloud
import os

parser = argparse.ArgumentParser(description='UCF101 Action Recognition preprocessing data, LRCN architecture')
parser.add_argument('--row_data_dir', default=r'C:\Users\Doron\Desktop\ObjectRecognition data\UCF101_row_data', type=str,
                    help='path to find the UCF101 row data')
parser.add_argument('--ucf_list_dir',
                    default=r'C:\Users\Doron\Desktop\ObjectRecognition\Data_UCF101\UCF101_video_list',
                    type=str, help='path to find the UCF101 list splitting the data to train and test')
parser.add_argument('--sampling_rate', default=10, type=int, help='how to sample the data')
parser.add_argument('--ucf101_fps', default=25, type=int, help='FPS of the UCF101 dataset')
parser.add_argument('--num_frames_to_extract', default=5, type=int, help='The number of frames what would be extracted from each video')
parser.add_argument('--video_file_name', default='y2mate.com - cute_happy_baby_crawling_BkJ6FJ2jJEQ_360p.mp4', type=str,
                    help='the video file name we would process, if none the script would run on all of the video files in the folder')
parser.add_argument('--dataset', default='UCF101', type=str,
                    help='the dataset name. options = youtube, UCF101')


def main_procesing_data(args, folder_dir, sampled_video_file=None, processing_mode='main'):
    """"
       Create the sampled data video,
       input - video, full length.
       function - 1. Read the video using CV2
                  2. from each video X (args.sampling_rate) frames are sampled reducing the FPS by args.sampling_rate (for example from 25 to 2.5 FPS)
                  3. The function randomly set the start point where the new sampled video would be read from, and Y(args.num_frames_to_extract) continues frames are extracted.
                  4. if processing_mode == 'main' The Y continues frames are extracted and save to a new video if not the data in tensor tyoe mode is passed to the next function
       Output: videos in length of X frames
       """
    if args.dataset == 'UCF101':
        for file_name in os.listdir(args.ucf_list_dir):
            # ===== reading all of the row data from the first split of train and test =====
            if '1' in file_name:
                with open(os.path.join(args.ucf_list_dir, file_name)) as f:
                    video_list = f.readlines()
                with tqdm(total=len(video_list)) as pbar:
                # with tqdm_notebook(total=len(dataloader)) as pbar:
                    for video_name in video_list:
                        video_name = video_name.split(' ')[0].rstrip('\n')
                        capture_and_sample_video(args.row_data_dir, video_name, args.num_frames_to_extract,
                                                 args.sampling_rate, args.ucf101_fps, folder_dir,
                                                 args.ucf101_fps, processing_mode)
                        pbar.update(1)
            else:
                pass

    elif args.dataset == 'youtube':
        video_original_size_dict = {}
        if args.video_file_name is None and sampled_video_file is None:
            for file_name in os.listdir(args.row_data_dir):
                video_test, video_original_size = capture_and_sample_video(args.row_data_dir, file_name, 'all', args.sampling_rate, 'Not known',
                                         folder_dir, args.ucf101_fps, processing_mode)
                video_original_size_dict[file_name] = video_original_size
        else:
            file_name = args.video_file_name if sampled_video_file is None else sampled_video_file[0]
            video_test, video_original_size = capture_and_sample_video(args.row_data_dir, file_name, 'all', args.sampling_rate, 'Not known',
                                     folder_dir, args.ucf101_fps, processing_mode)
            video_original_size_dict[file_name] = video_original_size
        save_video_original_size_dict(video_original_size_dict, save_path)
        if processing_mode == 'live':
            return video_test, video_original_size


if __name__ == '__main__':
    args = parser.parse_args()
    global_dir = os.path.normpath(args.row_data_dir + os.sep + os.pardir)
    folder_name = '{}_sampled_data_video_sampling_rate_{}_num frames extracted_{}'.format(args.dataset, args.sampling_rate,
        args.num_frames_to_extract)
    folder_dir = os.path.join(global_dir, folder_name)
    create_folder_dir_if_needed(folder_dir)
    save_setting_info(args, "cpu", folder_dir)
    main_procesing_data(args, folder_dir)
