from random import sample

import torch
import os
from utils_action_recognition import set_project_folder_dir, \
    save_setting_info,test_model_continues_movie_youtube, load_test_data, print_error_preprocessing_movie_mode,\
    load_and_extract_video_original_size
from LRCN.create_dataset import UCF101Dataset
from torch.utils.data import DataLoader
from LRCN.lrcn_model import ConvLstm
from LRCN.train import parser
from preprocessing_data import main_procesing_data

parser.add_argument('--model_dir', default=r'C:\Users\Doron\Desktop\ObjectRecognition\20191218-214903\Saved_model_checkpoints', type=str, help='The dir of the model we want to test')
parser.add_argument('--model_name', default='epoch_30.pth.tar', type=str, help='the name for the model we want to test on')
parser.add_argument('--video_file_name', default=None, type=str,
                    help='the video file name we would process, if none the script would run on all of the video files in the folder')
parser.add_argument('--preprocessing_movie_mode', default='preprocessed', type=str,
                    help='should we preprocess the video on the go (live) or using the preprocessed script (default:live, options: live/preprocessed)')
parser.add_argument('--dataset', default='youtube', type=str,
                    help='the dataset name. options = youtube, UCF101')
parser.add_argument('--sampling_rate', default=10, type=int, help='what was the sampling rate of the ucf-101 train dataset')
parser.add_argument('--ucf101_fps', default=25, type=int, help='FPS of the UCF101 dataset')
parser.add_argument('--row_data_dir', default=r'C:\Users\Doron\Desktop\ObjectRecognition data\youtube_videos', type=str,
                    help='path to find the UCF101 row data')

def main():
    # ====== set the run settings ======
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    folder_dir = set_project_folder_dir(args.open_new_folder, args.model_dir, use_model_folder_dir=True, mode='test_youtube_movie')
    print('The setting of the run are:\n{}\n' .format(args))
    print('The training would take place on {}\n'.format(device))
    print('The project directory is {}' .format(folder_dir))
    save_setting_info(args, device, folder_dir)
    label_decoder_dict = load_test_data(args.model_dir, mode='load_label_decoder_dict')
    print('Loading model...')
    num_class = len(label_decoder_dict) if args.number_of_classes is None else args.number_of_classes
    model = ConvLstm(args.latent_dim, args.hidden_size, args.lstm_layers, args.bidirectional, num_class)
    model = model.to(device)
    checkpoint = torch.load(os.path.join(args.model_dir, args.model_name))
    model.load_state_dict(checkpoint['model_state_dict'])
    # ====== inferance_mode ======
    if args.video_file_name is None and args.preprocessing_movie_mode != 'live':
        test_videos_names = [file_name for file_name in os.listdir(args.sampled_data_dir) if '.avi' in file_name]
    elif args.video_file_name is None:
        test_videos_names = [file_name for file_name in os.listdir(args.row_data_dir)]
    else:
        test_videos_names = [args.video_file_name]
    if args.preprocessing_movie_mode == 'preprocessed':
        dataset = UCF101Dataset(args.sampled_data_dir, [test_videos_names], mode='test', dataset='youtube')
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        video_original_size_dict = load_and_extract_video_original_size(args.sampled_data_dir)
        test_model_continues_movie_youtube(model, dataloader, device, folder_dir, label_decoder_dict, args.batch_size,
                                           args.preprocessing_movie_mode, video_original_size=video_original_size_dict)
    elif args.preprocessing_movie_mode == 'live':
        movie_name_to_test = sample(test_videos_names, 1)
        test_movie, video_original_size = main_procesing_data(args, folder_dir, sampled_video_file=movie_name_to_test,
                                                              processing_mode='live')
        test_model_continues_movie_youtube(model, torch.stack(test_movie), device, folder_dir, label_decoder_dict,
                                           args.batch_size, args.preprocessing_movie_mode,
                                           video_original_size=video_original_size)
    else:
        print_error_preprocessing_movie_mode()

if __name__ == '__main__':
    main()

