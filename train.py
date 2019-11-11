
import torch
import argparse
import os
import torchvision
from torch.utils.data import DataLoader
from data import *

parser = argparse.ArgumentParser(description='UCF101 Action Recognition, LRCN architecture')
parser.add_argument('--epochs', default=500, type=int, help='number of total epochs')
parser.add_argument('--batch-size', default=16, type=int, help='mini-batch size (default: 32)')
parser.add_argument('--lr', default=5e-4, type=float, help='initial learning rate')
parser.add_argument('--num_workers', default=1, type=int, help='initial num_workers, the number of processes that generate batches in parallel')
parser.add_argument('--sampled_data_path', default=r'C:\Users\Doron\Desktop\ObjectRecognition\UCF101_sampled_data_10', type=str
                    , help='the path for the sampled row data')
parser.add_argument('--ucf_list_root', default=r'C:\Users\Doron\PycharmProjects\ObjectRecognition\Data_UCF101\UCF101_video_list/', type=str, help='path to find the UCF101 list splitting the data to train and test')
parser.add_argument('--num_frames_video', default=5, type=int, help='the number of frames that would be taken from each video')
parser.add_argument('--seed', default=42, type=int, help='?')




def main():
    args = parser.parse_args()
    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Todo: 1.open a new folder where all of the data including the tensorboard will be saved in
    #  2. add a function that will help us understand the data a little bit, how many labels we have from each class, and etc.
    #  3. save all of the args in the folder
    #  4. add trasformores to the figure

    train_data, val_data, test_data  = split_data(args.ucf_list_root, args.seed)

    train_dataset = UCF101Dataset(args.sampled_data_path, args.num_frames_video, train_data, mode = 'train')
    data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                          shuffle=True) #how to shift it to multiprocessing?

    val_dataset = UCF101Dataset(args.sampled_data_path, args.num_frames_video, val_data, mode='val')
    data_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=True)  # how to shift it to multiprocessing?

    test_dataset = UCF101Dataset(args.sampled_data_path, args.num_frames_video, test_data, mode='test')
    data_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=True)  # how to shift it to multiprocessing?


    #we need to have train_loader, val loader and test loader

if __name__=='__main__':
    main()



