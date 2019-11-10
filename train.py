
import torch
import argparse
import os
import torchvision
import torchvision.transforms as trasforms
from torch.utils.data import DataLoader
from data import *

parser = argparse.ArgumentParser(description='UCF101 Action Recognition, LRCN architecture')
parser.add_argument('--epochs', default=500, type=int, help='number of total epochs')
parser.add_argument('--batch-size', default=16, type=int, help='mini-batch size (default: 32)')
parser.add_argument('--lr', default=5e-4, type=float, help='initial learning rate')
parser.add_argument('--num_workers', default=8, type=int, help='initial num_workers, the number of processes that generate batches in parallel')
parser.add_argument('--sampled_data_path_train', default=r'C:\Users\Doron\Desktop\ObjectRecognition\UCF101_sampled_data_3_np\train', type=str
                    , help='the path for the sampled row data')
parser.add_argument('--ucf_list_root', default=r'C:\Users\Doron\PycharmProjects\ObjectRecognition\Data_UCF101\UCF101_video_list/', type=str, help='path to find the UCF101 list splitting the data to train and test')
parser.add_argument('--data_format', default='np', type=str, help='the format the data is saved in')



def main():
    args = parser.parse_args()
    print(args)
    # open a new folder where all of the data including the tensorboard will be saved in
    #set what is the device the code will run on (    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    #add a function that will help us understand the data a little bit, how many labels we have from each class, and etc.
    #save all of the args in the folder
    #add trasformores to the figure

    transform = None
    ucf101_dataset = UCF101Dataset(args.sampled_data_path_train, args.ucf_list_root, args.data_format, transform)
    data_loader = torch.utils.data.DataLoader(args.sampled_data_path_train,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=args.num_workers)

if __name__=='__main__':
    main()



