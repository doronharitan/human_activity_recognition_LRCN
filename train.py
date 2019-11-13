
import torch
import argparse
import os
import torchvision
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import UCF101Dataset, SplitData
from model import ConvLstm
import torch.nn as nn
import matplotlib.pyplot as plt
import time


parser = argparse.ArgumentParser(description='UCF101 Action Recognition, LRCN architecture')
parser.add_argument('--epochs', default=500, type=int, help='number of total epochs')
parser.add_argument('--batch-size', default=64, type=int, help='mini-batch size (default: 32)')
parser.add_argument('--lr', default=5e-4, type=float, help='initial learning rate')
parser.add_argument('--num_workers', default=4, type=int, help='initial num_workers, the number of processes that generate batches in parallel')
parser.add_argument('--sampled_data_path', default=r'C:\Users\Doron\Desktop\ObjectRecognition\UCF101_sampled_data_10', type=str
                    , help='the path for the sampled row data')
parser.add_argument('--ucf_list_root', default=r'C:\Users\Doron\PycharmProjects\ObjectRecognition\Data_UCF101\UCF101_video_list/', type=str, help='path to find the UCF101 list splitting the data to train and test')
parser.add_argument('--num_frames_video', default=5, type=int, help='the number of frames that would be taken from each video')
parser.add_argument('--seed', default=42, type=int, help='?')
parser.add_argument('--smaller_dataset', default=True, type=bool, help='?')

#Todo add an arg of the model type
#Todo try also  NASNat-A-Large.



def main():
    args = parser.parse_args()
    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Todo: 1.open a new folder where all of the data including the tensorboard will be saved in
    #  2. add a function that will help us understand the data a little bit, how many labels we have from each class, and etc.
    #  3. save all of the args in the folder
    #  4. add trasformores to the figure
    print('Initializing Datasets and Dataloaders...')
    split = SplitData()
    train_data, val_data, test_data, label_encoder = split.split(args.ucf_list_root, args.seed, args.smaller_dataset)
    data = [train_data, val_data, test_data]
    datasets = {x: UCF101Dataset(args.sampled_data_path, args.num_frames_video, data[index], mode=x)
                for index, x in enumerate(['train', 'val', 'test'])}
    if args.smaller_dataset:
        list_x,list_y = [], []
        for idx in range(len(train_data[0])):
            x,y = datasets['train'].__getitem__(idx)
            list_x.append(x)
            list_y.append(y)
        img_stack = torch.stack(list_x)
        label_stack = torch.stack(list_y)
        img_stack = img_stack.to(device)
        label_stack = label_stack.to(device)
    else:
        dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val', 'test']}

    #Load the conv model weights
    print('Loading model...')
    # print('removing final layer and freezing all of the rest') # Todo - create a function which enable to unfreeze diffrent number of layers, so we can compare the results
    # using the number of possible labels to set what would be the size of the new FC layer.
    model = ConvLstm('resnet152',len(label_encoder))
    model = model.to(device)
    output = model.forward(img_stack)
    params_to_update = model.parameters()
    #check what are the paramters that we would update
    # params_to_update = []
    # for name,param in model.named_parameters():
    #     if param.requires_grad == True:
    #         params_to_update.append(param)
    #         print("\t",name)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    # run 25 epochs on the dataset
    for epoch in range(1):
        train_loss = 0.0
        with tqdm(total=len(train_data[0])/args.batch_size) as pbar:
            for local_batch, local_labels in dataloaders['train']: #todo change when we have small data
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                output = model.forward(local_batch)
                # zero the parameter gradients
                optimizer.zero_grad()
                start_2 = time.time()
                loss = criterion(output,local_labels) #takes time.
                end_2 = time.time()
                print(end_2- start_2,'\n')
                train_loss += loss.item()
                loss.backward()  # compute the gradients
                optimizer.step() #update the paramters with the gradients
                pbar.update(1)

    print(train_loss, epoch)
    # change it to a fuction i utilitize:
    value, index = output.max(axis=1)
    fig, ax = plt.subplots(16,4)
    for i in range(16):
        for j in range(4):
            ax[i,j] = plt.imshow(local_batch[i])
            ax[i,j].set_title(label_encoder[str(value[i])])
    plt.show()
    end_3 = time.time()
    print('the end', end_3-start_3)

if __name__=='__main__':
    main()



