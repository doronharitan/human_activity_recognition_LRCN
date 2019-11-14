import torch
import torch.nn as nn
import argparse
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import UCF101Dataset, SplitData
from model import ConvLstm
import matplotlib.pyplot as plt
from utils import *
import time


parser = argparse.ArgumentParser(description='UCF101 Action Recognition, LRCN architecture')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs')
parser.add_argument('--batch-size', default=16, type=int, help='mini-batch size (default:16)')
parser.add_argument('--lr', default=5e-4, type=float, help='initial learning rate (default:5e-4')
parser.add_argument('--num_workers', default=4, type=int,
                    help='initial num_workers, the number of processes that generate batches in parallel (default:4)')
parser.add_argument('--sampled_data_path', default=r'C:\Users\Doron\Desktop\ObjectRecognition\UCF101_sampled_data_10', type=str
                    , help='The dir for the sampled row data')
parser.add_argument('--ucf_list_root', default=r'C:\Users\Doron\PycharmProjects\ObjectRecognition\Data_UCF101\UCF101_video_list/',
                    type=str, help='path to find the UCF101 list, splitting the data to train and test')
parser.add_argument('--num_frames_video', default=5, type=int,
                    help='The number of frames that would be sampled from each video (default:5)')
parser.add_argument('--seed', default=42, type=int,
                    help='initializes the pseudorandom number generator on the same number (default:42)')
parser.add_argument('--smaller_dataset', default=True, type=bool,
                    help='Train the network on smaller dataset, mostly uuseful for debug mode. (default:False')
parser.add_argument('--latent_dim', default=512, type=int, help='The dim of the Conv FC output (default:512)')
parser.add_argument('--hidden_size', default=256, type=int, help='The number of featuers in the LSTM hidden state (default:256)')
parser.add_argument('--lstm_layers', default=2, type=int, help='Number of recurrent layers (default:2)')
parser.add_argument('--bidirectional', default=True, type=bool, help='set the LSTM to be bidirectional (default:True)')
parser.add_argument('--open_new_folder', default='debug', type=str,
                    help='open a new folder for saving the new info,'
                         ' if false the info would be saved in the project dir, if debug the info would be saved in debug folder(default:True)')

class Main():
    def __init__(self):
        super(Main, self).__init__()
        print(args)
        print(device)
        if args.open_new_folder != 'False':
            self.folder_dir = open_new_folder(args.open_new_folder)
        else:
            self.folder_dir = os.getcwd()
        save_setting_info(args, device, self.folder_dir)

    def run(self):
        print('Initializing Datasets and Dataloaders...')
        split = SplitData()
        train_data, val_data, test_data, label_encoder = split.split(args.ucf_list_root, args.seed, args.smaller_dataset)
        data = [train_data, val_data, test_data]
        datasets = {x: UCF101Dataset(args.sampled_data_path, args.num_frames_video, data[index], mode=x)
                    for index, x in enumerate(['train', 'val', 'test'])}
        plot_label_distrabution(datasets, self.folder_dir)

        # work on a small dataset
        if args.smaller_dataset:
            total_train_len = len(train_data[0])
            list_x,list_y = [], []
            for idx in range(total_train_len):
                x,y = datasets['train'].__getitem__(idx)            # run faster than using the dataloader
                list_x.append(x)
                list_y.append(y)
            x, y = torch.stack(list_x)
            y = torch.stack(list_y)
            x, y = x.to(device), y.to(device)
        #work with all of the data
        else:
            dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)
                       for x in ['train', 'val', 'test']}

        print('Data prepared\nLoading model...')
        model = ConvLstm(args.latent_dim, args.hidden_size, args.lstm_layers, args.bidirectional, len(label_encoder))
        self.model = model.to(device)

        # setting optimizer and criterion parameters
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        self.criterion = nn.CrossEntropyLoss()

        # todo add lr decay?
        for epoch in range(args.epochs):
            self.train_loss = 0.0
            if args.smaller_dataset:
                args.batch_size = args.batch_size if total_train_len < args.batch_size else args.batch_size
                for batch in range(0,total_train_len,args.batch_size):
                    batch_end = args.batch_size if (batch + args.batch_size)  <= total_train_len else total_train_len - batch
                    local_x = x[batch:batch + batch_end]
                    local_y = y[batch:batch + batch_end]
                    self.learining_step(local_x, local_y)
            else:
                with tqdm(total=len(total_train_len/args.batch_size)) as pbar:
                    for local_x, local_y in dataloaders['train']:
                        local_x, local_y = local_x.to(device), local_y.to(device)
                        self.learining_step(local_x, local_y)
                        pbar.update(1)

            print(self.train_loss, epoch)
            # change it to a fuction who can vizualize the results:


    def learining_step(self, x,y):
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # Must be done before you run a new batch. Otherwise the LSTM will treat a new batch as a continuation of a sequence
        self.model.Lstm.reset_hidden_state()
        output = self.model.forward(x)
        loss = self.criterion(output, y)
        #todo add a function which will evaluate the results. which one should we pix
        self.train_loss += loss.item()
        loss.backward()  # compute the gradients
        self.optimizer.step()  # update the parameters with the gradients


if __name__=='__main__':
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_run = Main()  #todo ask Alex for a help in changing the name
    model_run.run()


