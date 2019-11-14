import torch
import torch.nn as nn
import argparse
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import UCF101Dataset, SplitData
from model import ConvLstm
from utils import *


parser = argparse.ArgumentParser(description='UCF101 Action Recognition, LRCN architecture')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs')
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
        self.tensorboard_writer = save_setting_info(args, device, self.folder_dir)

    def run(self):
        print('Initializing Datasets and Dataloaders...')
        split = SplitData()
        train_data, val_data, test_data, label_encoder = split.split(args.ucf_list_root, args.seed, args.smaller_dataset)
        data = [train_data, val_data, test_data]
        datasets = {x: UCF101Dataset(args.sampled_data_path, args.num_frames_video, data[index], mode=x)
                    for index, x in enumerate(['train', 'val', 'test'])}
        plot_label_distribution(datasets, self.folder_dir)

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
            self.train_loss, self.val_loss = 0.0 , 0.0
            self.train_acc, self.val_acc = 0.0, 0.0
            self.train_model(datasets['train'], dataloaders['train'], epoch)
            self.test_model(datasets['val'], dataloaders['val'], epoch)
            print('\nEpoch %d :\n Train loss %.3f, Val loss %.3f\n Train acc %.3f, Val acc %.3f\n================'
                  %(epoch, self.train_loss, self.val_loss, self.train_acc, self.val_acc))

            self.tensorboard_writer.add_scalars('train/val loss', {'train_loss': self.train_loss,
                                                  'val loss': self.val_loss}, epoch)
            self.tensorboard_writer.add_scalars('train/val accuracy', {'train_accuracy': self.train_acc,
                                                                   'val accuracy': self.val_acc}, epoch)

            # todo change it to a fuction who can vizualize the results:


    def foward_step(self, x,y):
        # Must be done before you run a new batch. Otherwise the LSTM will treat a new batch as a continuation of a sequence
        self.model.Lstm.reset_hidden_state()
        output = self.model.forward(x)
        output_avr_frames = output.mean(dim=1)
        loss = self.criterion(output_avr_frames, y)
        # Accuracy calculation
        predicted_labels = output_avr_frames.detach().argmax(dim=1)
        acc = (predicted_labels == y).cpu().numpy().sum()
        return loss, acc

    def train_model(self, dataset, dataloader, epoch):
        self.model.train()
        iter_number = math.ceil(dataset.__len__() / args.batch_size)
        # print('Training epoch %d\n' %(epoch))
        with tqdm(total=iter_number) as pbar:
            for local_x, local_y in dataloader:
                local_x, local_y = local_x.to(device), local_y.to(device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                loss, acc = self.foward_step(local_x, local_y)
                # todo add a function which will evaluate the results. which one should we pix
                self.train_loss += loss.item()
                self.train_acc += acc
                loss.backward()  # compute the gradients
                self.optimizer.step()  # update the parameters with the gradients
                pbar.update(1)
        self.train_acc = 100 * (self.train_acc/dataset.__len__())

    def test_model(self, dataset, dataloader, epoch):
        self.model.eval()
        iter_number = math.ceil(dataset.__len__() / args.batch_size)
        # print('validation epoch %d\n' %(epoch))
        with tqdm(total=iter_number) as pbar:
            for local_x, local_y in dataloader:
                local_x, local_y = local_x.to(device), local_y.to(device)
                loss, acc = self.foward_step(local_x, local_y)
                # todo add a function which will evaluate the results. which one should we pix
                self.val_loss += loss.item()
                self.val_acc += acc
                pbar.update(1)
        self.val_acc = 100 * (self.val_acc/ dataset.__len__())


if __name__=='__main__':
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ActionRecognition = Main()  #todo ask Alex for a help in changing the name
    ActionRecognition.run()


