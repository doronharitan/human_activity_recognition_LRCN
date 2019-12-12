import torch
import torch.nn as nn
import argparse
import math
from torch.utils.data import DataLoader, TensorDataset
from data import UCF101Dataset
from model import ConvLstm
from utils_action_recognition import *
from torch.utils.tensorboard import SummaryWriter

# todo check if I need to capitalize
parser = argparse.ArgumentParser(description='UCF101 Action Recognition, LRCN architecture')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs')
parser.add_argument('--batch-size', default=32, type=int, help='mini-batch size (default:32)')
parser.add_argument('--lr', default=5e-4, type=float, help='initial learning rate (default:5e-4')
parser.add_argument('--num_workers', default=4, type=int,
                    help='initial num_workers, the number of processes that generate batches in parallel (default:4)')
parser.add_argument('--sampled_data_path', default=r'C:\Users\Doron\Desktop\ObjectRecognition data\UCF101_sampled_data_video_10', type=str
                    , help='The dir for the sampled row data')
parser.add_argument('--ucf_list_root', default=r'C:\Users\Doron\Google Drive\Object detection light\Data_UCF101\UCF101_video_list - Copy (2)/',
                    type=str, help='path to find the UCF101 list, splitting the data to train and test')
parser.add_argument('--num_frames_video', default=5, type=int,
                    help='The number of frames that would be sampled from each video (default:5)')
parser.add_argument('--seed', default=42, type=int,
                    help='initializes the pseudorandom number generator on the same number (default:42)')
parser.add_argument('--smaller_dataset', default=True, type=bool,
                    help='Train the network on smaller dataset, mostly uuseful for debug mode. (default:False')
parser.add_argument('--latent_dim', default=512, type=int, help='The dim of the Conv FC output (default:512)')
parser.add_argument('--hidden_size', default=256, type=int,
                    help='The number of featuers in the LSTM hidden state (default:256)')
parser.add_argument('--lstm_layers', default=2, type=int, help='Number of recurrent layers (default:2)')
parser.add_argument('--bidirectional', default=True, type=bool, help='set the LSTM to be bidirectional (default:True)')
parser.add_argument('--open_new_folder', default='True', type=str,
                    help='open a new folder for saving the run info, if false the info would be saved in the project dir, if debug the info would be saved in debug folder(default:True)')
parser.add_argument('--load_checkpoint', default=False, type=bool,
                    help='Loading a checkpoint and continue training with it')
parser.add_argument('--checkpoint_path', default='', type=str, help='Optional path to checkpoint model')
parser.add_argument('--checkpoint_interval', default=5, type=int, help='Interval between saving model checkpoints')
parser.add_argument('--val_check_interval', default=5, type=int, help='Interval between running validation test')
parser.add_argument('--local_dir', default=os.getcwd(), help='')
parser.add_argument('--number_of_classes', default=None, type=int,
                    help='')

# Todo - take small dataset and put it all on the GPU
#open an image folder
# open a model save folder
#change small dataset
# open all of the data on the ram and than run it on GPU? Try it on the colab?
# seed the init of the network
#check my code why does it have ups and downs?
# Todo take out from oop:
# maybe take more frames?
# todo upload all of the data to the disc and batch to the GPU form thier.
def main():
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args)
    print(device)
    if args.open_new_folder != 'False':
        folder_dir = open_new_folder(args.open_new_folder, args.local_dir)
    else:
        folder_dir = args.local_dir
    save_setting_info(args, device, folder_dir)
    tensorboard_writer = SummaryWriter(folder_dir)

    print('Initializing Datasets and Dataloaders...')
    train_data_names, val_data_names, test_data_names, label_decoder_dict = split_data(args.ucf_list_root, args.seed,
                                                                                       args.number_of_classes,
                                                                                       args.smaller_dataset)
    dataset_order = ['train', 'val', 'test']
    datasets = {dataset_order[index]: UCF101Dataset(args.sampled_data_path,
                                                    args.num_frames_video, x,mode=dataset_order[index])
                for index, x in enumerate([train_data_names, val_data_names, test_data_names])}
    plot_label_distribution(datasets, folder_dir)
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                 shuffle=True)
                   for x in ['train', 'val', 'test']}

    x_train, y_train, x_val, y_val = [], [], [], []  #todo change the names
    start = time.time()
    if args.smaller_dataset:  # todo change it so it would be prettier, change name
        for local_x_train, local_y_train in dataloaders['train']:
            x_train += [local_x_train]
            y_train += [local_y_train]
        for local_x_val, local_y_val in dataloaders['val']:
            x_val += [local_x_val]
            y_val += [local_y_val]
        end = time.time()
        print(end-start)
        start = time.time()
        x_train = torch.cat(x_train, axis=0)
        y_train = torch.cat(y_train, axis=0)
        x_val = torch.cat(x_val, axis=0)
        y_val = torch.cat(y_val, axis=0)
        end = time.time()
        print(end-start)
        start = time.time()
        datasets = {dataset_order[index]: TensorDataset(x[0], x[1]) for index, x in enumerate([[x_train, y_train], [x_val, y_val]])}
        dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                 shuffle=True)
                   for x in ['train', 'val']}
        end = time.time()
        print(end-start)
    # todo check how num of workers it influance the time of run, see if it works on windows num_workers=args.num_workers

    print('Data prepared\nLoading model...')
    start_loading = time.time()
    num_class = len(label_decoder_dict) if args.number_of_classes is None else args.number_of_classes
    # num_class = 101
    model = ConvLstm(args.latent_dim, args.hidden_size, args.lstm_layers, args.bidirectional, num_class)
    model = model.to(device)
    # setting optimizer and criterion parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()


    if args.load_checkpoint:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args.epoch = args.epoch - checkpoint['epoch']
        tensorboard_writer = checkpoint['tensorboard_state_dict']
    end_loading = time.time()
    print('loading time ', end_loading - start_loading)
    # todo add lr decay? .step with scheduler?
    for epoch in range(args.epochs):
        start_epoch = time.time()
        train_loss, train_acc = train_model(model, datasets['train'], dataloaders['train'], device, optimizer,
                                            criterion)
        # print('\nEpoch %d : Train loss %.3f Train acc %.3f ================'
        #       % (epoch, train_loss, train_acc))
        if (epoch % args.val_check_interval) == 0:
            val_loss, val_acc, predicted_labels, local_x = test_model(model, datasets['val'], dataloaders['val'], device,
                                           criterion)  # change it to every x epochs todo
            end_epoch = time.time()
            #todo change it to be prettier
            n_rows = 2
            n_cols = 2
            fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(10, 10))
            for row in range(n_rows):
                for col in range(n_cols):
                    img = local_x[col + (row * n_cols)][0].permute(1, 2, 0)
                    img_scale = (img - img.min()) / (img.max() - img.min())
                    ax[row, col].imshow(img_scale)
                    label_for_title = label_decoder_dict[predicted_labels[col + (row * n_cols)].item()]
                    ax[row, col].set_title(label_for_title)
                    ax[row, col].set_xticks([])
                    ax[row, col].set_yticks([])
            plt.savefig(os.path.join(folder_dir, 'predicted_labels %d epoch.png' % (epoch)))
            plt.close()
            print('Epoch %d : Train loss %.3f, Train acc %.3f, Val loss %.3f, Val acc %.3f, epoch time %.3f'
                  % (epoch,train_loss, train_acc, val_loss, val_acc, end_epoch - start_epoch))
            tensorboard_writer.add_scalars('train/val loss', {'train_loss': train_loss,
                                                              'val loss': val_loss}, epoch)
            tensorboard_writer.add_scalars('train/val accuracy', {'train_accuracy': train_acc,
                                                                  'val accuracy': val_acc}, epoch)
        if (epoch % args.checkpoint_interval) == 0:
            hp_dict = {'epoch': epoch,
                       'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optimizer}
            torch.save(hp_dict, os.path.join(folder_dir, 'epoch_%d.pth.tar' % (epoch)))
        # todo change it to a fuction who can vizualize the results:
        # tensorboard add a graph lr, adam? read about it todo?


def foward_step(model, x, y, criterion, mode=''):  # predections
    # Must be done before you run a new batch. Otherwise the LSTM will treat a new batch as a continuation of a sequence
    model.Lstm.reset_hidden_state()
    if mode == 'val':
        with torch.no_grad():
            output = model(x)
    else:
        output = model(x)
    # output_avr_frames = output.mean(dim=1) #todo now
    loss = criterion(output, y)
    # Accuracy calculation
    predicted_labels = output.detach().argmax(dim=1)
    acc = (predicted_labels == y).cpu().numpy().sum()
    return loss, acc, predicted_labels #need to be predicted_labels.cpu()


def train_model(model, dataset, dataloader, device, optimizer, criterion):
    train_loss, train_acc = 0.0, 0.0
    model.train()
    # iter_number = math.ceil(dataset.__len__() / dataloader.batch_size)
    # with tqdm(total=iter_number) as pbar:
    for local_x, local_y in dataloader:
        local_x, local_y = local_x.to(device), local_y.to(device)
        # zero the parameter gradients #todo module it or is it unnecessary?
        optimizer.zero_grad()
        loss, acc, ___ = foward_step(model, local_x, local_y, criterion, mode='train')
        # todo add a function which will evaluate the results. which one should we pix
        train_loss += loss.item()
        train_acc += acc
        loss.backward()  # compute the gradients
        optimizer.step()  # update the parameters with the gradients
            # pbar.update(1)
    train_acc = 100 * (train_acc / dataset.__len__())
    return train_loss, train_acc


def test_model(model, dataset, dataloader, device, criterion):
    val_loss, val_acc = 0.0, 0.0
    model.eval()  # todo add no grad so it would run faster, now
    iter_number = math.ceil(dataset.__len__() / dataloader.batch_size)
    # with tqdm(total=iter_number) as pbar:
    for local_x, local_y in dataloader:
        local_x, local_y = local_x.to(device), local_y.to(device)
        loss, acc, predicted_labels = foward_step(model, local_x, local_y, criterion, mode='val')
        # todo add a function which will evaluate the results. which one should we pix
        val_loss += loss.item()
        val_acc += acc
        # pbar.update(1)
    #todo cahnge to a module function
    val_acc = 100 * (val_acc / dataset.__len__())
    return val_loss, val_acc, predicted_labels, local_x.cpu()


if __name__ == '__main__':
    main()
