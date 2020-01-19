import torch
import torch.nn as nn
import argparse
import time
from torch.utils.data import DataLoader
from create_dataset import UCF101Dataset
from lrcn_model import ConvLstm
from utils_action_recognition import save_setting_info, plot_label_distribution, \
    plot_images_with_predicted_labels,  create_folder_dir_if_needed, load_all_dataset_to_RAM, split_data, \
    test_model, train_model, save_loss_info_into_a_file, set_project_folder_dir
from torch.utils.tensorboard import SummaryWriter
import os


parser = argparse.ArgumentParser(description='UCF101 Action Recognition, LRCN architecture')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs')
parser.add_argument('--batch-size', default=16, type=int, help='mini-batch size (default:32)')
parser.add_argument('--lr', default=5e-4, type=float, help='initial learning rate (default:5e-4')
parser.add_argument('--num_workers', default=4, type=int,
                    help='initial num_workers, the number of processes that generate batches in parallel (default:4)')
parser.add_argument('--split_size', default=0.2, type=int, help='set the size of the split size between validation '
                                                                'data and train data')
parser.add_argument('--sampled_data_dir',
                    default=r'C:\Users\Doron\Desktop\ObjectRecognition data\UCF101_sampled_data_video_sampling_rate_10_num frames extracted_5',
                    type=str, help='The dir for the sampled row data')
parser.add_argument('--ucf_list_dir', default=r'C:\Users\Doron\Desktop\ObjectRecognition\Data_UCF101\UCF101_video_list',
                    type=str, help='path to find the UCF101 list, splitting the data to train and test')
parser.add_argument('--num_frames_video', default=5, type=int,
                    help='The number of frames that would be sampled from each video (default:5)')
parser.add_argument('--seed', default=42, type=int,
                    help='initializes the pseudorandom number generator on the same number (default:42)')
parser.add_argument('--load_all_data_to_RAM', default=False, type=bool,
                    help='load dataset directly to the RAM, for faster computation. usually use when the num of class '
                         'is small (default:False')
parser.add_argument('--latent_dim', default=512, type=int, help='The dim of the Conv FC output (default:512)')
parser.add_argument('--hidden_size', default=256, type=int,
                    help="The number of features in the LSTM hidden state (default:256)")
parser.add_argument('--lstm_layers', default=2, type=int, help='Number of recurrent layers (default:2)')
parser.add_argument('--bidirectional', default=True, type=bool, help='set the LSTM to be bidirectional (default:True)')
parser.add_argument('--open_new_folder', default='True', type=str,
                    help='open a new folder for saving the run info, if false the info would be saved in the project '
                         'dir, if debug the info would be saved in debug folder(default:True)')
parser.add_argument('--load_checkpoint', default=False, type=bool,
                    help='Loading a checkpoint and continue training with it')
parser.add_argument('--checkpoint_path', default='', type=str, help='Optional path to checkpoint model')
parser.add_argument('--checkpoint_interval', default=5, type=int, help='Interval between saving model checkpoints')
parser.add_argument('--val_check_interval', default=5, type=int, help='Interval between running validation test')
parser.add_argument('--local_dir', default=os.getcwd(), help='The local directory of the project, setting where to '
                                                             'save the results of the run')
parser.add_argument('--number_of_classes', default=55, type=int, help='The number of classes we would train on')

def main():
    # ====== set the run settings ======
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    folder_dir = set_project_folder_dir(args.open_new_folder, args.local_dir)
    print('The setting of the run are:\n{}\n'.format(args))
    print('The training would take place on {}\n'.format(device))
    print('The project directory is {}'.format(folder_dir))
    save_setting_info(args, device, folder_dir)
    tensorboard_writer = SummaryWriter(folder_dir)

    print('Initializing Datasets and Dataloaders...')
    train_data_names, val_data_names, label_decoder_dict = split_data(args.ucf_list_dir, args.seed,
                                                                      args.number_of_classes, args.split_size, folder_dir)
    dataset_order = ['train', 'val']
    datasets = {dataset_order[index]: UCF101Dataset(args.sampled_data_dir, x, mode=dataset_order[index])
                for index, x in enumerate([train_data_names, val_data_names])}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True)
                   for x in ['train', 'val']}
    # ======= if args.load_all_data_to_RAM True load dataset directly to the RAM (for faster computation) ======
    if args.load_all_data_to_RAM:
        dataloaders = load_all_dataset_to_RAM(dataloaders, dataset_order, args.batch_size)
    plot_label_distribution(dataloaders, folder_dir, args.load_all_data_to_RAM, label_decoder_dict)
    print('Data prepared\nLoading model...')
    num_class = len(label_decoder_dict) if args.number_of_classes is None else args.number_of_classes
    model = ConvLstm(args.latent_dim, args.hidden_size, args.lstm_layers, args.bidirectional, num_class)
    model = model.to(device)
    # ====== setting optimizer and criterion parameters ======
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    if args.load_checkpoint:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    # ====== start training the model ======
    for epoch in range(args.epochs):
        start_epoch = time.time()
        train_loss, train_acc = train_model(model, dataloaders['train'], device, optimizer, criterion)
        if (epoch % args.val_check_interval) == 0:
            val_loss, val_acc, predicted_labels, images = test_model(model, dataloaders['val'], device,
                                           criterion)
            plot_images_with_predicted_labels(images, label_decoder_dict, predicted_labels, folder_dir, epoch)
            end_epoch = time.time()
            # ====== print the status to the console and write it in tensorboard =======
            print('Epoch {} : Train loss {:.8f}, Train acc {:.3f}, Val loss {:.8f}, Val acc {:.3f}, epoch time {:.4f}'
                  .format(epoch, train_loss, train_acc, val_loss, val_acc, end_epoch - start_epoch))
            tensorboard_writer.add_scalars('train/val loss', {'train_loss': train_loss,
                                                              'val loss': val_loss}, epoch)
            tensorboard_writer.add_scalars('train/val accuracy', {'train_accuracy': train_acc,
                                                                  'val accuracy': val_acc}, epoch)
            # ====== save the loss and accuracy in txt file ======
            save_loss_info_into_a_file(train_loss, val_loss, train_acc, val_acc, folder_dir, epoch)
        if (epoch % args.checkpoint_interval) == 0:
            hp_dict = {'model_state_dict': model.state_dict()}
            save_model_dir = os.path.join(folder_dir, 'Saved_model_checkpoints')
            create_folder_dir_if_needed(save_model_dir)
            torch.save(hp_dict, os.path.join(save_model_dir, 'epoch_{}.pth.tar'.format(epoch)))


if __name__ == '__main__':
    main()
