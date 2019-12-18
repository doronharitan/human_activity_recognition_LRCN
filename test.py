import argparse
import torch
from utils_action_recognition import set_project_folder_dir, save_setting_info

parser = argparse.ArgumentParser(description='UCF101 Action Recognition, LRCN architecture, test mode')
parser.add_argument('--model_dir', default=100, type=int, help='the dir for the model we want to test')
parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size (default:128)')
parser.add_argument('--sampled_data_path', default=r'C:\Users\Doron\Desktop\ObjectRecognition data\UCF101_sampled_data_video_10', type=str
                    , help='The dir for the sampled row data')
parser.add_argument('--ucf_list_root', default=r'C:\Users\Doron\Google Drive\Object detection light\Data_UCF101\UCF101_video_list - Copy (2)/',
                    type=str, help='path to find the UCF101 list, splitting the data to train and test')
parser.add_argument('--open_new_folder', default='True', type=str,
                    help='open a new folder for saving the run info, if false the info would be saved in the project dir, if debug the info would be saved in debug folder(default:True)')
parser.add_argument('--number_of_classes', default=None, type=int, help='The number of classes we would train on')



def main():
    # ====== set the run settings ======
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    folder_dir = set_project_folder_dir(args.open_new_folder, args.local_dir)
    print('The setting of the run are:\n %s' % args)
    print('The training would take place on %s' % device)
    print('The project directory is %s' % folder_dir)
    save_setting_info(args, device, folder_dir)

    train_data_names, val_data_names, test_data_names, label_decoder_dict = split_data(args.ucf_list_root, args.seed, args.number_of_classes, args.split_size)
    dataset_order = ['train', 'val', 'test']
    datasets = {dataset_order[index]: UCF101Dataset(args.sampled_data_path,
                                                    args.num_frames_video, x, mode=dataset_order[index])
                for index, x in enumerate([train_data_names, val_data_names, test_data_names])}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                 shuffle=True)
                   for x in ['train', 'val', 'test']}
    # ======= if args.smaller_dataset True load small portion of the dataset directly to the RAM (for faster computation) ======
    if args.smaller_dataset:
        dataloaders = get_small_dataset_dataloader(dataloaders, dataset_order, args.batch_size)
    plot_label_distribution(dataloaders, folder_dir)
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
        train_loss, train_acc = train_model(model, datasets['train'], dataloaders['train'], device, optimizer,
                                            criterion)
        if (epoch % args.val_check_interval) == 0:
            val_loss, val_acc, predicted_labels, local_x = test_model(model, datasets['val'], dataloaders['val'], device,
                                           criterion)
            plot_images_with_predicted_labels(local_x, label_decoder_dict, predicted_labels, folder_dir, epoch)
            end_epoch = time.time()
            # ====== print the status to the console and write it in tensorboard =======
            print('Epoch %d : Train loss %.3f, Train acc %.3f, Val loss %.3f, Val acc %.3f, epoch time %.3f'
                  % (epoch,train_loss, train_acc, val_loss, val_acc, end_epoch - start_epoch))
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
            torch.save(hp_dict, os.path.join(save_model_dir, 'epoch_%d.pth.tar' % (epoch)))


if __name__ == '__main__':
    main()

