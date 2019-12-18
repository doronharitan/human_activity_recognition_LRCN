import argparse
import torch
from torch import nn
from utils_action_recognition import set_project_folder_dir, \
    save_setting_info, load_test_data, get_small_dataset_dataloader_test, plot_label_distribution, test_model,\
    plot_images_with_predicted_labels, save_loss_info_into_a_file
from data import UCF101Dataset
from torch.utils.data import DataLoader
from model import ConvLstm


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
    print('The setting of the run are:\n{}\n' .format(args))
    print('The training would take place on {}\n'.format(device))
    print('The project directory is {}' .format(folder_dir))
    save_setting_info(args, device, folder_dir)
    test_videos_names, labels, label_decoder_dict = load_test_data(args.model_dir)
    dataset = UCF101Dataset(args.sampled_data_path, args.num_frames_video, [test_videos_names, labels], mode='test')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # ======= if args.smaller_dataset True load small portion of the dataset directly to the RAM (for faster computation) ======
    if args.smaller_dataset:
        dataloader = get_small_dataset_dataloader_test(dataloader, args.batch_size)
    plot_label_distribution(dataloader, folder_dir, args.smaller_dataset, mode='test')
    print('Data prepared\nLoading model...')
    num_class = len(label_decoder_dict) if args.number_of_classes is None else args.number_of_classes
    model = ConvLstm(args.latent_dim, args.hidden_size, args.lstm_layers, args.bidirectional, num_class)
    model = model.to(device)
    # ====== setting optimizer and criterion parameters ======
    criterion = nn.CrossEntropyLoss()
    checkpoint = torch.load(args.model_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    # ====== inferance_mode ======
    test_loss, test_acc, predicted_labels, images = test_model(model, dataloader, device, criterion)
    plot_images_with_predicted_labels(images, label_decoder_dict, predicted_labels, folder_dir, 'test')
    # ====== print the status to the console =======
    print('test loss {:.3f}, Val test_acc {:.3f}' .format(test_loss, test_acc))
    # ====== save the loss and accuracy in txt file ======
    save_loss_info_into_a_file(0, test_loss, 0, test_acc, folder_dir, 'test')


if __name__ == '__main__':
    main()

