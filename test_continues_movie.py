import torch
import os
from torch import nn
from utils_action_recognition import set_project_folder_dir, \
    save_setting_info, load_test_data, get_small_dataset_dataloader_test, plot_label_distribution, \
    plot_images_with_predicted_labels, save_loss_info_into_a_file, test_model_continues_movie, create_folder_dir_if_needed, \
    check_if_batch_size_bigger_than_num_classes
from create_dataset import UCF101Dataset, UCF101DatasetSampler
from torch.utils.data import DataLoader
from lrcn_model import ConvLstm
from train import parser

parser.add_argument('--model_dir', default=r'C:\Users\Doron\Desktop\ObjectRecognition\20191218-214903\Saved_model_checkpoints', type=str, help='The dir of the model we want to test')
parser.add_argument('--model_name', default='epoch_30.pth.tar', type=str, help='the name for the model we want to test on')


def main():
    # ====== set the run settings ======
    args = parser.parse_args()
    check_if_batch_size_bigger_than_num_classes(args.batch_size, args.number_of_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    folder_dir = set_project_folder_dir(args.open_new_folder, args.model_dir, use_model_folder_dir=True, mode='test_continues_movie')
    print('The setting of the run are:\n{}\n' .format(args))
    print('The training would take place on {}\n'.format(device))
    print('The project directory is {}' .format(folder_dir))
    save_setting_info(args, device, folder_dir)
    test_videos_names, labels, label_decoder_dict = load_test_data(args.model_dir)
    dataset = UCF101Dataset(args.sampled_data_dir, [test_videos_names, labels], mode='test')
    sampler = UCF101DatasetSampler(dataset, args.batch_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler)
    plot_label_distribution(dataloader, folder_dir, args.smaller_dataset, label_decoder_dict, mode='test')
    print('Data prepared\nLoading model...')
    num_class = len(label_decoder_dict) if args.number_of_classes is None else args.number_of_classes
    model = ConvLstm(args.latent_dim, args.hidden_size, args.lstm_layers, args.bidirectional, num_class)
    model = model.to(device)
    # ====== setting optimizer and criterion parameters ======
    criterion = nn.CrossEntropyLoss()
    checkpoint = torch.load(os.path.join(args.model_dir, args.model_name))
    model.load_state_dict(checkpoint['model_state_dict'])
    # ====== inferance_mode ======
    test_loss, test_acc, predicted_labels, images = test_model_continues_movie(model, dataloader, device, criterion,
                                                                               folder_dir, label_decoder_dict)
    plot_images_with_predicted_labels(images, label_decoder_dict, predicted_labels, folder_dir, 'test')
    # ====== print the status to the console =======
    print('test loss {:.8f}, test_acc {:.3f}' .format(test_loss, test_acc))
    # ====== save the loss and accuracy in txt file ======
    save_loss_info_into_a_file(0, test_loss, 0, test_acc, folder_dir, 'test')


if __name__ == '__main__':
    main()

