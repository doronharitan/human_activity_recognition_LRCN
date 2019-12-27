import torch
import os
from torch import nn
from utils_action_recognition import set_project_folder_dir, \
    save_setting_info, load_test_data, get_small_dataset_dataloader_test, plot_label_distribution, test_model,\
    plot_images_with_predicted_labels, save_loss_info_into_a_file, create_video_with_labels, \
    plot_confusion_matrix, create_folder_dir_if_needed, plot_acc_per_class
from create_dataset import UCF101Dataset
from torch.utils.data import DataLoader
from lrcn_model import ConvLstm
from train import parser
from random import sample

parser.add_argument('--model_dir', default=r'C:\Users\Doron\Desktop\ObjectRecognition\20191218-214903\Saved_model_checkpoints', type=str, help='The dir of the model we want to test')
parser.add_argument('--model_name', default='epoch_30.pth.tar', type=str, help='the name for the model we want to test on')


def main():
    # ====== set the run settings ======
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    folder_dir = set_project_folder_dir(args.open_new_folder, args.model_dir, use_model_folder_dir=True, mode='test')
    print('The setting of the run are:\n{}\n' .format(args))
    print('The training would take place on {}\n'.format(device))
    print('The project directory is {}' .format(folder_dir))
    save_setting_info(args, device, folder_dir)
    test_videos_names, labels, label_decoder_dict = load_test_data(args.model_dir)
    dataset = UCF101Dataset(args.sampled_data_dir, args.num_frames_video, [test_videos_names, labels], mode='test')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # ======= if args.smaller_dataset True load small portion of the dataset directly to the RAM (for faster computation) ======
    if args.smaller_dataset:
        dataloader = get_small_dataset_dataloader_test(dataloader, args.batch_size)
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
    # test_loss, test_acc, predicted_labels, images, true_labels = test_model(model, dataloader, device, criterion, mode='save_prediction_label_list')
    # print('test loss {:.8f}, test_acc {:.3f}'.format(test_loss, test_acc))
    # save_loss_info_into_a_file(0, test_loss, 0, test_acc, folder_dir, 'test')
    # ====== analyze the test results ======
    # plot_images_with_predicted_labels(images, label_decoder_dict, predicted_labels[-1], folder_dir, 'test')
    save_path_plots = os.path.join(folder_dir, 'Plots')
    create_folder_dir_if_needed(save_path_plots)
    # for i in range(len(images)):
    #     create_video_with_labels(folder_dir, '{}_{}.avi'.format(label_decoder_dict[predicted_labels[-1][i].item()], i),
    #                              images[i], None, [predicted_labels[-1][i]], label_decoder_dict)
    import numpy as np
    with np.load(r'C:\Users\Doron\Desktop\ObjectRecognition\20191218-214903\test\predicted_labels.npz') as f:
        predicted_labels = torch.tensor(f['arr_0'])
    with np.load(r'C:\Users\Doron\Desktop\ObjectRecognition\20191218-214903\test\true_labels.npz') as f:
            true_labels = torch.tensor(f['arr_0'])
    # predicted_labels, true_labels = torch.cat(predicted_labels), torch.cat(true_labels)
    plot_confusion_matrix(predicted_labels, true_labels, label_decoder_dict, save_path_plots)
    plot_acc_per_class(predicted_labels, true_labels, label_decoder_dict, save_path_plots)



if __name__ == '__main__':
    main()

