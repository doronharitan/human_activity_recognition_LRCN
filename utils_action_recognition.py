import time
import os
import pickle
import cv2
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import math
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
# from tqdm import tnrange, tqdm_notebook #used when I run in colab/GCloud
import torchvision.transforms as transforms
from random import sample
import numpy as np
import matplotlib.animation as manimation
import matplotlib.patheffects as pe


def set_project_folder_dir(if_open_new_folder, local_dir, use_model_folder_dir=False, mode=None):
    if use_model_folder_dir:
        folder_dir = os.path.join(os.path.normpath(local_dir + os.sep + os.pardir), mode)
        create_folder_dir_if_needed(folder_dir)
    else:
        if if_open_new_folder != 'False':
            folder_dir = open_new_folder(if_open_new_folder, local_dir)
        else:
            folder_dir = local_dir
    return folder_dir


def open_new_folder(if_open_new_folder, local_dir):
    if if_open_new_folder == 'True':
        folder_name = time.strftime("%Y%m%d-%H%M%S")
    else:
        folder_name = 'debug'
    folder_dir = os.path.join(local_dir, folder_name)
    create_folder_dir_if_needed(folder_dir)
    return folder_dir


def save_setting_info(args, device, folder_dir):
    setting_file_name = os.path.join(folder_dir, 'setting_info.txt')
    args_dict = args.__dict__
    with open(setting_file_name, 'w') as f:
        for key, value in args_dict.items():
            f.write(key + ' : ' + str(value) + '\n')
        f.write(str(device))


def plot_label_distribution(dataloaders, folder_dir, small_data_set_mode, mode='train'):
    if mode == 'train':
        for dataloader_name in dataloaders.keys():
            plot_distribution(dataloaders[dataloader_name].dataset, dataloader_name, small_data_set_mode, folder_dir)
    else:
        plot_distribution(dataloaders.dataset, 'test', small_data_set_mode, folder_dir)


def plot_distribution(dataset, dataset_name, small_data_set_mode, folder_dir):
    if small_data_set_mode:
        plt.hist(dataset.tensors[1], rwidth=0.9)
    else:
        plt.hist(dataset.labels, rwidth=0.9)
    plt.title('Histogram showing the frequency of each label\n' + dataset_name)
    plt.xlabel('label')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(folder_dir, dataset_name + '.jpg'))
    plt.clf()


def split_data(ucf_list_root, seed, number_of_classes, split_size, folder_dir):
    video_names_train, video_names_test, labels, labels_decoder_dict = get_video_list(ucf_list_root, number_of_classes)
    video_names_train, video_names_val, labels_train, labels_val = train_test_split(video_names_train, labels, test_size=split_size, random_state=seed)
    save_video_names_test_and_add_labels(video_names_test, labels_decoder_dict, folder_dir, number_of_classes)
    # save labels_decoder_dict
    with open(os.path.join(folder_dir,'labels_decoder_dict.pkl'), 'wb') as f:
        pickle.dump(labels_decoder_dict, f, pickle.HIGHEST_PROTOCOL)
    return [video_names_train, labels_train], [video_names_val, labels_val], labels_decoder_dict


def get_data(mode, video_names, list, number_of_classes, labels=[]):
    # setting the data files as a list so the not overpower the system
    for video_name in video_names:
        if mode == 'train':
            video_name, label = video_name.split(' ')
            label = int(label.rstrip('\n'))
            if number_of_classes is None or label in range(1, number_of_classes + 1):
                labels.append(label - 1)
                list.append(video_name.split('.')[0])
            else:
                continue
        else:
            list.append(video_name.split('.')[0])
    return list, labels


def get_video_list(ucf_list_root, number_of_classes):
    # ====== get a list of video names ======
    video_names_train, video_names_test, labels = [], [], []
    for file in os.listdir(ucf_list_root):
        if 'train' in file:
            with open(ucf_list_root + file) as f:
                video_names = f.readlines()
            video_names_train, labels = get_data('train', video_names, video_names_train, number_of_classes, labels)
        elif 'classInd' in file:
            with open(ucf_list_root + file) as f:
                labels_decoder = f.readlines()
            labels_decoder_dict = {int(x.split(' ')[0]) -1 : x.split(' ')[1].rstrip('\n') for x in labels_decoder}
        else:
            with open(ucf_list_root + file) as f:
                video_names = f.readlines()
            video_names_test, _ = get_data('test', video_names, video_names_test, number_of_classes)
    return video_names_train, video_names_test, labels, labels_decoder_dict


def save_video_names_test_and_add_labels(video_names_test, labels_decoder_dict, folder_dir,number_of_classes):
    save_test_video_details = os.path.join(folder_dir, 'test_videos_detailes.txt')
    with open(save_test_video_details, 'w') as f:
            for text_video_name in video_names_test:
                label_string = text_video_name.split('/')[0]
                # endoce label
                for key,value in labels_decoder_dict.items():
                    if value == label_string:
                        label_code = key
                    else:
                        continue
                if number_of_classes is None or label_code in range(1, number_of_classes + 1):
                    f.write(text_video_name + ' ' + str(label_code) + '\n')
                else:
                    continue



def plot_images_with_predicted_labels(local_x, label_decoder_dict, predicted_labels, folder_dir, epoch):
    folder_save_images = os.path.join(folder_dir, 'Images')
    create_folder_dir_if_needed(folder_save_images)
    n_rows =  math.trunc(math.sqrt(len(local_x)))
    n_cols = n_rows
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
    plt.savefig(os.path.join(folder_save_images, 'predicted_labels {} epoch.png' .format(epoch)))
    plt.close()


def create_folder_dir_if_needed(folder_save_dir, mode='single_folder'):
    if not os.path.exists(folder_save_dir):
        os.makedirs(folder_save_dir)
        if mode == 'preprocessing_data':
            os.makedirs(os.path.join(folder_save_dir, 'test'))
            os.makedirs(os.path.join(folder_save_dir, 'train'))


def get_small_dataset_dataloader(dataloaders, dataset_order, batch_size):
    images_train, labels_train, images_val, labels_val = [], [], [], []
    with tqdm(total=len(dataloaders['train'])) as pbar:
    # with tqdm_notebook(total=len(dataloaders['train'])) as pbar:
        for local_images_train, local_label_train in dataloaders['train']:
            images_train += [local_images_train]
            labels_train += [local_label_train]
            pbar.update(1)
    with tqdm(total=len(dataloaders['val'])) as pbar:
    # with tqdm_notebook(total=len(dataloaders['val'])) as pbar:
        for local_images_val, local_labels_val in dataloaders['val']:
            images_val += [local_images_val]
            labels_val += [local_labels_val]
            pbar.update(1)
    images_train = torch.cat(images_train, axis=0)
    labels_train = torch.cat(labels_train, axis=0)
    images_val = torch.cat(images_val, axis=0)
    labels_val = torch.cat(labels_val, axis=0)
    datasets = {dataset_order[index]: TensorDataset(x[0], x[1]) for index, x in
                enumerate([[images_train, labels_train], [images_val, labels_val]])}
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size,
                                 shuffle=True)
                   for x in ['train', 'val']}
    return dataloaders


def get_small_dataset_dataloader_test(dataloader, batch_size):
    images_test, labels_test =  [], []
    with tqdm(total=len(dataloader)) as pbar:
    # with tqdm_notebook(total=len(dataloader)) as pbar:
        for local_images, local_label in dataloader:
            images_test += [local_images]
            labels_test += [local_label]
            pbar.update(1)
    images_test = torch.cat(images_test, axis=0)
    labels_test = torch.cat(labels_test, axis=0)
    dataset = TensorDataset(images_test, labels_test)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def foward_step(model, images, labels, criterion, mode=''):  # predections
    # Must be done before you run a new batch. Otherwise the LSTM will treat a new batch as a continuation of a sequence
    model.Lstm.reset_hidden_state()
    if mode == 'test':
        with torch.no_grad():
            output = model(images)
    else:
        output = model(images)
    loss = criterion(output, labels)
    # Accuracy calculation
    predicted_labels = output.detach().argmax(dim=1)
    acc = (predicted_labels == labels).cpu().numpy().sum()
    return loss, acc, predicted_labels.cpu()


def train_model(model, dataloader, device, optimizer, criterion):
    train_loss, train_acc = 0.0, 0.0
    model.train()
    with tqdm(total=len(dataloader)) as pbar:
    # with tqdm_notebook(total=len(dataloader)) as pbar:
        for local_images, local_labels in dataloader:
            local_images, local_labels = local_images.to(device), local_labels.to(device)
            optimizer.zero_grad()       # zero the parameter gradients
            loss, acc, ___ = foward_step(model, local_images, local_labels, criterion, mode='train')
            train_loss += loss.item()
            train_acc += acc
            loss.backward()             # compute the gradients
            optimizer.step()            # update the parameters with the gradients
            pbar.update(1)
    train_acc = 100 * (train_acc / dataloader.dataset.__len__())
    train_loss = train_loss / len(dataloader)
    return train_loss, train_acc


def test_model(model, dataloader, device, criterion):
    val_loss, val_acc = 0.0, 0.0
    model.eval()
    with tqdm(total=len(dataloader)) as pbar:
    # with tqdm_notebook(total=len(dataloader)) as pbar:
        for local_images, local_labels in dataloader:
            local_images, local_labels = local_images.to(device), local_labels.to(device)
            loss, acc, predicted_labels = foward_step(model, local_images, local_labels, criterion, mode='test')
            val_loss += loss.item()
            val_acc += acc
            pbar.update(1)
    val_acc = 100 * (val_acc / dataloader.dataset.__len__())
    val_loss = val_loss / len(dataloader)
    return val_loss, val_acc, predicted_labels, local_images.cpu()


def test_model_continues_movie(model, dataloader, device, criterion, path_save_movies, label_decoder_dict, checkpoint_interval, num_frames_to_sample=5):
    val_loss, val_acc = 0.0, 0.0
    model.eval()
    labels_for_plot_analysis, predicted_labels_for_plot_analysis = [], []
    transform = set_transforms()
    with tqdm(total=len(dataloader)) as pbar:
        # with tqdm_notebook(total=len(dataloader)) as pbar:
        for index, (local_images, local_labels) in enumerate(dataloader):
            predicted_labels_list = []
            # ===== create continues movie and labels tensor, with X frames from each movie ======
            # ===== and stack a sliding window of size 5 frames to new dim so they will act as batch ======
            sliding_window_images, continues_labels, continues_movie = create_sliding_window_x_frames_size_dataset\
                (local_images, local_labels, num_frames_to_sample, transform)
            # ====== predict the label of each sliding window, use batches beacuse of GPU memory ======
            for batch_boundaries in range(0, len(sliding_window_images), dataloader.batch_size):
                batch_images_to_plot = sliding_window_images[batch_boundaries: batch_boundaries + dataloader.batch_size].to(device)
                batch_labels = continues_labels[batch_boundaries: batch_boundaries + dataloader.batch_size].to(device)
                loss, acc, predicted_labels = foward_step(model, batch_images_to_plot, batch_labels, criterion, mode='test')
                predicted_labels_list += [predicted_labels.detach().cpu()]
            predicted_labels = torch.cat(predicted_labels_list, axis=0)
            labels_for_plot_analysis += [continues_labels.detach().cpu()]
            predicted_labels_for_plot_analysis += [predicted_labels]
            val_loss += loss.item()
            val_acc += acc
            if index % checkpoint_interval == 0:
                create_video_with_labels(path_save_movies, 'prediction_{}_batch.avi'.format(index), continues_movie, continues_labels, predicted_labels, label_decoder_dict)
            pbar.update(1)
    analysis_of_predicted_labels_in_sliding_window(predicted_labels_for_plot_analysis, labels_for_plot_analysis, num_frames_to_sample, path_save_movies)
    val_acc = 100 * (val_acc / dataloader.dataset.__len__())
    val_loss = val_loss / len(dataloader)
    return val_loss, val_acc, predicted_labels, local_images.cpu()


def create_sliding_window_x_frames_size_dataset(local_images, local_labels, num_frames_to_sample, transform):
    # ===== create continues movie, with X frames from each movie ======
    local_images = local_images[:, :num_frames_to_sample]
    continues_frames = local_images.view(local_images.shape[0] * local_images.shape[1], local_images.shape[2],
                                         local_images.shape[3], local_images.shape[4])
    # ==== create continues label tensor where each frame has its own label ======
    continues_labels = local_labels.view(-1, 1).repeat(1, num_frames_to_sample).view(-1)
    continues_labels = continues_labels[:len(continues_labels) - num_frames_to_sample + 1] # remove the last frames where the sliding window couldn't pass
    sliding_window_images = []
    for num_frame in range(continues_frames.shape[0] - 5):  # todo see if I can change the sliding window size
        # ===== normalize the frames according to the imagenet preprocessing =======
        sliding_window_images += [torch.stack([transform(image) for image in continues_frames[num_frame: num_frame + 5]])]
    sliding_window_images += [torch.stack([transform(image) for image in continues_frames[continues_frames.shape[0] - 5: continues_frames.shape[0]]])]
    sliding_window_images = torch.stack(sliding_window_images)
    return sliding_window_images, continues_labels, continues_frames


def analysis_of_predicted_labels_in_sliding_window(predicted_labels_for_plot_analysis, labels_for_plot_analysis, num_frames_to_sample, path_save_movies):
    std_acc_array, mean_acc_array = [], []
    predicted_labels = torch.cat(predicted_labels_for_plot_analysis, axis=0)
    labels = torch.cat(labels_for_plot_analysis, axis=0)
    for num_frames in range(num_frames_to_sample):
        predicted_labels_with_num_frames_in_window = np.array([predicted_labels[i] for i in range(num_frames, len(predicted_labels), num_frames_to_sample)])
        labels_with_num_frames = np.array([labels[i] for i in range(num_frames, len(labels), num_frames_to_sample)])
        acc = (predicted_labels_with_num_frames_in_window == labels_with_num_frames)
        mean_acc_array += [acc.mean()]
        std_acc_array += [acc.std()]
    mean_acc_array.reverse()
    std_acc_array.reverse()
    plt.errorbar(np.arange(1, 1+ num_frames_to_sample), mean_acc_array, yerr=std_acc_array, linestyle='-', marker="o")
    # plt.errorbar(std_acc_array)
    plt.savefig(os.path.join(path_save_movies, 'analysis_of_predicted_labels_in_sliding_window.png'), dpi=300)

def save_loss_info_into_a_file(train_loss, val_loss, train_acc, val_acc, folder_dir, epoch):
    file_name = os.path.join(folder_dir, 'loss_per_epoch.txt')
    with open(file_name, 'a+') as f:
        f.write('Epoch {} : Train loss {:.8f}, Train acc {:.4f}, Val loss {:.8f}, Val acc {:.4f}\n'
                  .format(epoch, train_loss, train_acc, val_loss, val_acc))


def set_transform_and_save_path(folder_dir, file_name):
    if 'test' in file_name:
        transform = set_transforms('test')
        save_path = os.path.join(folder_dir, 'test')
    elif 'train' in file_name:
        transform = set_transforms('train')
        save_path = os.path.join(folder_dir, 'train')
    return transform, save_path


def set_transforms(mode):
    if mode == 'train':
        transform = transforms.Compose(
            [transforms.Resize(256),  # this is set only because we are using Imagenet pretrain model.
             transforms.RandomCrop(224),
             transforms.RandomHorizontalFlip()
             ])
    else:
        transform = transforms.Compose([transforms.Resize((224, 224))])
    return transform


def create_new_video(save_path, video_name, image_array):
    (h, w) = image_array[0].size[:2]
    save_video_path = os.path.join(save_path, video_name.split('/')[1])
    output_video = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'MJPG'), 5, (w, h), True)
    for frame in range(len(image_array)):
        output_video.write(image_array[frame])
    output_video.release()
    cv2.destroyAllWindows()

def create_video_with_labels(save_path, video_name, image_array, continues_labels, predicted_labels, label_decoder_dict):
    bool_array = continues_labels == predicted_labels
    image_array = image_array.transpose(2, 1).transpose(2, 3).numpy()
    n_frames = len(predicted_labels)
    h_fig = plt.figure(figsize=(4, 4))
    h_ax = h_fig.add_axes([0.05, 0.05, 0.9, 0.87])
    h_im = h_ax.matshow(image_array[0])
    h_ax.set_axis_off()
    h_im.set_interpolation('none')
    h_ax.set_aspect('equal')
    h_text_1 = plt.text(0.02, 0.97, 'Original_labels', color='black', fontsize=8, transform=plt.gcf().transFigure)
    h_text_2 = plt.text(0.02, 0.94, 'Predicted_labels', color='blue', fontsize=8, transform=plt.gcf().transFigure)
    h_text_3 = plt.text(0.47, 0.01, 'True/False', color='red', fontsize=8, transform=plt.gcf().transFigure,
                        path_effects=[pe.withStroke(linewidth=1, foreground="black")])
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=video_name, artist='Matplotlib')
    writer = FFMpegWriter(fps=3, metadata=metadata)
    with writer.saving(h_fig, os.path.join(save_path, video_name), dpi=150):  # change from 600 dpi
        for i in range(n_frames):
            h_im.set_array(image_array[i])
            h_text_1.set_text('Original label - {}'.format(label_decoder_dict[continues_labels[i].item()]))
            h_text_2.set_text('Predicted labels - {}'.format(label_decoder_dict[predicted_labels[i].item()]))
            color = 'green' if bool_array[i].item() else 'red'
            h_text_3.remove()
            h_text_3 = plt.text(0.44, 0.01, str(bool_array[i].item()), color=color, fontsize=8, transform=plt.gcf().transFigure,
                                path_effects=[pe.withStroke(linewidth=1, foreground="black")])
            writer.grab_frame()
    plt.close()



def setting_sample_rate(num_frames_to_extract, sampling_rate, video, fps):
    video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    video_length = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
    num_frame = int(video_length * fps)
    if video_length < (num_frames_to_extract * sampling_rate):
        sample_start_point = 0
        sampling_rate = 2
    else:
        sample_start_point = sample(range(num_frame - (num_frames_to_extract * sampling_rate)), 1)[0]
    return sample_start_point, sampling_rate

def load_test_data(model_dir):
    globel_dir = os.path.normpath(model_dir + os.sep + os.pardir)
    with open(os.path.join(globel_dir, 'test_videos_detailes.txt')) as f:
        video_list = f.readlines()
    test_videos_names, labels = [], []
    for video_name_with_label in video_list:
        video_name, label = video_name_with_label.split(' ')
        test_videos_names += [video_name]
        labels += [label]
    # open labels_decoder_dict
    with open(os.path.join(globel_dir,'labels_decoder_dict.pkl'), 'rb') as f:
        labels_decoder_dict = pickle.load(f)
    return test_videos_names, labels, labels_decoder_dict

def set_transforms():
    # ===== the separated transform for train and test was done in the preprocessing data script =======
    transform = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                             std=(0.229, 0.224, 0.225))
    return transform