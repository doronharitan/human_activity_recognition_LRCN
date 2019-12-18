import time
import os

import cv2
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import math
from torch.utils.data import DataLoader, TensorDataset
import tqdm
from tqdm import tnrange, tqdm_notebook #used when I run in colab/GCloud
import torchvision.transforms as transforms
from random import sample


def set_project_folder_dir(if_open_new_folder, local_dir):
    if if_open_new_folder != 'False':
        folder_dir = open_new_folder(open_new_folder, local_dir)
    else:
        folder_dir = local_dir
    return folder_dir

def open_new_folder(open_folder_status, dir):
    if open_folder_status == 'True':
        folder_name = time.strftime("%Y%m%d-%H%M%S")
    else:
        folder_name = 'debug'
    folder_dir = os.path.join(dir, folder_name)
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
    return folder_dir

def save_setting_info(args, device, folder_dir):
    setting_file_name = os.path.join(folder_dir, 'setting_info.txt')
    args_dict = args.__dict__
    with open(setting_file_name, 'w') as f:
        for key, value in args_dict.items():
            f.write(key + ' : ' + str(value) + '\n')
        f.write(str(device))


def plot_label_distribution(dataloaders, folder_dir):
    for dataloader_name in dataloaders.keys():
        if dataloader_name != 'test':
            plt.hist(dataloaders[dataloader_name].dataset.tensors[1], rwidth=0.9)
            plt.title('Histogram showing the frequency of each label\n' + dataloader_name)
            plt.xlabel('label')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(folder_dir, dataloader_name + '.jpg'))
            plt.clf()

def split_data(ucf_list_root, seed, number_of_classes, split_size):
    video_names_train, video_names_test, labels, labels_decoder_dict = get_video_list(ucf_list_root, number_of_classes)
    video_names_train, video_names_val, labels_train, labels_val = train_test_split(video_names_train, labels, test_size=split_size, random_state=seed)
    return [video_names_train, labels_train], [video_names_val, labels_val], [video_names_test], labels_decoder_dict


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


def plot_images_with_predicted_labels(local_x, label_decoder_dict, predicted_labels, folder_dir, epoch):
    folder_save_images = os.path.join(folder_dir, 'Images')
    create_folder_dir_if_needed(folder_save_images)
    n_rows = round(math.sqrt(len(local_x)))
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
    plt.savefig(os.path.join(folder_save_images, 'predicted_labels %d epoch.png' % (epoch)))
    plt.close()


def create_folder_dir_if_needed(folder_save_dir, mode='single_folder'):
    if not os.path.exists(folder_save_dir):
        os.makedirs(folder_save_dir)
        if mode == 'preprocessing_data':
            os.makedirs(os.path.join(folder_save_dir, 'test'))
            os.makedirs(os.path.join(folder_save_dir, 'train'))


def get_small_dataset_dataloader(dataloaders, dataset_order, batch_size):
    images_train, labels_train, images_val, labels_val = [], [], [], []
    for local_images_train, local_label_train in dataloaders['train']:
        images_train += [local_images_train]
        labels_train += [local_label_train]
    for local_images_val, local_labels_val in dataloaders['val']:
        images_val += [local_images_val]
        labels_val += [local_labels_val]
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

def foward_step(model, images, labels, criterion, mode=''):  # predections
    # Must be done before you run a new batch. Otherwise the LSTM will treat a new batch as a continuation of a sequence
    model.Lstm.reset_hidden_state()
    if mode == 'val':
        with torch.no_grad():
            output = model(images)
    else:
        output = model(images)
    loss = criterion(output, labels)
    # Accuracy calculation
    predicted_labels = output.detach().argmax(dim=1)
    acc = (predicted_labels == labels).cpu().numpy().sum()
    return loss, acc, predicted_labels.cpu()


def train_model(model, dataset, dataloader, device, optimizer, criterion):
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
    train_acc = 100 * (train_acc / dataset.__len__())
    return train_loss, train_acc


def test_model(model, dataset, dataloader, device, criterion):
    val_loss, val_acc = 0.0, 0.0
    model.eval()
    with tqdm(total=len(dataloader)) as pbar:
    # with tqdm_notebook(total=len(dataloader)) as pbar:
        for local_images, local_labels in dataloader:
            local_images, local_labels = local_images.to(device), local_labels.to(device)
            loss, acc, predicted_labels = foward_step(model, local_images, local_labels, criterion, mode='val')
            val_loss += loss.item()
            val_acc += acc
            pbar.update(1)
    val_acc = 100 * (val_acc / dataset.__len__())
    return val_loss, val_acc, predicted_labels, local_images.cpu()


def save_loss_info_into_a_file(train_loss, val_loss, train_acc, val_acc, folder_dir, epoch):
    file_name = os.path.join(folder_dir, 'loss_per_epoch.txt')
    with open(file_name, 'a+') as f:
        f.write('Epoch %d : Train loss %.4f, Train acc %.4f, Val loss %.4f, Val acc %.4f\n'
                  % (epoch, train_loss, train_acc, val_loss, val_acc))


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