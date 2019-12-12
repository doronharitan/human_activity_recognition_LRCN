import time
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def open_new_folder(open_folder_status, dir):
    if open_folder_status == 'True':
        folder_name = time.strftime("%Y%m%d-%H%M%S")
    else:
        folder_name = 'debug' #todo add when it is False
    folder_dir = os.path.join(dir, folder_name)
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
    return folder_dir

def save_setting_info(args, device, folder_dir):
    setting_file_name = os.path.join(folder_dir, 'setting_info.txt')
    args_dict = args.__dict__
    with open(setting_file_name, 'w') as f:
        for key, value in args_dict.items(): #Todo print it using json
            f.write(key + ' : ' + str(value) + '\n')
        f.write(str(device))


def plot_label_distribution(datasets, folder_dir):
    for dataset in datasets.keys():
        if dataset != 'test':
            plt.hist(datasets[dataset].ys, rwidth=0.9)
            plt.title('Histogram showing the frequency of each label\n' + dataset)
            plt.xlabel('label')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(folder_dir, dataset + '.jpg'))
            plt.clf()

def split_data(ucf_list_root, seed, number_of_classes, smaller_dataset=False):
    video_names_train, video_names_test, labels, labels_decoder_dict = get_video_list(ucf_list_root, number_of_classes)
    # if smaller_dataset:
        # video_names_train, labels = split_to_train_val(video_names_train, labels, seed, smaller_dataset=True)
    train_data, val_data = split_to_train_val(video_names_train, labels, seed)
    return train_data, val_data, [video_names_test] , labels_decoder_dict

def get_data(mode, video_names, list, number_of_classes, labels=[]):  #todo make it prettier
    # setting the train data files as a list so the not overpower the system
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

def get_video_list(ucf_list_root, number_of_classes):  # get_list_of_video_names
    video_names_train, video_names_test, labels = [], [], []
    for file in os.listdir(ucf_list_root):
        if 'train' in file:
            with open(ucf_list_root + file) as f:
                video_names = f.readlines()
            video_names_train, labels = get_data('train', video_names, video_names_train, number_of_classes, labels)
        elif 'classInd' in file:
            with open(ucf_list_root + file) as f:
                labels_decoder = f.readlines()     #decoder, todo check if we have a function that can read it stright to dict
            labels_decoder_dict = {int(x.split(' ')[0]) -1 : x.split(' ')[1].rstrip('\n') for x in labels_decoder}
        else:
            with open(ucf_list_root + file) as f:
                video_names = f.readlines()
            video_names_test, _ = get_data('test', video_names, video_names_test, number_of_classes)
    return video_names_train, video_names_test, labels, labels_decoder_dict

def split_to_train_val(video_names_train, labels, seed, smaller_dataset=False): #todo add the split size as an arg
    if smaller_dataset:
        _, video_names_train, _, labels_train = train_test_split(video_names_train, labels, test_size=0.001, random_state=seed)
        return video_names_train, labels_train
    else:
        video_names_train, video_names_val, labels_train, labels_val = \
                                                    train_test_split(video_names_train, labels, test_size=0.2, random_state=seed)
        return [video_names_train, labels_train], [video_names_val, labels_val]

