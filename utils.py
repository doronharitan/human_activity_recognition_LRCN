import time
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def open_new_folder(open_folder_status):
    if open_folder_status == 'True':
        folder_name = time.strftime("%Y%m%d-%H%M%S")
    else:
        folder_name = 'debug'
    folder_dir = os.path.join(os.getcwd(), folder_name)
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
    writer = SummaryWriter(folder_dir)
    return writer


def plot_label_distribution(datasets, folder_dir):
    for dataset in datasets.keys():
        if dataset != 'test':
            plt.hist(datasets[dataset].ys, rwidth=0.9)
            plt.title('Histogram showing the frequency of each label\n' + dataset)
            plt.xlabel('label')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(folder_dir, dataset + '.jpg'))
            plt.clf()