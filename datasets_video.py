import os
import torch
import torchvision
import torchvision.datasets as datasets


ROOT_DATASET= '/usr/home/kop/MFF-pytorch'

def return_jester(modality):
    filename_categories = 'jester/category.txt'
    filename_imglist_train = 'jester/train_videofolder.txt'
    filename_imglist_val = 'jester/val_videofolder.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = '/usr/home/kop/MFF-pytorch/datasets/jester'
    elif modality == 'RGBFlow':
        prefix = '{:05d}.jpg'
        root_data = '/usr/home/kop/MFF-pytorch/datasets/jester'
    else:
        print('no such modality:'+modality)
        os.exit()
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_nvgesture(modality):
    filename_categories = 'nvgesture/category.txt'
    filename_imglist_train = 'nvgesture/train_videofolder.txt'
    filename_imglist_val = 'nvgesture/val_videofolder.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = '/data2/nvGesture'
    elif modality == 'RGBFlow':
        prefix = '{:05d}.jpg'
        root_data = '/data2/nvGesture'
    else:
        print('no such modality:'+modality)
        os.exit()
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_chalearn(modality):
    filename_categories = 'chalearn/category.txt'
    filename_imglist_train = 'chalearn/train_videofolder.txt'
    filename_imglist_val = 'chalearn/val_videofolder.txt'
    #filename_imglist_val = 'chalearn/test_videofolder.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = '/data2/ChaLearn'
    elif modality == 'RGBFlow':
        prefix = '{:05d}.jpg'
        root_data = '/data2/ChaLearn'
    else:
        print('no such modality:'+modality)
        os.exit()
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_dataset(dataset, modality):
    dict_single = {'jester':return_jester, 'nvgesture': return_nvgesture, 'chalearn': return_chalearn}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    file_categories = os.path.join(ROOT_DATASET, file_categories)
    with open(file_categories) as f:
        lines = f.readlines()
    categories = [item.rstrip() for item in lines]
    return categories, file_imglist_train, file_imglist_val, root_data, prefix

