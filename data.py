import tensorflow
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from os import listdir
import random
import cv2
import numpy as np

def read_image(path, input_shape):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_shape[:2])
    return image

def create_train_and_test(path, input_shape):
    total_imgs = sum([len(os.listdir(os.path.join(path, dir))) for dir in os.listdir(path)])
    dirs = os.listdir(path)
    random.shuffle(dirs)
    cnt = 0
    split_len = 0
    for i in range(len(dirs)):
        cnt += len(os.listdir(os.path.join(path, dirs[i])))
        split_len = i
        if cnt >= 8 * total_imgs // 10:
            break
    
    train_dirs, test_dirs = dirs[:split_len], dirs[split_len:]
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    for dir in train_dirs:
        imgs = os.listdir(os.path.join(path, dir))
        for i in range(len(imgs)-1):
            for j in range(i+1, len(imgs)):
                path1 = os.path.join(os.path.join(path, dir), imgs[i])
                path2 = os.path.join(os.path.join(path, dir), imgs[j])
                img1 = read_image(path1, input_shape)
                img2 = read_image(path2, input_shape)
                train_data.append((img1, img2))
                train_labels.append(1)

    for dir in test_dirs:
        imgs = os.listdir(os.path.join(path, dir))
        for i in range(len(imgs)-1):
            for j in range(i+1, len(imgs)):
                path1 = os.path.join(os.path.join(path, dir), imgs[i])
                path2 = os.path.join(os.path.join(path, dir), imgs[j])
                img1 = read_image(path1, input_shape)
                img2 = read_image(path2, input_shape)
                test_data.append((img1, img2))
                test_labels.append(1)

    train_tmp, test_tmp = [], []
    train_label_tmp, test_label_tmp = [], []
    used = []
    while len(train_tmp) < len(train_data):
        dir1, dir2 = random.sample(train_dirs, 2)
        path1 = os.path.join(os.path.join(path, dir1), random.choice(os.listdir(os.path.join(path, dir1))))
        path2 = os.path.join(os.path.join(path, dir2), random.choice(os.listdir(os.path.join(path, dir2))))
        if (path1, path2) in used:
            continue
        used.append((path1, path2))
        img1 = read_image(path1, input_shape)
        img2 = read_image(path2, input_shape)
        train_tmp.append((img1, img2))
        train_label_tmp.append(0)

    used = []
    while len(test_tmp) < len(test_data):
        dir1, dir2 = random.sample(test_dirs, 2)
        path1 = os.path.join(os.path.join(path, dir1), random.choice(os.listdir(os.path.join(path, dir1))))
        path2 = os.path.join(os.path.join(path, dir2), random.choice(os.listdir(os.path.join(path, dir2))))
        if (path1, path2) in used:
            continue
        used.append((path1, path2))
        img1 = read_image(path1, input_shape)
        img2 = read_image(path2, input_shape)
        test_tmp.append((img1, img2))
        test_label_tmp.append(0)

    train_data.extend(train_tmp)
    test_data.extend(test_tmp)
    train_labels.extend(train_label_tmp)
    test_labels.extend(test_label_tmp)

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    ind = np.arange(train_data.shape[0])
    np.random.shuffle(ind)
    train_data = train_data[ind]
    train_labels = train_labels[ind]
    
    ind = np.arange(test_data.shape[0])
    np.random.shuffle(ind)
    test_data = test_data[ind]
    test_labels = test_labels[ind]

    return train_data, train_labels, test_data, test_labels
