from __future__ import absolute_import
from __future__ import division

import argparse
import os
import sys
import random

import cv2
import numpy as np

import tensorflow as tf

base_path = '.'
train_name = 'SFA_pixel_regions_training_random_pixels_set'
eval_name = 'SFA_pixel_regions_test_set'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


window_size = 64
halfsize = window_size//2

resized_window_size = 32

depth = 3


train_filename = os.path.join(base_path, train_name + '.tfrecords')
# eval_filename = os.path.join(base_path, eval_name + '.tfrecords')

train_writer = tf.python_io.TFRecordWriter(train_filename)
# eval_writer = tf.python_io.TFRecordWriter(eval_filename)

gt_basefolder = '../../FullSkinDataset/GT/Corrected'
ori_basefolder = '../../FullSkinDataset/ORI'

train_filenames = ['img ('+str(i)+').jpg' for i in range(1,783) ]
train_filenames.sort()

full_gt_train_filenames = [os.path.join(gt_basefolder,tr) for tr in train_filenames]

full_ori_train_filenames = [os.path.join(ori_basefolder,tr) for tr in train_filenames]


n_skin_examples_per_image = 500
n_background_examples_per_image = int(n_skin_examples_per_image*1.5)
skin_per_image = []
background_per_image = []

total_skin = 0
f = 0
for filename in full_gt_train_filenames:
    file_skin = []
    file_back = []
    im = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j] > 0:
                file_skin.append([i,j])
            else:
                file_back.append([i,j])

    random.shuffle(file_skin)
    random.shuffle(file_back)
    file_skin = file_skin[:n_skin_examples_per_image]
    file_back = file_back[:n_background_examples_per_image]

    skin_per_image.append(file_skin)
    background_per_image.append(file_back)


total_examples = n_skin_examples_per_image*len(train_filenames) + n_background_examples_per_image*len(train_filenames)

for fidx, filename in enumerate(full_ori_train_filenames):

    img = cv2.imread(filename,cv2.IMREAD_COLOR)
    rows = img.shape[0]
    cols = img.shape[1]
    for sk_idx in range(n_skin_examples_per_image):
        os.system('clear')
        print "Full_set_bigger_patch_train: %f%%" % ((float(sk_idx + 1 + fidx*(n_skin_examples_per_image+n_background_examples_per_image) )/ total_examples) * 100)
              
        position = skin_per_image[fidx][sk_idx]

        initial_i = position[0] - halfsize if position[0] - halfsize > 0 else 0
        final_i = initial_i + window_size if initial_i + window_size < rows else rows

        initial_j = position[1] - halfsize if position[1] - halfsize > 0 else 0
        final_j = initial_j + window_size if initial_j + window_size < cols else cols


        img_example = img[initial_i:final_i, initial_j:final_j, :]

        img_example = cv2.resize(img_example, (resized_window_size,resized_window_size), interpolation = cv2.INTER_LANCZOS4)

        image_raw = tf.compat.as_bytes(img_example.tostring())
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(window_size),
            'width': _int64_feature(window_size),
            'depth': _int64_feature(depth),
            'i': _int64_feature(i),
            'j': _int64_feature(j),
            'fidx': _int64_feature(fidx),            
            'label': _int64_feature(1),
            'image': _bytes_feature(image_raw)}))

        train_writer.write(example.SerializeToString())

    for b_idx in range(n_background_examples_per_image):
        os.system('clear')
        print "Full_set_bigger_patch_train: %f%%" % ((float(b_idx + 1 + fidx*(n_skin_examples_per_image+n_background_examples_per_image) + n_skin_examples_per_image )/ total_examples) * 100)
              
        position = background_per_image[fidx][b_idx]

        initial_i = position[0] - halfsize if position[0] - halfsize > 0 else 0
        final_i = initial_i + window_size if initial_i + window_size < rows else rows

        initial_j = position[1] - halfsize if position[1] - halfsize > 0 else 0
        final_j = initial_j + window_size if initial_j + window_size < cols else cols


        img_example = img[initial_i:final_i, initial_j:final_j, :]

        img_example = cv2.resize(img_example, (resized_window_size,resized_window_size), interpolation = cv2.INTER_LANCZOS4)

        image_raw = tf.compat.as_bytes(img_example.tostring())
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(window_size),
            'width': _int64_feature(window_size),
            'depth': _int64_feature(depth),
            'i': _int64_feature(i),
            'j': _int64_feature(j),
            'fidx': _int64_feature(fidx),            
            'label': _int64_feature(0),
            'image': _bytes_feature(image_raw)}))

        train_writer.write(example.SerializeToString())

train_writer.close()
