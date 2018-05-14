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
train_name = 'SFA_pixel_regions_training_set'
eval_name = 'SFA_pixel_regions_test_set'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


window_size = 64
halfsize = window_size//2

resized_window_size = 32

depth = 3

reduced_size = 10

eval_filename = os.path.join(base_path, eval_name + '.tfrecords')


eval_writer = tf.python_io.TFRecordWriter(eval_filename)

gt_basefolder = '../../FullSkinDataset/GT/Corrected'
ori_basefolder = '../../FullSkinDataset/ORI'


test_filenames = ['img ('+str(i)+').jpg' for i in range(951,1119) ]

random.shuffle(test_filenames)

test_filenames = test_filenames[:reduced_size]

F = open("test_filenames.txt","w") 

for w_file in test_filenames:
    F.write(w_file+'\n')

F.close()


full_gt_test_filenames = [os.path.join(gt_basefolder,te) for te in test_filenames]

full_ori_test_filenames = [os.path.join(ori_basefolder,te) for te in test_filenames]


skin_per_image = []
background_per_image = []

im = cv2.imread(full_gt_test_filenames[0],cv2.IMREAD_COLOR)
rows = im.shape[0]
cols = im.shape[1]

total_examples = (rows*cols)*len(test_filenames)

for fidx, filename in enumerate(full_ori_test_filenames):

    img = cv2.imread(filename,cv2.IMREAD_COLOR)
    gt_img = cv2.imread(full_gt_test_filenames[fidx],cv2.IMREAD_GRAYSCALE)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            os.system('clear')
            print "Full_set_bigger_patch_small_test_full_file: %f%%" % ((float(idx + 1 + fidx*(rows*cols) )/ total_examples) * 100)

            label = int(gt_img[i,j] > 0)

            idx = idx + 1     

            initial_i = i - halfsize if i - halfsize > 0 else 0
            final_i = initial_i + window_size if initial_i + window_size < rows else rows

            initial_j = j - halfsize if j - halfsize > 0 else 0
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
                'label': _int64_feature(label),
                'image': _bytes_feature(image_raw)}))

            eval_writer.write(example.SerializeToString())

eval_writer.close()
