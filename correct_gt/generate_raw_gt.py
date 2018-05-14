import os
import re

import cv2
import numpy as np


jpg_pattern = re.compile('\w*\.jpg')



gt_basefolder = '../SkinDataset/GT'
ori_basefolder = '../SkinDataset/ORI'

out_basefolder = os.path.join(gt_basefolder,'Binary')

if not os.path.isdir(out_basefolder):
    os.mkdir(out_basefolder)


all_files = os.listdir(gt_basefolder)
filenames = [i for i in all_files if jpg_pattern.search(i) is not None]

t_val = 0
for filename in filenames:
    ori_filename = os.path.join(ori_basefolder,filename)
    gt_filename = os.path.join(gt_basefolder,filename)        
    ori_im = cv2.imread(ori_filename)
    gt_im = cv2.imread(gt_filename)
    gray_gt = cv2.cvtColor(gt_im, cv2.COLOR_BGR2GRAY)

    bin_im = np.copy(gray_gt)
    neg_im = np.copy(ori_im)
    for i in range(ori_im.shape[0]):
        for j in range(ori_im.shape[1]):
            norm = gray_gt[i,j]
            if norm > t_val:
                bin_im[i,j] = 255
            else:
                bin_im[i,j] = 0

    cv2.imwrite(os.path.join(out_basefolder,filename),bin_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

