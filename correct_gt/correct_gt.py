import os

import cv2
import numpy as np


minimum_connected_component_size = 500

def remove_small_connected_components(img):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = minimum_connected_component_size  

    #your answer image
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255    
    return img2


gt_basefolder = '../SkinDataset/GT'
ori_basefolder = '../SkinDataset/ORI'

out_basefolder = os.path.join(gt_basefolder,'Corrected')

if not os.path.isdir(out_basefolder):
    os.mkdir(out_basefolder)

train_filenames = ['11.jpg','24.jpg','44.jpg','83.jpg','331.jpg','429.jpg','789.jpg','841.jpg']
test_filenames = ['243.jpg','278.jpg']

t_val = 9
for filename in train_filenames:
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

    corrected_bin_im = remove_small_connected_components(bin_im)

    cv2.imwrite(os.path.join(out_basefolder,filename),corrected_bin_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

