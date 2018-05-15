import numpy as np
import cv2
import os
import re

jpg_pattern = re.compile('\w*\.jpg')


gt_basefolder = '../FullSkinDataset/GT/Binary'
pred_basefolder = './Req_2_Early_Stopping'

all_files = os.listdir(pred_basefolder)
filenames = [i for i in all_files if jpg_pattern.search(i) is not None]

filenames_gt = [os.path.join(gt_basefolder,k) for k in filenames]
filenames_predictions = [os.path.join(pred_basefolder,k) for k in filenames]

f_jac = 0.
f_acc = 0.

for i in range(len(filenames_gt)):
    a = cv2.imread(filenames_gt[i],cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(filenames_predictions[i],cv2.IMREAD_GRAYSCALE)

    or_arr = np.bitwise_or(a,b)
    and_arr = np.bitwise_and(a,b)
    xor_arr = np.bitwise_xor(a,b)
    xnor_arr = 255 - xor_arr

    jaccard = float(np.sum(and_arr!= 0))/np.sum(or_arr!= 0)
    acc = float(np.sum(xnor_arr!= 0))/(a.shape[0]*a.shape[1])

    f_jac = f_jac + (jaccard/len(filenames_gt))
    f_acc = f_acc + (acc/len(filenames_gt))


print "Jaccard: %.3f"%(f_jac)
print "Accuracy: %.3f"%(f_acc)
