import numpy as np
import cv2
import os

gt_basefolder = '../FullSkinDataset/GT/Corrected'
pred_basefolder = '.'

filenames = [str(i) + '.jpg' for i in range(10) ]

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

    jaccard = np.sum(and_arr)/np.sum(or_arr)
    acc = np.sum(xnor_arr)/(a.shape[0]*a.shape[1])

    f_jac = f_jac + (jaccard/len(filenames_gt))
    f_acc = f_acc + (acc/len(filenames_gt))


print "Jaccard: %.3f"%(f_jac)
print "Accuracy: %.3f"%(f_acc)
