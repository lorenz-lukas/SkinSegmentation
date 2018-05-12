#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np
import cv2 as cv
def data(directory):
  (db, Folders, img) = os.walk(directory).next()
  if(os.listdir(directory) == []):
      print('There is no image\n')
      exit(-1)
  else:
      return img
def segmentation(img,name):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    if thresh[0][0] == 1:
        ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
    cv.imshow('skin',unknown)
    cv.waitKey(1000)
    cv.destroyAllWindows()
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (40,40,400,600)
    cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, img = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    cv.imshow('skin',img)
    cv.waitKey(1000)
    cv.destroyAllWindows()
    #result = cv.add(unknown,img)
    result = cv.bitwise_and(unknown, img, mask=None)
    cv.imshow('skin',result)
    cv.waitKey(1000)
    cv.destroyAllWindows()
    if(not os.walk('SkinDataset/ORI/Luv/')):
            os.mkdir('SkinDataset/ORI/Luv/test/')
    cv.imwrite('SkinDataset/ORI/Luv/test/{}'.format(name), result)
    return result

def statistical_analysis(result,gt):
    true = 0
    false = 0
    white = 0
    i = 0
    h,w = result.shape[:2]
    for i in xrange(h):
        for j in xrange(w):
            if (result[i][j]/255 == gt[i][j].all() and gt[i][j].all() == 1):
                true+=1
            if (gt[i][j].all()==1):
                white+=1
    return true,(white - true)

def main():
    img_list = data(directory = 'SkinDataset/ORI/Luv/')
    gt_list = data(directory = 'SkinDataset/GT/Corrected')
    error = []
    img_list_size = len(img_list)
    for i in xrange(img_list_size):
        img = cv.imread('SkinDataset/ORI/Luv/{}'.format(img_list[-1]))
        gt = cv.imread('SkinDataset/GT/Corrected/{}'.format(gt_list[-1]))
        result = segmentation(img,img_list[-1])
        error.append(statistical_analysis(result,gt))
        print error[-1][-2]/error[-1][-1]
        del img_list[-1]
        del gt_list[-1]
    print error
main()
