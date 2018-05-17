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

def segmentation(img,name,gt):
    ## Watershed transformation
    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    #if thresh[364][256] == 0 and thresh[354][256] == 0 and thresh[374][256] == 0:
    #    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 15)
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=15)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.9*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)

    cv.imshow('skin',unknown)
    cv.waitKey(1000)
    cv.destroyAllWindows()

    if(jaccard(unknown,gt) < 50):
        ## BAckground removal
        mask = np.zeros(img.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (40,40,600,500)
        cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]
        gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
        ret, img = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        cv.imshow('skin',img)
        cv.waitKey(1000)
        cv.destroyAllWindows()

        result = cv.bitwise_and(img,unknown, mask=None)
        #cv.imshow('skin',result)
        cv.waitKey(1000)
        cv.destroyAllWindows()
        if(not os.walk('fullSkinDataset/ORI/results/')):
                os.mkdir('fullSkinDataset/ORI/results/train/')
        cv.imwrite('fullSkinDataset/ORI/results/train/{}'.format(name), result)
        return result
    else:
        if(not os.walk('fullSkinDataset/ORI/results/')):
                os.mkdir('fullSkinDataset/ORI/results/train/')
        cv.imwrite('SkinDataset/ORI/results/train/{}'.format(name), unknown)
        return unknown

def statistical_analysis(result,gt,name):
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
    jac = jaccard(result,gt)
    ac = accuracy(white,true)
    print "\nJaccard Index value:  "+name
    print jac
    print "\nAccuracy value:       "
    print ac
    return jac, ac

def jaccard(result,gt):
    gray = cv.cvtColor(gt,cv.COLOR_RGB2GRAY)
    ret, gt = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    if(np.sum(np.array(cv.bitwise_or(result, gt, mask=None))) != 0 ):
        j = float(np.sum(cv.bitwise_and(result, gt, mask=None)))/float(np.sum(cv.bitwise_or(result, gt, mask=None)))
    else:
        j = 0
    return j

def accuracy(w,t):
    return float(t)/float(w)

def main():
    print "###################################################"
    print "             Opencv 3.4.1  -  Python 2             "
    print "Skin classifier based on image processing methods  "
    print "###################################################"
    print ("\n\nWait a second while the parameters is being computed\n")

    img_list = ['img (985).jpg','img (1036).jpg','img (1042).jpg','img (1063).jpg','img (1071).jpg','img (1074).jpg','img (1084).jpg','img (1097).jpg','img (1101).jpg','img (1104).jpg']
    #img_list = ['img (985).jpg']
    jaccard_index = []
    accuracy_percentage = []
    img_list_size = len(img_list)
    for i in xrange(img_list_size):
        img = cv.imread('FullSkinDataset/ORI/{}'.format(img_list[-1]))
        #img = cv.cvtColor(img, cv.COLOR_BGR2Luv)#Luv
        gt = cv.imread('FullSkinDataset/GT/{}'.format(img_list[-1]))
        result = segmentation(img,img_list[-1],gt)
        j,a = statistical_analysis(result,gt,img_list[-1])
        jaccard_index.append(j)
        accuracy_percentage.append(a)
        j = 0
        a = 0
        del img_list[-1]
    print "\n\nMean jaccard:"
    print np.mean(jaccard_index)
    print np.std(jaccard_index)
    print "\n\nMean Accuracy:"
    print np.mean(accuracy_percentage)
    print np.std(accuracy_percentage)
main()
