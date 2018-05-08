#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np
import cv2 as cv

def main():
    directory = 'SkinDataset/ORI'
    (db, Folders, img) = os.walk(directory).next()
    if(os.listdir(directory) == []):
        print('There is no image\n')
    else:
        while(img != []):
            pic = cv.imread('SkinDataset/ORI/{}'.format(img[-1]))
            pic = cv.cvtColor(pic, cv.COLOR_BGR2Luv)#Luv
            cv.imshow('image',pic)
            cv.waitKey(1000)
            cv.destroyAllWindows()
            cv.imwrite('SkinDataset/luv_{}'.format(img[-1]), pic)
            del img[-1]

main()
