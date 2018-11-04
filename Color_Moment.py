# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 17:32:27 2018

@author: 12088_000
"""
from __future__ import division  
from skimage import data
from skimage import color
from skimage import io

import numpy as np
import os
import scipy
import csv

#file='D:/美感分类实验信号-分组/气势美/2.85_l212.jpg'
path1='D:/美感分类实验信号-分组/'
lei='气势美','生机美','雅致美','萧瑟美','无法分类','清幽美'
zong=[]
for leibie in lei:
    path=path1+leibie+'/'
    filelist=os.listdir(path)
    for imname in filelist:
        file=path+imname
        image=io.imread(file)    
        ihsv=color.rgb2hsv(image)
        h=ihsv[:,:,0]
        s=ihsv[:,:,1]
        v=ihsv[:,:,2]
        r=image[:,:,0]
        g=image[:,:,1]
        b=image[:,:,2]
        
        moment=[imname,np.mean(r),np.mean(g),np.mean(b),\
                np.std(r),np.std(g),np.std(b),\
                #        np.mean(b),np.mean(b),np.mean(b),\
                abs(np.mean((r-np.mean(r))**3))**(1/3)*(-1 if np.mean((r-np.mean(r))**3)<0 else 1),\
                abs(np.mean((g-np.mean(g))**3))**(1/3)*(-1 if np.mean((g-np.mean(g))**3)<0 else 1),\
                abs(np.mean((b-np.mean(b))**3))**(1/3)*(-1 if np.mean((b-np.mean(b))**3)<0 else 1),\
                np.mean(v),np.std(v)]
        zong.append(moment)

myFile = open('D:/feature/brightness&colormoment.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(zong)