# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 18:21:48 2018

@author: 12088_000
"""



from __future__ import division  
from skimage import data
from skimage import color
from skimage import io
from skimage import feature

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
        #rule of thirds feature of h channel,
        #来源：2006 Studying Aesthetics in Photographic Images Using a Computational Approach
        ihsv=color.rgb2hsv(image)
        h=ihsv[:,:,0]
        s=ihsv[:,:,1]
        v=ihsv[:,:,2]
        ro3h=9*np.mean(ihsv[len(h[:,0])//3:2*len(h[:,0])//3,
                            len(h[0,:])//3:2*len(h[0,:])//3,
                            0])
        ro3s=9*np.mean(ihsv[len(s[:,0])//3:2*len(s[:,0])//3,
                            len(s[0,:])//3:2*len(s[0,:])//3,
                            0])
        ro3v=9*np.mean(ihsv[len(v[:,0])//3:2*len(v[:,0])//3,
                            len(v[0,:])//3:2*len(v[0,:])//3,
                            0])
        
        #size feature
        size1=len(h[0,:])
        size2=len(h[:,0])
        
        #aspect ratio
        ratio=size1/size2
        
        zong.append([imname,ro3h,ro3s,ro3v,size1,size2,ratio])

csvFile = open("rule_of_thirds_size_ratio.csv", "w+")
writer = csv.writer(csvFile)
for item in zong:
    writer.writerow(item)
csvfile.close()



#        fhog=feature.hog(color.rgb2grey(image),orientations=8, pixels_per_cell=(16, 16),
#                    cells_per_block=(1, 1))
        
#        fdaisy=feature.daisy(color.rgb2grey(image))
