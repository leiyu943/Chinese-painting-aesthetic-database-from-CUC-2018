# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:45:21 2018

@author: 12088_000
"""

from __future__ import division  
from skimage import data
from skimage import color
from skimage import io
from skimage import feature
from skimage.morphology import octagon

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
        imgrey=color.rgb2gray(image)
        
        r=image[:,:,0]
        g=image[:,:,1]
        b=image[:,:,2]
#        imcanny=feature.canny(imgrey)
#        imint=color.rgb2
#        daisy=feature.daisy(imgrey,  step=180, radius=58, rings=2, histograms=6,
#                         orientations=8, visualize=True)#没看懂
#        hog=feature.hog(imgrey, orientations=9, pixels_per_cell=(8, 8), 
#                        cells_per_block=(3, 3), 
#                        visualise=None, transform_sqrt=False, 
#                        feature_vector=True)#没看懂 一维矩阵 长度不定
        
        glcm=feature.greycomatrix(r+g+b, [1],[0,np.pi/4,np.pi/2,3*np.pi/4,np.pi], levels=3*256)#
        greycghg=feature.greycoprops(glcm, prop='homogeneity')/(256*256*9)#size1*5
        greycgcl=feature.greycoprops(glcm, prop='correlation')/(256*256*9)
        greycgeg=feature.greycoprops(glcm, prop='energy')/(256*256*9)
        greycgasm=feature.greycoprops(glcm, prop='ASM')/(256*256*9)
        greycgctt=feature.greycoprops(glcm, prop='contrast')/(256*256*9)
        
        
        lbp=feature.local_binary_pattern(imgrey, 8, np.pi/4)#size同图片
        plm=feature.peak_local_max(imgrey, min_distance=1)#多个坐标对
        st=feature.structure_tensor(imgrey, sigma=1, mode='constant', cval=0)#三个同图片一样大的矩阵
        ste=feature.structure_tensor_eigvals(st[0],st[1],st[2])#两个个同图片一样大的矩阵
#        hmf=feature.hessian_matrix(image, sigma=1, mode='constant', 
#                                   cval=0, order=None)#6个三通道的原图大小矩阵
        hmd=feature.hessian_matrix_det(imgrey, sigma=1)#原图大小矩阵
#        hme=feature.hessian_matrix_eigvals(hmf, Hxy=None, Hyy=None)
        si=feature.shape_index(imgrey, sigma=1, mode='constant', cval=0)#原图大小矩阵
#        ckr=feature.corner_kitchen_rosenfeld(image, mode='constant', cval=0) ##原图大小矩阵 三通道               
#        ch=feature.corner_harris(imgrey, method='k', k=0.05, eps=1e-06, sigma=1)#原图大小矩阵
#        cht=feature.corner_shi_tomasi(imgrey, sigma=1)#原图大小矩阵
#        cfs=feature.corner_foerstner(imgrey, sigma=1)#2个 #原图大小矩阵
#        csb=feature.corner_subpix(image, ch, window_size=11, alpha=0.99)
        cps=feature.corner_peaks(imgrey, min_distance=1, threshold_abs=None, 
                                 threshold_rel=0.1, exclude_border=True, indices=True, 
                                 footprint=None, labels=None)#一堆坐标值
#        cmr=feature.corner_moravec(imgrey, window_size=1)#原图大小矩阵
#        cft=feature.corner_fast(imgrey, n=12, threshold=0.15)#原图大小矩阵
        corners = feature.corner_peaks(feature.corner_fast(imgrey, 9), min_distance=1)#一堆坐标
        corts=feature.corner_orientations(imgrey, corners, octagon(3, 2))#一维矩阵长度不定
#        mtem=feature.match_template(image, template, pad_input=False,
#                                    mode='constant', constant_values=0)
#        bldg=feature.blob_dog(imgrey, min_sigma=1, max_sigma=50, 
#                              sigma_ratio=1.6, threshold=2.0, overlap=0.5)#不懂
#        bldoh=feature.blob_doh(imgrey, min_sigma=1, max_sigma=30, num_sigma=10, 
#                               threshold=0.01, overlap=0.5, log_scale=False)#不懂
#        bllog=feature.blob_log(imgrey, min_sigma=1, max_sigma=50, num_sigma=10, 
#                               threshold=0.2, overlap=0.5, log_scale=False)#不懂
        zong.append([imname,
                     greycghg[0,0],greycghg[0,1],greycghg[0,2],greycghg[0,3],greycghg[0,4],
                     greycgcl[0,0],greycgcl[0,1],greycgcl[0,2],greycgcl[0,3],greycgcl[0,4],
                     greycgeg[0,0],greycgeg[0,1],greycgeg[0,2],greycgeg[0,3],greycgeg[0,4],
                     greycgasm[0,0],greycgasm[0,1],greycgasm[0,2],greycgasm[0,3],greycgasm[0,4],
                     greycgctt[0,0],greycgctt[0,1],greycgctt[0,2],greycgctt[0,3],greycgctt[0,4],
                     np.mean(lbp),np.std(lbp),len(plm)/(len(image[:,0,0])*len(image[0,:,0])),
                     np.mean(ste[0]),np.std(ste[0]),np.mean(ste[1]),np.std(ste[1]),
                     np.mean(hmd),np.std(hmd),np.mean(si),np.std(si),
                     np.mean(cps[:,0]),np.std(cps[:,0]),np.mean(cps[:,1]),np.std(cps[:,1]),
                     np.mean(corners[:,0]),np.std(corners[:,0]),
                     np.mean(corners[:,1]),np.std(corners[:,1]),
                     len(corts)/(len(image[:,0,0])*len(image[0,:,0]))                
                     ])

csvFile = open("skimage_feature.csv", "w+")
writer = csv.writer(csvFile)
for item in zong:
#    tmp = len(item)-1#获得数据长度
#    q=item.strip('\n')        
    writer.writerow(item)
csvFile.close()

    
    
#        hlf=feature.haar_like_feature(r+g+b, 0,0,5,5,'type-3-x')
