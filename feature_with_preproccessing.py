# -*- coding: utf-8 -*-
"""
这是最新版的图像特征提取程序，将被应用于经过或未经过预处理的国画图像上，并用于正式的图像分类，以及对比分析；
程序中标注了所有特征的文献来源和基本原理;
尽量选择贴近绘画/国画本身特点的特征
Created on Fri Dec 14 21:12:26 2018
@author: leiyu94
"""


from __future__ import division  
from skimage import color,io,feature,measure,transform,filters,morphology
import skimage.transform as st
from skimage.transform import warp, AffineTransform
from skimage.feature import corner_harris, corner_subpix, corner_peaks
import skimage.filters.rank as sfr
import skimage
import cv2,urllib
import numpy as np
import os,scipy,csv
import scipy.io as sio
import time
from collections import Counter
from skimage.morphology import disk
from math import atan,pi,log
#path1='/home/leiyu94/database_code/yuchuli - without noise and flip/'
path1='D:\\database_code\\yuchuli - without noise and flip\\'
lei='qishi','qingyou','shengji','yazhi','xiaose'
zong=[]
aa=0
bb=1

#原始数据四周补0
def pad_data(data,nei_size):
    m,n = data.shape
    t1 = np.zeros([nei_size//2,n]) 
    data = np.concatenate((t1,data,t1))
    m,n = data.shape
    t2 = np.zeros([m,nei_size//2])
    data = np.concatenate((t2,data,t2),axis=1)  
    return data

def gen_dataX(data,nei_size):
    x,y = data.shape
    m = x-nei_size//2*2;n = y-nei_size//2*2
    res = np.zeros([m*n,nei_size**2])
    print m,n
    k = 0
    for i in range(nei_size//2,m+nei_size//2):
        for j in range(nei_size//2,n+nei_size//2):
            res[k,:] = np.reshape(data[i-nei_size//2:i+nei_size//2+1,j-nei_size//2:j+nei_size//2+1].T,(1,-1))
            k += 1
    print k
    return res

#def coarseness(pic,ks):
    
    
    

for leibie in lei:
    aa=aa+1
    path=path1+leibie+'/'
    filelist=os.listdir(path)
    for imname in filelist:
        zong1=[]
        file=path+imname
        image=io.imread(file)    
        imgrey=color.rgb2gray(image)  
        imhsv=color.rgb2hsv(image)    
        print('doing picture',bb) 
        r=image[:,:,0]
        g=image[:,:,1]
        b=image[:,:,2]       
        h=imhsv[:,:,0]
        s=imhsv[:,:,1]
        v=imhsv[:,:,2] 
        
        
#'''   
#适用于国画的边界量化方法，出自论文《基于表现手法的国画分类方法研究》高峰 2017，
#该特征为邻域差异描述子(neighborhood difference discriptor)，描述工笔画和写意画在绘画技法上的差别。
#该文章还提到SIFT的KNN量化方法，在我的程序中未实现。
#因为本程序结合了其他特征，而原文只采用了两个特征，本程序将原文算法进行了大幅简化。
#该算法计算边界点周边的像素与边界点本身的相似性和差异性，以此来描述国画笔法的锋利程度，类似于图像处理中的“锐度”
#'''
#        theta=7
#        TT=15/255
#        edges = feature.canny(imgrey) 
#        im1= pad_data(imgrey,theta)           
#        data = gen_dataX(im1,theta)               
#        data1=np.delete(data,int((theta*theta-1)/2),1)            
#        im2=imgrey.flatten()            
#        edges=edges.flatten()            
#        im3=list(im2)*(theta*theta-1)            
#        im4=np.array(im3).reshape(len(edges),theta*theta-1)        
#        sub=np.abs(data1-im4)                    
#        q=[]         
#        
#        for num,rows in zip(edges,sub):               
#            q.append(rows.T*num)
#        subb=np.array(q)            
#        dif=np.histogram([x for x in subb.flatten() if x!=0],                               
#                          bins=10, density =True,range =(0,1))            
#        dif1=np.array(dif[0]/sum(dif[0]))#邻域相似性1--0-9       
#        
#        pro=1-sum(subb.T>TT)/(theta*theta-1)        
#        pro1 = [x for x in pro if x!=1]
#        pro2 = np.histogram(pro1, bins=15, density =True,range =(0,1))        
#        pro3=pro2[0]/sum(pro2[0])#邻域相似性2--10-24  
#        
##        gg=abs(np.abs(np.max(sub,1)-np.min(sub,1)))        
##        qq=edges*gg        
##        dg=[x for x in qq if x!=0]        
##        ndif=np.histogram(dg, bins=10, density =True,range =(0,1))        
##        ndif1=ndif[0]/sum(ndif[0])#邻域差异性--25-34
#        print('done:邻域特征')
#        
#        #留白计算，来源《基于艺术风格的绘画图像分类研究》杨冰 2003 
#        #本程序对原文算法进行了参数调整，使之更符合本数据库的情况。
#        #先将图像二值化，灰度图中接近白色的区域赋值为1，反之为0
#        im2grey=(imgrey>0.7)*1
#        liantong=[]        
#        #对二值画图像进行8联通区域搜索，给留白的区域大小和区域个数打标签
#        labels = measure.label(im2grey, connectivity=2)
#        #将所有联通的留白区域查找出来，将其中最大的一个所占的像素点数与图像尺寸求比例
#        #以此定义“留白”。,
#        region=measure.regionprops(labels)
#        for i in range(np.size(region)-1):        
#            liantong.append(region[i].area)   
#            liubai=np.max(liantong)
#            liubaizhanbi=liubai/np.size(imgrey)#“留白”--35
#        print('done:留白特征')
#        
##        #粗糙度、粒度特征
##        h=227
##        w=227
##        pic=imgrey
##        ks=7
##        h1=220
##        w1=220
##        picmean=np.zeros((h1,w1,ks))
##        picmean[:,:,0]=pic[0:(h1),0:(w1)]
##        for k in range(ks-1): 
##            for i in range(h1):
##                for j in range(w1):
##                    picwindow=pic[i:i+k,j:j+k]
##                    picmean[i,j,k+1]=np.mean(picwindow.flatten())
##        h2=h1-ks
##        w2=w1-ks
##        picmax3=np.zeros((h2,w2,ks))
##        for k in range(ks):
##            pic_h_deference=picmean[1+k:h2+k,1:w2,k]
##            pic_v_deference=picmean[1:h2,1+k:w2+k,k]
##            pic_d_deference=picmean[1+k:h2+k,1+k:w2+k,k]
##            pic_h_deference=np.abs(pic_h_deference-picmean[0:h2-1,0:w2-1,k])
##            pic_v_deference=np.abs(pic_v_deference-picmean[0:h2-1,0:w2-1,k])
##            pic_d_deference=np.abs(pic_d_deference-picmean[0:h2-1,0:w2-1,k])
##            picmax3[:,:,k]=np.max(np.max(pic_h_deference,pic_v_deference),pic_d_deference)     
##        picmax2,maxk=np.max(picmax3[:,2])#cunyi 
##        
##        ent=np.mean(2**maxk.flatten())
#
#        
#        #出处：https://prism.ucalgary.ca/handle/1880/51900
#        #灰度共生矩阵一种通过使用灰度空间的相关特性来描述纹理的常用方法,
#        #为了生成纹理特征,将亮度量化为８个等级，步长定为５个像素间隔,
#        #并且在水平、垂直、４５°和１３５°４个方向上分别构建共生矩阵,
#        #其中包括３６维的局部纹理特征和描述留白特点的１维全局特征．
##        glcm=feature.greycomatrix(r, [5],[0,np.pi/4,np.pi/2,3*np.pi/4,np.pi], levels=256)
##        
##        greycghg=feature.greycoprops(glcm, prop='homogeneity')/(256)
##        greycghg=greycghg/np.sum(greycghg)#同质性，反应纹理局部变化的多少--36-40
##        
###        greycgcl=feature.greycoprops(glcm, prop='correlation')/(256)，
##        #求解时发现五个值都一样，故略去。
###        greycgcl=greycgcl/np.sum(greycgcl)#相关，各值的大小关系决定纹理的方向--41-45
##        
##        greycgeg=feature.greycoprops(glcm, prop='energy')/(256)
##        greycgeg=greycgeg/np.sum(greycgeg)#能量，反应纹理的均匀程度和粗细度--41-45
##        
##        greycgctt=feature.greycoprops(glcm, prop='contrast')/(256)
##        greycgctt=greycgctt/np.sum(greycgctt)#对比度，反应清晰度和沟纹深浅--46-50
##        print('done:灰度共生矩阵')
#        
#        
#        
#        #出处《监督式异构稀疏特征选择的国画分类和预测》 王征 2013
#        #将整个颜色空间的 Ｈ，Ｓ，Ｖ３个维度中的 Ｈ 维平均分为１６份，
#        #Ｓ 维平均分为８份，Ｖ 维平均分为８份，每个子空间对应直方图中的
#        #一维（ｂｉｎ），然后统计落在直方图每一维对应的子空间内的像素数，
#        #得到颜色直方图
#        hhist,bi=np.histogram(h, bins=16, density =True,range =(0,1))
##        shist,bi=np.histogram(s, bins=8, density =True,range =(0,1))
#        vhist,bi=np.histogram(v, bins=8, density =True,range =(0,1))
#        hhist=hhist/sum(hhist)#h通道直方图--
##        shist=shist/sum(shist)#s通道直方图--
#        vhist=vhist/sum(vhist)#v通道直方图--
#        print('done:HSV特征')
##        
##        #来源《中国画的特征提取及分类》陈俊杰 2008
##        #使用霍夫变换（Houph）进行直线检测：ln为直线段的个数，LP为平行线个数的归一值
#        edges = feature.canny(imgrey)
#        lines = st.probabilistic_hough_line(edges, threshold=10, line_length=5,line_gap=3)        
#        ln=len(lines)# 直线段的个数
#        cc=[]
#        for dian in lines:
#            hh=np.array(dian[0])-np.array(dian[1])
#            cc.append((hh[0]**2+hh[1]**2)**0.5)
#        lc=sum(cc)
#        le=sum(edges.flatten())
#        lpp=lc/le
#        g = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#        ret,g= cv2.threshold(g,145,255,0)        
#        a,b,c = cv2.findContours(g,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#        cnt = b[0]
#        i = cv2.drawContours(image,b, -1, (255,0,0),1)
#        l = cv2.arcLength(cnt,True)
#        chull = morphology.convex_hull_object(edges)
#        labels = measure.label(chull,connectivity=None)
#        label_att = measure.regionprops(labels) 
#        prmt=[]
#        area=[]
#        for obj in label_att:
#            prmt.append(obj.perimeter)
#            area.append(obj.area)
#        pof=sum(prmt)
#        ar=sum(area)
#        geshu=len(label_att)
#        ratio=[]
#        angle=[]
#        for dian in lines:
#            hh=np.array(dian[0])-np.array(dian[1])
#            ratio.append(hh[1]/hh[0])
#            angle.append(atan(hh[1]/hh[0]))
#        
#        lp1=Counter(ratio)
#        lp,bi=np.histogram(angle, bins=15, density =True,range =(-pi/2,pi/2))
#        lp=lp/sum(lp)
#         #线段夹角直方图
#        print('done:线段特征')   
###        
#        #rule of thirds feature of h channel,hsv三个通道的三分法则计算
##        来源：2006 Studying Aesthetics in Photographic Images Using a Computational Approach
##        ihsv=imhsv
##        ro3h=9*np.mean(ihsv[len(h[:,0])//3:2*len(h[:,0])//3,
##                            len(h[0,:])//3:2*len(h[0,:])//3,
##                            0])
##        ro3s=9*np.mean(ihsv[len(s[:,0])//3:2*len(s[:,0])//3,
##                            len(s[0,:])//3:2*len(s[0,:])//3,
##                            0])
##        ro3v=9*np.mean(ihsv[len(v[:,0])//3:2*len(v[:,0])//3,
##                            len(v[0,:])//3:2*len(v[0,:])//3,
##                            0])      
##        print('done:三分法特征')
###        
###        #
###        
###        
###        
#        #来源：Koenderink, J. J. & van Doorn, A. J., “Surface shape and curvature scales”, 
#        #Image and Vision Computing, 1992, 10, 557-564. DOI:10.1016/0262-8856(92)90076-F
#        #形状指数，用来度量线条曲率，弯曲的线较多的话则值较大
#        shape=skimage.feature.shape_index(imgrey, sigma=1, mode='constant', cval=0)
#        shf=(shape*edges)
#        shf=shf.flatten()
#        shf1=[]
#        for num in shf:
#            if num != 0:
#                shf1.append(num)
#        
#        histsi,bi=np.histogram(shf1, bins=8, density =True,range =(-1,1))
#        histsi=histsi/sum(histsi)
#        
##        bwe=np.sum(shape*edges)/np.sum(edges)       
#        #
#        print('done:形状指数特征')#
##        
##        
#        #暗通道直方图
#        #来源：Single Image Haze Removal Using Dark Channel Prior，2009
#        dark1=np.min(image,2)
#        dark1=dark1/255       
#        dst =sfr.minimum(dark1, disk(5))        
##        fodc=np.mean(dst)#        
#        histdc,bi=np.histogram(dst, bins=10, density =True,range =(0,255))
#        histdc=histdc/sum(histdc)#--
#        print('done:暗通道特征')
#        
#        #来源：Photo and Video Quality Evaluation:Focusing on the Subject， 2009
#        #ECCV 2008, Part III, LNCS 5304, pp. 386–399, 2008，颜色简明度--
#        ghue=h*(s>0.2)*(v>0.15)*(v<0.95)
#        ghue=ghue.flatten()
#        ghue1=[]        
#        for num in ghue:
#            if num!=0:
#                ghue1.append(num)
#        histgh,bi=np.histogram(ghue, bins=10, density =True,range =(0,1))
#        histgh=histgh/sum(histgh)
#        histgh1=[]
#        for num in histgh:
#            if num!=0:
#                histgh1.append(num)
#        simp=10-sum(histgh>0.05*min(histgh1))
#        print('done:颜色简明度特征')
###        
###        
###        #角点特征
###        tform = AffineTransform(scale=(1.3, 1.1), rotation=1, shear=0.7, 
###                                translation=(210, 50))
###        imagec = warp(imgrey, tform.inverse, output_shape=(350, 350))
###        coords = corner_peaks(corner_harris(imagec), min_distance=5)
###        noc=len(coords)
###        print('done:角点特征')
###        coords_subpix = corner_subpix(imagec, coords, window_size=13)
##        
#        #求图像曲率半径，即一阶导数，来源高等数学 ,以2为底绘制对数尺度的直方图。9维，
#        Ix=cv2.Sobel(imgrey,cv2.CV_64F,1,0,ksize=5)
#        Iy=cv2.Sobel(imgrey,cv2.CV_64F,0,1,ksize=5)
#        Ixy=cv2.Sobel(Ix,cv2.CV_64F,0,1,ksize=5)
#        Ixx=cv2.Sobel(Ix,cv2.CV_64F,1,0,ksize=5)
#        Iyy=cv2.Sobel(Iy,cv2.CV_64F,0,1,ksize=5)
#        Idown=(Ix*Ix+Iy*Iy)**1.5;
#        Iup=(Iy*Iy*Ixx-2*Ix*Ixy*Iy+Ix*Ix*Iyy);
#        K=Idown/Iup
#        K=K*edges
#        K[K<1]=0
#        K[np.isnan(K)]=0        
#        K=np.log2(K)  
#        K1=K
#        K[np.isnan(K)]=0
#        cr=[]
#        K=K.flatten()
#        K1[np.isneginf(K1)]=0
#        for num in K:
#            if num!=0:
#                if not np.isnan(num):             
#                    cr.append(num)
#        histcr,bi=np.histogram(cr, bins=9, density =True,range =(0,9))
#        histcr=histcr/sum(histcr)
#        
#        print('done:曲率特征')
##        
##        
#        #模糊度：来源The Design of High-Level Features for Photo Quality Assessment
#        fft2 = np.fft.fft2(imgrey)
#        iqt=np.sum([fft2>5])/227/227           
#        print('done:图像清晰度特征')
##        
##        
#        #灰度直方图
#        histgr,bi=np.histogram(imgrey, bins=16, density =True,range =(0,1))
#        histgr=histgr/sum(histgr)
#        print('done:灰度直方图特征')
##        
#        #红绿蓝直方图
#        b=image[:,:,2]  
#        histr,bi=np.histogram(r, bins=10, density =True,range =(0,256))
#        histg,bi=np.histogram(g, bins=10, density =True,range =(0,256))
#        histb,bi=np.histogram(b, bins=10, density =True,range =(0,256))
#        histr=histr/sum(histr)
#        histg=histg/sum(histg)
#        histb=histb/sum(histb)
#        print('done:彩色直方图特征')
##        
#        #边界统计特征
#        tl=np.mean(edges[0:113,0:113])#左上
#        bl=np.mean(edges[0:113,113:226])#左下
#        tr=np.mean(edges[113:226,0:113])#右上
#        br=np.mean(edges[113:226,113:226])#右下
#        print('done:边界统计特征')
##        
#        #对比度
#        histrr,bi=np.histogram(r, bins=255, density =True,range =(0,255))
#        histgg,bi=np.histogram(g, bins=255, density =True,range =(0,255))
#        histbb,bi=np.histogram(b, bins=255, density =True,range =(0,255))
#        histt= histrr+ histgg+ histbb
#        histt=histt/sum(histt)
#        for i in range(255):
#            con=sum(histt[0:i])
#            if con>0.01 :
#                con1=i
#                break
#        for i in range(255):  
#            con=sum(histt[0:i])  
#            if con>0.09 :   
#                con2=i     
#                break        
#        contrast=con2-con1
##        
#        
#        #骨架特征
#        skeleton=morphology.skeletonize(1-imgrey)
#        skeleton=morphology.skeletonize(imgrey)
#        
#        #角点
        corner=feature.corner_fast(imgrey)
        histcn,bi=np.histogram(r, bins=255, density =True,range =(0,255))
        
        #总矩阵赋值
        zong1.append(imname)
#        zong1.append(float(aa))      
#        
#        for num in dif1:
#            zong1.append(float(num))#邻域相似性1--0-9       
#        for num in pro3:
#            zong1.append(float(num))#邻域相似性2--10-24  
#        zong1.append(float(liubaizhanbi))  #  “留白”--25
#        for num in hhist:#颜色直方图--26-41
#            zong1.append(float(num))
#        for num in vhist:#亮度直方图--42-49
#            zong1.append(float(num))     
##        
#        zong1.append(float(ln))#直线段个数--50
##        
#        for num in histdc:#暗通道--51-60
#            zong1.append(float(num))
##        
#        zong1.append(float(simp))  #颜色简明度--61
##        zong1.append(float(noc)) #角点个数--62
#        for num in histcr:#曲率半径直方图，最大为30--62-70
#            zong1.append(float(num))
#        zong1.append(float(iqt))#图像清晰度特征 71
#        for num in histgr:#灰度直方图，最大为30--72-87
#            zong1.append(float(num))
#        for num in histr:#红色直方图，最大为255--88-97
#            zong1.append(float(num))    
#        for num in histg:#绿色直方图，最大为255--98-107
#            zong1.append(float(num))
#        for num in histb:#蓝色直方图，最大为255--108-117
#            zong1.append(float(num))        
#        zong1.append(float(np.mean(edges)))#边界复杂度--118
#        zong1.append(float(np.mean(edges[0:113,0:113])))#边界复杂度左上--119
#        zong1.append(float(np.mean(edges[0:113,113:226])))#--120 边界复杂度左下
#        zong1.append(float(np.mean(edges[0:113,113:226])))#--121 边界复杂度右上
#        zong1.append(float(np.mean(edges[113:226,113:226])))#边界复杂度右下--122
#        zong1.append(float(np.mean(edges[56:168,56:168])))#边界复杂度中间--123
#        zong1.append(tl/br) #左上/右下--124
#        zong1.append(tr/bl)#右上/左下--125
#        zong1.append((tr+tl)/(br+bl)) #上/下--126
#        zong1.append((tr+br)/(tl+bl)) #左/右--127
#        zong1.append(float(contrast))#对比度--128
#        zong1.append(np.mean(skeleton))#反向边界和原图的骨架复杂度--135-136
#        zong1.append(lpp)#直线段长度只和/所有轮廓线像素数,第一个为坐标轴长度，第二个为斜线长度--137-138
#        zong1.append(pof)#所有凸包周长139
#        zong1.append(ar)#所有凸包面积140
#        zong1.append(float(geshu))#所有凸包个数141
        #显著性区域的个数、面积均值和方差 4 dimension129-134
        #粗糙度  1维      
        
        
        zong.append(zong1)
        print('done:',bb)
        bb=bb+1
    print('done:',leibie)


        
csvFile = open(time.strftime("%Y-%m-%d_%H_%M_%S_", time.localtime())+str(len(zong1)+2)+"_features_with_preprocdessing_withfilename.csv", "wb")

file=time.strftime("%Y-%m-%d_%H_%M_%S_", time.localtime())+str(len(zong1)+2)+"_features_with_preprocdessing.csv"
writer = csv.writer(csvFile)
for item in zong:
    writer.writerow(item)
#csvfile.close()

