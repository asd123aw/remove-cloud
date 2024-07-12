# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 19:21:41 2024

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:49:44 2024

@author: Administrator
"""

from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
import math
import rasterio as rio
import datetime
import torch
from osgeo import gdal,gdal_array
from 分类结果 import result_Classification
import os
from skimage.segmentation import slic    #SCLI算法包
from skimage.segmentation import mark_boundaries  #根据SLIC分割结果生成边界
from skimage.util import img_as_float    #读取影像数据为float型
from skimage import segmentation, io,color, feature     #颜色库
from skimage.io import imread
from skimage import graph
import matplotlib.pyplot as plt   #绘图制图
from PIL import Image


def float_int(scaled_data):
    for i in range(scaled_data.shape[2]):
        layer = scaled_data[:, :, i]
        min_val = np.min(layer)
        max_val = np.max(layer)
        scaled_data[:, :, i] = (layer - min_val) * (255 / (max_val - min_val))
        scaled_data[:, :, i] = np.clip(scaled_data[:, :, i], 0, 255).astype(np.uint8)
    return scaled_data
#def super_pixel()
def stretchImg(data, resultPath, lower_percent, higher_percent):
    n = data.shape[2]
    print(n)
    out = np.zeros_like(data, dtype=np.uint8)
    for i in range(n):
        a = 0
        b = 255
        c = np.percentile(data[:, :,i], lower_percent)
        d = np.percentile(data[:, :,i], higher_percent)
        t = a + (data[:, :,i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :,i] = t
    outImg=Image.fromarray(np.uint8(out))
    outImg.save(resultPath)
    return outImg

def linear2(ref_img,resultPath):
    #ref_img_path=r"E:/book/1019.jpg"  
    clear_img  = gdal_array.LoadFile(ref_img)
    
    data = np.array([clear_img[2],clear_img[1],clear_img[0]])
    clear_img = np.transpose(data, (1, 2, 0))
    clear_img = stretchImg(clear_img, resultPath, lower_percent=2, higher_percent=98)
    print(clear_img)
    return clear_img


def super_pixel_image(ref_img,out_file,resultPath,x,y):    
    Tpan = np.array(linear2(ref_img,resultPath))
    #plt.imshow(Tpan)
    TpanRGB=img_as_float(color.gray2rgb(Tpan)); #color.gray2rgb(Tpan)
    SingleBand = TpanRGB[:,:,1];
    TpanRGB = Tpan
    segments = slic (TpanRGB, n_segments =x*y, sigma = 5);
    #merge similar patches
    g = graph.rag_mean_color(SingleBand,segments) #define feture bands here
    seg_merg = graph.cut_threshold(segments,g,0.05)
    out2 = color.label2rgb(seg_merg,TpanRGB,kind='avg',bg_label=0)
    #plt.imshow(seg_merg)
    #plt.imshow(out2)
    image = Image.fromarray(seg_merg.astype('uint8'))
    #image.save(r'E:/book/101.png')  # 保存为PNG格式
    img1 = rio.open(ref_img)
    band1 = img1.count 
    prof = img1.profile  
    prof.update(count=3)
    seg_ = np.array([seg_merg,seg_merg,seg_merg])
    with rio.open(out_file,'w',**prof) as dst:
        dst.write(seg_)  
        
# =============================================================================
# ref_img=r"E:/book/中间文件5/1019.tif"
# resultPath = r"E:/book/中间文件5/real_img.png"       
# super_pixel_image(ref_img,resultPath)        
# =============================================================================
        
        
        
        
        
        
        
        
        
        
        
        
        