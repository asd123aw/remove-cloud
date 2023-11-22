# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 16:49:19 2023

@author: Administrator
"""



import numpy as np
import rasterio as rio
import datetime
from osgeo import gdal,gdal_array
from KMeans非监督分类 import unsupervised_classification
from 栅格转矢量 import Raster_vector
from 裁剪矢量 import cut_shp
from maclearn_result import random_test


def result_Classification(maks_path,ref_img_path,tar_img_path,multiple_images_file,class_data_file,class_noyundata_file,class_shp_file,nocloud_shp_file,out_file,jizhun_file):   
    num_classes = 15
    num_iterations = 20
    img = rio.open(maks_path)  
    img1 = rio.open(ref_img_path)
    band = img1.count
    prof = img.profile  
    prof.update(count=1)
    prof1 = img.profile  
    prof1.update(count=band)
    prof2 = img.profile  
    prof2.update(count=band*2)
    
    mask =  gdal_array.LoadFile(maks_path)
    ref_img = gdal_array.LoadFile(ref_img_path)
    
    
    tar_img = gdal_array.LoadFile(tar_img_path)
    maks_fan = 1-mask
    a = np.r_[ref_img*maks_fan,tar_img*maks_fan]#结合成多波段变化量用于分类
    with rio.open(multiple_images_file,'w',**prof2) as dst:
        dst.write(a)
    unsupervised_classification(multiple_images_file, num_classes, num_iterations,class_data_file)#非监督分类
    unsupervised_classification(ref_img_path, num_classes, num_iterations,jizhun_file)#非监督分类
    num_class_data = gdal_array.LoadFile(class_data_file)
    kmean_data = (num_class_data + 1 )*maks_fan
    with rio.open(class_noyundata_file,'w',**prof) as dst:
        dst.write(kmean_data,1)   
    Raster_vector(class_noyundata_file,class_shp_file)#栅格转矢量
    
    cut_shp(class_shp_file,nocloud_shp_file)
    
    random_test(ref_img_path,nocloud_shp_file,out_file)


# =============================================================================
# num_classes = 20
# num_iterations = 10
# 
# maks_path = r'E:/aaa徐强毕设/遥感影像/豫中/yuzhong/无云/shp/50'
# ref_img_path=r"E:/aaa徐强毕设/遥感影像/豫中/yuzhong/无云/500/20220201T030941_20220201T031822_T49SGT_500"
# tar_img_path=r"E:/aaa徐强毕设/遥感影像/豫中/yuzhong/无云/500/20220308T030549_20220308T031828_T49SGT_500"
# 
# multiple_images_file = r'C:\Users\Administrator\Desktop\jieguo\多镜像结合.tif'
# class_data_file = r'C:\Users\Administrator\Desktop\jieguo\class_data.tif'
# class_noyundata_file = r'C:\Users\Administrator\Desktop\jieguo\class_noyundata.tif'
# class_shp_file = r'C:\Users\Administrator\Desktop\jieguo\class_shp.shp'
# nocloud_shp_file = r'C:\Users\Administrator\Desktop\jieguo\nocloud_shp.shp'
# out_file = r'C:\Users\Administrator\Desktop\jieguo\result_class0201.tif'
# jizhun_file = r'C:\Users\Administrator\Desktop\jieguo\0201_class0201.tif'
# 
# img = rio.open(maks_path)  
# prof = img.profile  
# prof.update(count=1)
# prof1 = img.profile  
# prof1.updaate(count=6)
# prof2 = img.profile  
# prof2.update(count=12)
# 
# mask =  gdal_array.LoadFile(maks_path)
# ref_img = gdal_array.LoadFile(ref_img_path)
# tar_img = gdal_array.LoadFile(tar_img_path)
# maks_fan = 1-mask
# a = np.r_[ref_img*maks_fan,tar_img*maks_fan]#结合成多波段变化量用于分类
# 
# with rio.open(multiple_images_file,'w',**prof2) as dst:
#     dst.write(a)
# unsupervised_classification(multiple_images_file, num_classes, num_iterations,class_data_file)#非监督分类
# unsupervised_classification(ref_img_path, num_classes, num_iterations,jizhun_file)#非监督分类
# num_class_data = gdal_array.LoadFile(class_data_file)
# 
# kmean_data = (num_class_data + 1 )*maks_fan
# with rio.open(class_noyundata_file,'w',**prof) as dst:
#     dst.write(kmean_data,1)
#     
# Raster_vector(class_noyundata_file,class_shp_file)#栅格转矢量
# 
# cut_shp(class_shp_file,nocloud_shp_file)
# 
# random_test(ref_img_path,nocloud_shp_file,out_file)
# 
# =============================================================================
























