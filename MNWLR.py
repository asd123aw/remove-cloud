# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:25:37 2023

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:12:31 2023

@author: Administrator
"""

import numpy as np

import rasterio as rio
import datetime
from osgeo import gdal,gdal_array
from Classification_results import result_Classification
import os
import mwlr_fun as mf

starttime = datetime.datetime.now()

#def wlr_sim_pix(mask,ref_da,x,y,class_data,jizhun):     # data is the data, m is the number of categories, xy is the coordinate pixel
maks_path = r'C:/Users/Administrator/Desktop/ENVI/0805mask.tif'
ref_img_path=r"C:/Users/Administrator/Desktop/ENVI/1jizhun"
tar_img_path=r"C:/Users/Administrator/Desktop/ENVI/20220805"
out_path = r"C:/Users/Administrator/Desktop/ENVI/0805cloud.tif"

result_class_file = r'C:/Users/Administrator/Desktop/50change_file'
os.makedirs(result_class_file, exist_ok=True)
multiple_images_file = result_class_file+"/"+'多镜像结合.tif'
class_data_file = result_class_file+"/"+'class_data.tif'
class_noyundata_file =result_class_file+"/"+ r'class_noyundata.tif'
class_shp_file = result_class_file+"/"+'class_shp.shp'
nocloud_shp_file = result_class_file+"/"+'nocloud_shp.shp'
out_file = result_class_file+"/"+'result_class0201.tif'
jizhun_file = result_class_file+"/"+'jizhun_class.tif'
result_Classification(maks_path,ref_img_path,tar_img_path,multiple_images_file,  
                      class_data_file,class_noyundata_file,class_shp_file,nocloud_shp_file,out_file,jizhun_file)
mask =  gdal_array.LoadFile(maks_path)
clear_da = gdal_array.LoadFile(ref_img_path)
cloud_img = gdal_array.LoadFile(tar_img_path)
class_data = gdal_array.LoadFile(out_file)
refer_data = gdal_array.LoadFile(jizhun_file)

band=clear_da.shape[0]#波段
rows,cols=clear_da.shape[1],clear_da.shape[2]


#   开始计算
# 第一步     找到云的位置做循环
# 第二步     寻找相似像元
# 第三步     计算权重

mask_locas = np.argwhere(mask == 1) 
cloud_sum_num = mask_locas.shape[0]

res_classic,res_Benchmark_das = mf.sim_classic(class_data)
ref_classic,rdf_Benchmark_das = mf.sim_classic(class_data)



# =============================================================================
# import threading
# 
# # 共享变量，用于存储计算的结果
# total_sum = cloud_img * (1-mask)
# 
# # 定义一个函数用于计算从start到end的和      
# def Calculation_results(mask_locas,a,b):
#     global total_sum
#     local_sum = cloud_img * (1-mask)
#     for i in range(a, b + 1):
#         print(mask_locas[i])
#         x, y = mask_locas[i][0],mask_locas[i][1]
#         sim_mask, rmsd_wi, wei_distance,x,y = mf.Dif_goals_Rain(clear_da, x, y)
#         clas_of_result = class_data[x,y]
#         clas_of_refer = refer_data[x,y]
#         res_classic,res_Benchmark_das = mf.sim_classic(class_data)
#         ref_classic,ref_Benchmark_das = mf.sim_classic(refer_data)
#         index_res = res_Benchmark_das.index(clas_of_result)
#         index_ref = ref_Benchmark_das.index(clas_of_refer)
#         loc_res,loc_ref = res_classic[index_res],ref_classic[index_ref]
#         wi = mf.Calculate_weight(clear_da,x,y,rows,cols,loc_res,loc_ref,mask)
#         cloud_da ,spatial_image ,time_image = mf.Regression_parameters(clear_da,cloud_img,x,y,band,loc_res,loc_ref,mask,rows,cols)
#         local_sum += cloud_da * mask
#     with lock:
#         total_sum += local_sum    
#     
# # 创建多线程
# num_threads = 100
# threads = []
# lock = threading.Lock()
# range_per_thread = cloud_sum_num // num_threads  # 每个线程计算的范围
# 
# for i in range(num_threads):
#     start = i * range_per_thread + 1
#     end = (i + 1) * range_per_thread if i < num_threads - 1 else cloud_sum_num
#     thread = threading.Thread(target=Calculation_results, args=(mask_locas,start, end))
#     threads.append(thread)
#     thread.start()
# 
# # 等待所有线程完成
# for thread in threads:
#     thread.join()
# 
# print("Total Sum:", total_sum)
# 
# =============================================================================
for mask_loc in mask_locas:
    print(mask_loc)
    x,y = mask_loc[0],mask_loc[1]
    sim_mask, rmsd_wi, wei_distance,x,y = mf.Dif_goals_Rain(clear_da, x, y)
    clas_of_result = class_data[x,y]
    clas_of_refer = refer_data[x,y]
    res_classic,res_Benchmark_das = mf.sim_classic(class_data)
    ref_classic,ref_Benchmark_das = mf.sim_classic(refer_data)
    index_res = res_Benchmark_das.index(clas_of_result)
    index_ref = ref_Benchmark_das.index(clas_of_refer)
    loc_res,loc_ref = res_classic[index_res],ref_classic[index_ref]
    wi = mf.Calculate_weight(clear_da,x,y,rows,cols,loc_res,loc_ref,mask)
    cloud_da ,spatial_image ,time_image = mf.Regression_parameters(clear_da,cloud_img,x,y,band,loc_res,loc_ref,mask,rows,cols)
    cloud_img[:,x,y] = cloud_da   
    
img = rio.open(maks_path) 
img1 = rio.open(ref_img_path)
band = img1.count 
prof = img.profile  
prof.update(count=band)

with rio.open(out_path,'w',**prof) as dst:
    dst.write(cloud_img)  

endtime = datetime.datetime.now()
print (endtime - starttime)













