# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:43:55 2024

@author: Administrator
"""

import numpy as np
from osgeo import gdal,gdal_array
import matplotlib.pyplot as plt
import math
import rasterio as rio
import datetime
from rasterio.plot import show
starttime = datetime.datetime.now()

#寻找相似像元
def nspi_sim_pix(img_da,x,y):     # data is the data, m is the number of categories, xy is the coordinate pixel
    data = img_da
    #data=data.astype(int)
    rows,cols=img_da.shape[1],img_da.shape[2]  
    num_bands=img_da.shape[0]
    RMSD=np.zeros((rows,cols))
    print(data.shape)
    Target= data[:-1,x,y].flatten()
    win_ = data[:-1,x-2:x+3,y-2:y+3].mean(axis=(1,2)).flatten()
    for i in range(rows):
        for j in range(cols):
            resemblance = data[:-1,i,j].flatten()
            rmsd=np.sqrt(sum(np.square(resemblance-Target))/(num_bands))
            T = np.sqrt(sum(np.square(resemblance-win_))/(num_bands))
            if rmsd <= T:
                RMSD[i,j] = 1
    return RMSD



def add_rsed_nspi(ref_da,mask,jizhun,class_data,x,y):
    a,b = jizhun[x,y],class_data[x,y] #基准，变化 
    jizhun[jizhun!=a] =0
    class_data [class_data!=b] =0
    jizhun[jizhun ==a] = 1
    class_data [class_data==b] =1
    data = ref_da*10000
    data=data.astype(int)
    num_bands=ref_da.shape[0]-1  #波段  
    rmsd_wi =np.sqrt(sum(np.square(data-data[:,x,y].reshape(num_bands+1, 1, 1)))/(num_bands))
    sim_pix =(1-mask)*class_data*jizhun
    return rmsd_wi,sim_pix

def add_weight_nspi(mask,ref_da,tar_img,rmsd_wi,sim_pix,c_x,c_y):
    rows,cols = mask.shape[0],mask.shape[1]
    D_W=np.zeros((rows,cols))
    DI = []
    RMSDI = []
    
    sim_locas = np.argwhere(sim_pix == 1)
    for sim_loc in sim_locas:
        i,j = sim_loc[0],sim_loc[1] 
        di = np.sqrt((c_x-i)**2 + (c_y-j)**2)
        DI.append(di)
        RMSDI.append(rmsd_wi[i,j])
        D_W[i,j] = di  
    di_min,di_max = min(DI),max(DI)
    rmsd_min,rmsd_max = min(RMSDI),max(RMSDI)    
    di_da = (((D_W - di_min)/(di_max - di_min))+1 )
    rm_da = (((rmsd_wi - rmsd_min)/(rmsd_max - rmsd_min))+1 ) 
    wi_data=np.zeros((rows,cols))
    wi_data_sum = 0
    for sim_loc1 in sim_locas:
        p,q    = sim_loc1[0],sim_loc1[1]  
        wi_data[p,q ] = 1/(di_da[p,q ]*rm_da[p,q ])
        wi_data_sum += 1/(di_da[p,q ]*rm_da[p,q ])
    wi =wi_data/wi_data_sum
    #print(np.sum(wi))   
    #data1为参考ref_da  ，data2为目标tar_img,rmsd相似像元掩膜sim_pix
    cankao_data = sim_pix*ref_da
    mubiao_data = sim_pix*tar_img
    band = tar_img.shape[0]
    cankao_data_av = (cankao_data.sum(axis=(1,2)))/len(np.nonzero(cankao_data[0].flatten())[0])    
    mubiao_data_av = (mubiao_data.sum(axis=(1,2)))/len(np.nonzero(mubiao_data[0].flatten())[0])
    a_da_list,b_da_list= [],[]
    for i in range (band):
        a_shang = np.sum(wi*(cankao_data[i]-cankao_data_av[i])*(mubiao_data[i]-mubiao_data_av[i]))
        a_xia = np.sum(wi*(cankao_data[i]-cankao_data_av[i])*(cankao_data[i]-cankao_data_av[i]))
        a_da =a_shang/a_xia
        a_da_list.append(a_da)
        b_da = mubiao_data_av[i] - a_da * cankao_data_av[i] 
        b_da_list.append(b_da) 
    x,y = c_x,c_y
    a_a_a = tar_img[:,x,y]#真实
    a_a_a_a = a_da_list*ref_da[:,x,y] +b_da_list#新+
    a_a_a_b = (wi * tar_img).sum(axis=(1,2))#空间结果
    aab_kongjian = (wi * ref_da).sum(axis=(1,2))#空间基准
    a_a_a_aa = a_da_list*aab_kongjian +b_da_list   #新+
    a_a_a_c = ref_da[:,x,y]+(wi * (tar_img-ref_da)).sum(axis=(1,2))#时间
    aa_zhong = ref_da[:,x,y]*a_a_a_b/aab_kongjian#基准调准结果
    a_a = (a_a_a_a+a_a_a_b+a_a_a_c)/3   
    return a_a_a_a,a_a_a_c,a_a_a_b,a_a 

def sim_pixels(ref_img,tar_img,mask,jizhun,class_data):#草考影像   目标影像     掩膜   
    
    tar_cloud = tar_img * (1-mask)
    time_result,space_result,mean_result = tar_img * (1-mask),tar_img * (1-mask),tar_img * (1-mask)
    x,y = mask.shape[0],mask.shape[1]
    
    mask_locas = np.argwhere(mask == 1)
    for mask_loca in mask_locas:
        i ,j = mask_loca[0] , mask_loca[1]                
        
        sum_sim = 0
        win_side = 25
        while sum_sim < 20:
            #cen_value = ref_img[i,j]                    
            WinLeft = max([0,j-win_side])
            WinRight = min([x-1,j+win_side])
            WinUp = max([0,i-win_side])
            WinDown = min([y-1,i+win_side])
            print(x,y)
            print(WinLeft,WinRight,WinUp,WinDown)
            ref_da = ref_img [:,WinLeft:WinRight , WinUp:WinDown ] 
            jizhun_win = jizhun[WinLeft:WinRight , WinUp:WinDown ]
            class_data_win = class_data[WinLeft:WinRight , WinUp:WinDown ]
            ref_mask = mask [  WinLeft:WinRight , WinUp:WinDown ]          
            cen_x,cen_y = i-WinUp , j-WinLeft        
            print(i,j)
            sim_mask = nspi_sim_pix(ref_da, cen_x,cen_y)
            rsed_data = add_rsed_nspi(ref_da,ref_mask,jizhun_win,class_data_win,cen_x,cen_y)   
            sim_pixel = sim_mask * (1-ref_mask) *rsed_data[1]       
            sum_sim = np.sum(sim_pixel)
            
            print('sum_sim',sum_sim)
            if win_side > x/2:
                break
            print('win_side',win_side)
            print('...............')
            win_side += 25
        tar_da = tar_img [:,WinLeft:WinRight , WinUp:WinDown ]   
        print('************')
        rmsd_wi = rsed_data[0]
        if sum_sim ==0:
            tar_cloud[:,j,i] = ref_img[:,j,i]
        else:    
            com_cen_da = add_weight_nspi(ref_mask,ref_da,tar_da,rmsd_wi,sim_pixel,cen_x,cen_y)     #mask,ref_da,tar_img,rmsd_wi,sim_pix,c_x,c_y   
            my_result,time_result1,space_result1,mean_result1 = com_cen_da[0],com_cen_da[1],com_cen_da[2],com_cen_da[3]
            
            tar_cloud[:,j,i] = my_result
        #time_result[:,j,i],space_result[:,j,i],mean_result[:,j,i] = time_result1,space_result1,mean_result1
                    
    return tar_cloud#,time_result,space_result,mean_result



maks_path = r'C:/Users/Administrator/Desktop/去云/1allmask.tif'
ref_img_path=r"C:/Users/Administrator/Desktop/去云/20221019.tif"   #参考影响
tar_img_path=r"C:/Users/Administrator/Desktop/去云/20220929.tif"   #目标影响

result_class_file = r'C:\Users\Administrator\Desktop\云检测实验数据'
jizhun_file = result_class_file+"/"+'jizhun_class.tif'
out_file = result_class_file+"/"+'result_class0201.tif'




mask =  gdal_array.LoadFile(maks_path)
ref_img = gdal_array.LoadFile(ref_img_path)
tar_img = gdal_array.LoadFile(tar_img_path)
jizhun = gdal_array.LoadFile(jizhun_file)
class_data = gdal_array.LoadFile(out_file)


mask =  gdal_array.LoadFile(maks_path)
show(mask)
ref_img = gdal_array.LoadFile(ref_img_path)
tar_img = gdal_array.LoadFile(tar_img_path)
jizhun = gdal_array.LoadFile(jizhun_file)
class_data = gdal_array.LoadFile(out_file)
sim_locas = np.argwhere(mask == 1)
rows,cols = mask.shape[1],mask.shape[1]

i,j = max(sim_locas[:,0]+1),min(sim_locas[:,0])
a,b = max(sim_locas[:,1]+1),min(sim_locas[:,1])

print(a,b,i,j)
daLeft = max([0,math.ceil((3*j-i)/2)])
daRight = min([cols-1,math.ceil((3*i-j)/2)])

daUp = max([0,math.ceil((3*b-a)/2)])
daDown = min([rows-1,math.ceil((3*a-b)/2)])

print(daLeft,daRight,daUp,daDown)

mask =  mask[daLeft:daRight,daUp:daDown] 
show(mask)
ref_img = ref_img[:,daLeft:daRight,daUp:daDown] 
tar_img1 = tar_img[:,daLeft:daRight,daUp:daDown] *mask
jizhun = jizhun[daLeft:daRight,daUp:daDown] 
class_data = class_data[daLeft:daRight,daUp:daDown] 



# =============================================================================
# abc = tar_img[:,50,50] 
# aabc = tar_img.sum(axis=(1,2))
# =============================================================================

result = sim_pixels(ref_img,tar_img1,mask,jizhun,class_data)

tar_img[:,daLeft:daRight,daUp:daDown]  =  result

img = rio.open(maks_path)  
prof = img.profile  
prof.update(count=6)
with rio.open('C:/Users/Administrator/Desktop/time.tif','w',**prof) as dst:
    dst.write(tar_img)


    
endtime = datetime.datetime.now()

#print (endtime - starttime)



































































