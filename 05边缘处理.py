# -*- coding: utf-8 -*-

from osgeo import gdal
import numpy as np
import math
import os
import rasterio as rio
import xarray as xr
from tqdm import tqdm, trange

def LoadData(filename):
    file = gdal.Open(filename)
    if file == None:
        print(filename + " can't be opened!")
        return
    nb = file.RasterCount

    L = []
    for i in range(1, nb + 1):
        band = file.GetRasterBand(i)
        background = band.GetNoDataValue()
        data = band.ReadAsArray()
        data = data.astype(np.float32)
        index = np.where(data == background)
        data[index] = 0
        L.append(data)
    data = np.stack(L,0)
    if nb == 1:
        data = data[0,:,:]
    return data

def WriteTiff(im_data,inputdir, path):
    raster = gdal.Open(inputdir)
    im_width = raster.RasterXSize #栅格矩阵的列数
    im_height = raster.RasterYSize #栅格矩阵的行数
    im_bands = raster.RasterCount #波段数
    im_geotrans = raster.GetGeoTransform()#获取仿射矩阵信息
    im_proj = raster.GetProjection()#获取投影信息

    
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
        # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset
    
size = 256
startyear = 2015
arrpath = "../transunet_result/Suitability_1_" + str(startyear) + "_lb.tif"
arr = LoadData(arrpath)

rows = arr.shape[0]//size
cols = arr.shape[1]//size

# 处理行
for row in range(rows + 1):
    # 处理0等行
    for col in range(arr.shape[1]):
        arr[row * size][col] = arr[row * size + 1][col]
    # 处理255等行
    if row != rows:
        for col in range(arr.shape[1]):
            arr[row * size + size - 1][col] = arr[row * size + size - 2][col]
                
# 处理列
for col in range(cols + 1):
    # 处理0等行
    for row in range(arr.shape[0]):
        if row%size == 0: # 如果在左上角就等于后面的
            arr[row][col* size] = arr[row + 1][col * size  + 1]
        elif row%size == 255: # 如果在左下角就等于后面的
            arr[row][col* size] = arr[row - 1][col * size  + 1]
        else:
            arr[row][col* size] = arr[row][col * size + 1]
    # 处理255等行
    if col != cols:
        for row in range(arr.shape[0]):
            if row%size == 255:  # 如果在右下角就等于后面的
                arr[row][col * size + size - 1] = arr[row - 1][col * size + size - 2]
            elif row%size == 0: # 如果在右上角就等于后面的
                arr[row][col * size + size - 1] = arr[row + 1][col * size  - 2]
            else:
                arr[row][col * size + size - 1] = arr[row][col * size + size - 2]
                
resultpath = "../transunet_result/Suitability_1_" + str(startyear) + "_lb_1.tif"
WriteTiff(arr, arrpath, resultpath)

            


