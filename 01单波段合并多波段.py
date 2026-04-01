# -*- coding: utf-8 -*-
from osgeo import gdal
import numpy as np
import os

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

def WriteTiff(im_data, im_width, im_height, im_bands, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # if len(im_data.shape) == 3:
    #     im_bands, im_height, im_width = im_data.shape
    # elif len(im_data.shape) == 2:
    #     im_data = np.array([im_data])
    # else:
    #     im_bands, (im_height, im_width) = 1, im_data.shape
        # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset
    
def compositeBands(data_dirs,im_bands,ResultPath):
    arr_list = []
    for data_dir in data_dirs:
        data_arr = LoadData(data_dir)
        arr_list.append(data_arr)
        # print(data_dir)
    all_arr = np.stack(arr_list)
    raster = gdal.Open(data_dir)
    im_width = raster.RasterXSize #栅格矩阵的列数
    im_height = raster.RasterYSize #栅格矩阵的行数
    im_geotrans = raster.GetGeoTransform() #获取仿射矩阵信息
    im_proj = raster.GetProjection() #获取投影信息
    WriteTiff(all_arr, im_width, im_height, im_bands, im_geotrans, im_proj, ResultPath) 
    

im_bands = 29 # 要合并的影像数量
path = os.path.dirname(os.getcwd())
data_path = path + "/inputdata"
# data_dirs = [] # 要合并的影像，通过一个循环写入
names =["ALT","SLO","HID","NAD","PRD","RAD","HSD","DC1","DC2","DC3","DC4","RD1","RD2","SCL","SOC","SSD","SST","TEM"] # 18
years = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010]
for i in years:
    names.append("/lulc/lulc" + str(i))

data_dirs = [] # 要合并的影像，通过一个循环写入
for name in names:
    data_dirs.append(data_path+ "/" + name + ".tif")
# data_dirs.append(data_path + "/lulc" + "/lulc" + str(year) + ".tif")
# data_dirs.append(data_path + "/distance" + "/d" + str(year) + ".tif.tif")
ResultPath = data_path + "/all" + str(im_bands)
if not os.path.exists(ResultPath):
    os.makedirs(ResultPath)

Resultfile = ResultPath + "/all" + ".tif"
compositeBands(data_dirs,im_bands,Resultfile)
    
