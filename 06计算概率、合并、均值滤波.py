# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 23:11:46 2021

@author: Xhpan
"""

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

def GetExtent(in_fn):
    ds=gdal.Open(in_fn)
    geotrans=list(ds.GetGeoTransform())
    xsize=ds.RasterXSize 
    ysize=ds.RasterYSize
    min_x=geotrans[0]
    max_y=geotrans[3]
    max_x=geotrans[0]+xsize*geotrans[1]
    min_y=geotrans[3]+ysize*geotrans[5]
    ds=None
    return min_x,max_y,max_x,min_y

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

def get_extent(tiffileList):
    filename = tiffileList
    rioData = rio.open(filename[0])
    left = rioData.bounds[0]
    bottom = rioData.bounds[1]
    right = rioData.bounds[2]
    top = rioData.bounds[3]
    for ifile in filename[1:]:
        rioData = rio.open(ifile)
        left = min(left, rioData.bounds[0])
        bottom = min(bottom, rioData.bounds[1])
        right = max(right, rioData.bounds[2])
        top = max(top, rioData.bounds[3])
    return left, bottom, right, top, filename

def getRowCol(left, bottom, right, top,resolution):
    cols = int((right - left) / resolution)
    rows = int((top - bottom) / resolution)
    return cols, rows


def to0(output_name,boundary_name):
    arr1 = LoadData(output_name)
    arr2 = LoadData(boundary_name)
    arr1[np.where(arr2 == 0)] = 0
    WriteTiff(arr1,boundary_name,output_name)

def UnifiedLineNumber(in_fn,criterion_fn,output_name,resolution):
    arr = xr.open_rasterio(criterion_fn).data[0,:,:]
    im_data = np.zeros(arr.shape)
    WriteTiff(im_data,criterion_fn, "zero.tif")
    
    tiffileList = ["zero.tif",in_fn]
    left, bottom, right, top, filename = get_extent(tiffileList)
    cols, rows= getRowCol(left, bottom, right, top,resolution)
    n_bands = 1
    arr = np.zeros((n_bands, rows, cols), dtype=np.float64)
    # 打开一个tif文件
    in_ds = gdal.Open(filename[0])
    
    # 新建一个tif文件
    driver = gdal.GetDriverByName('gtiff')
    out_ds = driver.Create(output_name, cols, rows,1,gdal.GDT_Float32)
    # 设置tif文件的投影
    out_ds.SetProjection(in_ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    # 设置新tif文件的地理变换
    gt = list(in_ds.GetGeoTransform())
    gt[0], gt[3] = left, top
    out_ds.SetGeoTransform(gt)
    # 对要拼接的影像进行循环读取
    for ifile in tiffileList:
        in_ds = gdal.Open(ifile)
        # 计算新建的tif文件及本次打开的tif文件之间的坐标漂移
        trans = gdal.Transformer(in_ds, out_ds, [])
        # 得到偏移起始点
        success, xyz = trans.TransformPoint(False, 0, 0)
        x = round(xyz[0])
        y = round(xyz[1])
        
        # x, y, z = map(int, xyz)
        # 读取波段信息
        fnBand = in_ds.GetRasterBand(1)
        data = fnBand.ReadAsArray()
    
        out_band.WriteArray(data, x, y)
    del out_band, out_ds
    os.remove("zero.tif")
    
    
def mosaic(in_path,output_name,arr_files):
    # os.chdir(in_files)
    in_files = os.listdir(in_path)
    in_fn=in_files[0]
    #获取待镶嵌栅格的最大最小的坐标值
    min_x,max_y,max_x,min_y=GetExtent(in_path + '/' + in_fn)
    pbar = tqdm(in_files[1:])
    for in_fn in pbar:
        minx,maxy,maxx,miny=GetExtent(in_path + '/' + in_fn)
        min_x=min(min_x,minx)
        min_y=min(min_y,miny)
        max_x=max(max_x,maxx)
        max_y=max(max_y,maxy)
    #计算镶嵌后影像的行列号
    in_ds=gdal.Open(in_path + '/' + in_files[0])
    geotrans=list(in_ds.GetGeoTransform())
    width=geotrans[1]
    height=geotrans[5]
    
    columns=math.ceil((max_x-min_x)/width)
    rows=math.ceil((max_y-min_y)/(-height))
    in_band=in_ds.GetRasterBand(1)
    
    driver=gdal.GetDriverByName('GTiff')
    
    out_ds=driver.Create(output_name,columns,rows,1,in_band.DataType)
    out_ds.SetProjection(in_ds.GetProjection())
    geotrans[0]=min_x
    geotrans[3]=max_y
    out_ds.SetGeoTransform(geotrans)
    out_band=out_ds.GetRasterBand(1)
    #定义仿射逆变换
    inv_geotrans=gdal.InvGeoTransform(geotrans)
    #开始逐渐写入
    pbar = tqdm(in_files)
    for in_fn in pbar:
        in_ds=gdal.Open(in_path + '/' + in_fn)
        in_gt=in_ds.GetGeoTransform()
        #仿射逆变换
        offset=gdal.ApplyGeoTransform(inv_geotrans,in_gt[0],in_gt[3])
        x,y=map(int,offset)
        # print(x,y)
        trans=gdal.Transformer(in_ds,out_ds,[])#in_ds是源栅格，out_ds是目标栅格
        success,xyz=trans.TransformPoint(False,0,0)#计算in_ds中左上角像元对应out_ds中的行列号
        x,y,z=map(int,xyz)
        # print(x,y,z)
        arr = np.load(arr_files + "/" + in_fn[:-4] + ".npy")
        # arr = arr[1::]/(arr[1::] + 1)
        data = arr[1]
        out_band.WriteArray(data,x,y)#x，y是开始写入时左上角像元行列号
        # print(in_fn)
    del in_ds,out_band,out_ds


def aveFilter(inputdir, resultpath):
    arr = LoadData(inputdir)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i][j] == 0.5:
                nums = []
                if arr[i - 1][j - 1] != 0.5:
                    nums.append(arr[i - 1][j - 1])
                elif arr[i - 1][j] != 0.5:
                    nums.append(arr[i - 1][j])
                elif arr[i - 1][j + 1] != 0.5:
                    nums.append(arr[i - 1][j + 1])
                elif arr[i][j - 1] != 0.5:
                    nums.append(arr[i][j - 1])
                elif arr[i][j + 1] != 0.5:
                    nums.append(arr[i][j + 1])
                elif arr[i + 1][j - 1] != 0.5:
                    nums.append(arr[i + 1][j - 1])
                elif arr[i + 1][j] != 0.5:
                    nums.append(arr[i + 1][j])
                elif arr[i + 1][j + 1] != 0.5:
                    nums.append(arr[i + 1][j + 1])
                num = sum(nums) / len(nums)
                arr[i][j] = num

    WriteTiff(arr, inputdir, resultpath)
    
years = [2015]
resolution = 250

for startyear in years:
    print(startyear)
    path = "../simdata1_{}_0".format(str(startyear))
    in_path = path + "/images"
    arr_files = path + "/np_save"
    output_name = path + "/Suitability.tif"
    boundary_name = "../inputdata/boundary/boundary.tif"
    mosaic(in_path,output_name,arr_files)
    
    criterion_fn = "../origindata/urban/urban2/urban" +str(startyear) + ".tif"
    output_name1 ="../transunet_result/Suitability_1_{}.tif".format(str(startyear))
    UnifiedLineNumber(output_name,criterion_fn,output_name1,resolution)
    to0(output_name1,boundary_name)

    print("开始均值滤波...")
    inputdir = "../transunet_result/Suitability_1_" + str(startyear) + ".tif"
    resultpath = "../transunet_result/Suitability_1_" + str(startyear) + "_lb.tif"
    aveFilter(inputdir, resultpath)
    boundary_name = "../inputdata/boundary/boundary.tif"
    to0(resultpath, boundary_name)
print("ok")