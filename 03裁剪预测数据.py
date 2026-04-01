# -*- coding: utf-8 -*-

from osgeo import gdal
import numpy as np
import os
from tqdm import trange

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
    if nb == 0:
        data = data[0,:,:]
    
    return data

def Boundary_Check(x,y,dis,bou):
    # x_min = x - dis
    x_max = x + dis + 1
    # y_min = y - dis
    y_max = y + dis + 1
    # if (x_min < 0) | (y_min < 0) | (x_max > bou[0]) | (y_max > bou[1]):
    if (x_max > bou[0]) | (y_max > bou[1]):
        return False
    else:
        return True
    

def mkdir(path):
    # 引入模块
    import os
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        #print(path +' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        #print(path +' 目录已存在')
        return False
    
def calTransform(ori_transform,offset_x,offset_y):
    # 读取原图仿射变换参数值
    top_left_x = ori_transform[0]  # 左上角x坐标
    w_e_pixel_resolution = ori_transform[1] # 东西方向像素分辨率
    top_left_y = ori_transform[3] # 左上角y坐标
    n_s_pixel_resolution = ori_transform[5] # 南北方向像素分辨率
    # 根据反射变换参数计算新图的原点坐标
    top_left_x = top_left_x + offset_x * w_e_pixel_resolution
    top_left_y = top_left_y + offset_y * n_s_pixel_resolution
    # 将计算后的值组装为一个元组，以方便设置
    dst_transform = (top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4], ori_transform[5])
    return dst_transform
    
def WriteTiff(im_data,inputfile,outputfile):
    raster = gdal.Open(inputfile)
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
    dataset = driver.Create(outputfile, im_width, im_height, im_bands, datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset



def cutdata(image_size,cutdata_dir,boundary_dir,overlap,resultpath,pixarr_save):
    print("begin")
    boundary_arr = LoadData(boundary_dir)[0,:,:]
    Boundary = (boundary_arr.shape)
    
    CAdata_ds = gdal.Open(cutdata_dir)
    band_num = CAdata_ds.RasterCount
    ori_transform = CAdata_ds.GetGeoTransform()
    driver = gdal.GetDriverByName('gtiff')
    pix_list = []
    i = 0
    num = Boundary[0]//100
    move_num = int((1-overlap) * image_size)
    for x in trange(0,Boundary[0],move_num):
        for y in range(0,Boundary[1],move_num):
            if Boundary_Check(x, y, image_size, Boundary): # 检查是否超出边界
                pix_list.append([x, y, i])
                if not os.path.exists(resultpath + "/images"):
                    os.makedirs(resultpath + "/images")
                Out_name = resultpath + "/images/" + str(i).zfill(5) +  ".tif"
                out_ds = driver.Create(Out_name, image_size, image_size, band_num, gdal.GDT_Float64)
                row = int(x)
                col = int(y)
                out_ds.SetGeoTransform(calTransform(ori_transform,col,row))
                for band in range(band_num):
                    out_band = out_ds.GetRasterBand(band + 1)
                    Input_band = CAdata_ds.GetRasterBand(band + 1)
                    Out_arr = Input_band.ReadAsArray(xoff=col, yoff=row, win_xsize=image_size, win_ysize=image_size)
                    out_band.WriteArray(Out_arr)
                    out_band.FlushCache()
                    del out_band, Input_band
            
                i += 1
        if x % num == 0:
            print(" {:.2f}% ".format(x/Boundary[0] * 100), end = "|")
    del CAdata_ds
    pix_arr = np.array(pix_list)
    np.save(pixarr_save, arr=pix_arr)
    print("ok")
    
image_size = 256
uppath = os.path.dirname(os.getcwd())
overlap = 0
bands = 29
years = [2015]

for startyear in years:
    print(startyear)

    cutdata_dir = uppath + "/inputdata/all{}/all.tif".format(str(bands)) # 被切的数据
    resultpath = uppath + "/simdata1_{}_{}".format(str(startyear),str(overlap))
    pixarr_save = resultpath + "/pix_list{}.npy".format(str(startyear))
    boundary_dir = uppath + "/inputdata/boundary/boundary.tif" # 研究区范围，1为研究区，0为非研究区
    cutdata(image_size,cutdata_dir,boundary_dir,overlap,resultpath,pixarr_save)






