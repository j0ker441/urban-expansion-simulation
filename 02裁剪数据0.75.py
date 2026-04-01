# -*- coding: utf-8 -*-

from osgeo import gdal
import numpy as np
import os
import random 
from tqdm import trange
import shutil

def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

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

def rotarr(filepath):
    print("裁剪结束，开始旋转....")
    filenames = os.listdir(filepath + "/labels")
    for i in trange(len(filenames)):
        filename = filenames[i]
        png = LoadData(filepath + "/labels/" + filename)
        
        for rot_num in range(1,4):
            png1 = np.rot90(png, rot_num)
            outputfile = filepath + "/labels/" + filename[:-4] + "_" + str(rot_num) + ".tif"
            WriteTiff(png1,filepath + "/labels/" + filename,outputfile)
            
            jpg = LoadData(filepath + "/images/" + filename)
            new_jpg = []
            for j in jpg:
                new_jpg.append(np.rot90(j, rot_num))
            jpg1 = np.array(new_jpg)
            outputfile = filepath + "/images/" + filename[:-4] + "_" + str(rot_num) + ".tif"
            WriteTiff(jpg1,filepath + "/images/" + filename,outputfile)


def tounet(dataset_path):
    random.seed(0)
    segfilepath= dataset_path + '/labels' # 标签数据路径
    saveBasePath= dataset_path + "/Segmentation" # 存放输出txt的路径
     
    if not os.path.exists(saveBasePath):
    	os.makedirs(saveBasePath)
        
    train_percent= 0.9 # tarin：trainval数据 = 9 ： 1
    
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".tif"):
            total_seg.append(seg)
            
    num=len(total_seg)  
    list=range(num) 
    tr=int(num *train_percent) 
    trainval= random.sample(list,tr) 
    train=random.sample(trainval,tr)  
     
    print("train and trainval size",tr)
    ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
     
    for i in list:  
        name=total_seg[i][:-4]+'\n'   
        if i in train:  
            ftrain.write(name)  
        else:  
            ftrainval.write(name)  
            
    ftrain.close()  
    ftrainval.close()  


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
    
        
def get_extent(fn):
    '''Returns min_x, max_y, max_x, min_y'''
    ds = gdal.Open(fn)
    gt = ds.GetGeoTransform()
    return (gt[0], gt[3], gt[0] + gt[1] * ds.RasterXSize,
            gt[3] + gt[5] * ds.RasterYSize)
    

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



def cutdata(image_size,label_dir,cutdata_dir,boundary_dir,overlap,resultpath,pixarr_save):
    print("begin")
    # urban_arr = xr.open_rasterio(label_dir).data[0,:,:]
    # cutdata__arr = xr.open_rasterio(cutdata_dir).data[0,:,:]
    boundary_arr = LoadData(boundary_dir)
    Boundary = (boundary_arr.shape)
    
    CAdata_ds = gdal.Open(cutdata_dir)
    band_num = CAdata_ds.RasterCount
    ori_transform = CAdata_ds.GetGeoTransform()
    driver = gdal.GetDriverByName('gtiff')
    pix_list = []
    i = 0
    num = Boundary[0]//100
    
    # 裁剪标签数据
    label_ds = gdal.Open(label_dir)
    
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
                
                # 裁剪标签，单波段
                if not os.path.exists(resultpath + "/labels"):
                    os.makedirs(resultpath + "/labels")
                Out_name1 = resultpath + "/labels/" + str(i).zfill(5) +  ".tif"
                out1_ds = driver.Create(Out_name1, image_size, image_size, 1, gdal.GDT_Float64)
                out1_ds.SetGeoTransform(calTransform(ori_transform,col,row))
                out1_band = out1_ds.GetRasterBand(1)
                Input1_band = label_ds.GetRasterBand(1)
                out1_arr = Input1_band.ReadAsArray(xoff=col, yoff=row, win_xsize=image_size, win_ysize=image_size)
                out1_band.WriteArray(out1_arr)
                out1_band.FlushCache()
                del out1_band, Input1_band
                
                i += 1
    del CAdata_ds,label_ds
    pix_arr = np.array(pix_list)
    np.save(pixarr_save, arr=pix_arr)
    print("ok")
    
image_size = 256
overlap = 0.75
uppath = os.path.dirname(os.getcwd())
bands = 29

startyear = 2015 # 训练数据中预测的年份
print(startyear)
urbanpath = uppath + "/origindata/urban/urban2/urban"
urban1 = LoadData(urbanpath + str(startyear) + ".tif")
# urban2 = xr.open_rasterio(urbanpath + endyear + ".tif").data[0,:,:]
label_arr = urban1 # 0非城市，1城市扩展区域，2城市不变区域

cutdata_dir = uppath + "/inputdata/all{}/all.tif".format(str(bands)) # 被切的数据
boundary_dir = uppath + "/inputdata/boundary/boundary.tif" # 研究区范围，1为研究区，0为非研究区
resultpath = uppath + "/traindata1_{}_{}".format(str(startyear),str(overlap))
pixarr_save = resultpath + "/pix_list{}.npy".format(str(startyear) )
label_dir = resultpath + "/urban{}.tif".format(str(startyear) )

if not os.path.exists(resultpath):
    os.makedirs(resultpath)
else:
    del_file(resultpath)

WriteTiff(label_arr,urbanpath + str(startyear)  + ".tif",label_dir)
cutdata(image_size,label_dir,cutdata_dir,boundary_dir,overlap,resultpath,pixarr_save)

rotarr(resultpath)

tounet(resultpath)






