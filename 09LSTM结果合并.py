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
    if nb == 0:
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


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
    
uppath = os.path.dirname(os.getcwd())
arr = np.load(uppath + "/BiLSTMdata/probability_result.npy")[:,1]
inputdir = uppath + "/inputdata/boundary/boundary.tif"
arr_boundary = LoadData(inputdir)
index = np.where(arr_boundary == 1)
arr_boundary[index] = arr
resultpath = "../BiLSTMdata/BiLSTM2015.tif"
WriteTiff(arr_boundary,inputdir, resultpath)

arr_boundary1 = normalization(arr_boundary)
resultpath = "../BiLSTMdata/BiLSTM2015_nor.tif"
WriteTiff(arr_boundary1,inputdir, resultpath)