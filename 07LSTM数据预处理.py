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
    if nb == 0:
        data = data[0,:,:]
    
    return data


uppath = os.path.dirname(os.getcwd())

arr_boundary = LoadData(uppath + "/inputdata/boundary/boundary.tif")

startyear = 2015
arr_label= LoadData(uppath +  "/origindata/urban/urban2/urban{}.tif".format(str(startyear)))
arr_label[np.where(arr_boundary == 0)] = np.nan
labeldata = arr_label.reshape(-1)
labeldata = labeldata[~np.isnan(labeldata)]
if not os.path.exists("../BiLSTMdata/label"):
	os.makedirs("../BiLSTMdata/label")
np.save("../BiLSTMdata/label/labeldata.npy",labeldata)

files = []

years = [2000,2005,2010]
for i in years:
    # files.append(uppath +  "/origindata/urban/urban2/urban{}.tif".format(str(i)))
    files.append(uppath +  "/inputdata/lulc/lulc{}.tif".format(str(i)))
    

for file in files:
    arr_train= LoadData(file)
    arr_train[np.where(arr_boundary == 0)] = np.nan
    traindata = arr_train.reshape(-1)
    traindata = traindata[~np.isnan(traindata)]
    
    if file == files[0]:
        resultdata = np.copy(traindata)
    else:
        resultdata = np.vstack((resultdata,traindata))
    print(file)
    if not os.path.exists("../BiLSTMdata/train"):
    	os.makedirs("../BiLSTMdata/train")
np.save("../BiLSTMdata/train/traindata.npy",resultdata)
    


