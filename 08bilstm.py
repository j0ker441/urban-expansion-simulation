# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(years,20,2, bidirectional=True) #
        self.fc = nn.Linear(20 * 2, 2)

    def forward(self, x):
        out, h = self.lstm(x)
        out = self.fc(out[:,-1,:]) #(batch,time_step,input))
        return out

years = 36
batch = 50000
data = np.load("../LSTMdata/train/traindata.npy")
label = np.load("../LSTMdata/label/labeldata.npy")
scaler = MinMaxScaler()
data = scaler.fit_transform(data.transpose())
train_x,test_x,train_y,test_y=train_test_split(data,label,test_size=0.1,random_state=42)
train_dataset = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)

test_dataset = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y))
test_loader = DataLoader(dataset=test_dataset, batch_size=batch, shuffle=False)

model = Model()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    with tqdm(total=len(train_loader)) as pbar:
        pbar.set_description(str(epoch) + ' train:')
        for X, y in train_loader:
            X = Variable(X.reshape(-1,1,years).to(device))
            y = Variable(y.long().to(device))
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
    
            predict = out.argmax(dim=1, keepdim=True)
            correct = predict.eq(y.view_as(predict)).sum().item()
            accuracy = correct / len(X)
            print("epoch:",epoch,"loss:",loss.item(), "accuracy:",accuracy)
            pbar.update(1)
            
            
    correct = 0
    total =0
    with tqdm(total=len(test_loader)) as pbar:
        pbar.set_description(str(epoch) + ' test:')
        for data_test in test_loader:
            temp_x,temp_y = data_test
            X = Variable(temp_x.reshape(-1, 1, years).to(device))
            pred = model(X)
            pred = torch.max(pred, 1)[1]
            num_correct = (pred.cpu().detach().numpy() == temp_y.detach().cpu().numpy()).sum()
            correct =correct+num_correct
            total=total+len(temp_y)
            print("Test ACC:",correct/total)
            pbar.update(1)
#保存所有的预测结果
print("开始预测")
model.eval()
dataset = TensorDataset(torch.Tensor(data), torch.Tensor(label))
data_load = DataLoader(dataset=dataset, batch_size=batch, shuffle=False)
total_result = []
total_label = []
probability_result = []
for i, (X, y) in enumerate(data_load):
    X = Variable(X.reshape(-1, 1, years).to(device))
    y = Variable(y.long().to(device))
    pred_1 = model(X)
    probability = torch.softmax(pred_1,dim=1)
    
    probability = probability.cpu().detach().numpy()
    probability_result.extend(probability)
    pred = torch.max(pred_1, 1)[1]
    total_result.extend(pred.cpu().detach().numpy().reshape(-1))
    total_label.extend(y.cpu().detach().numpy())
print("total_number:",len(total_result),"Totall ACC:",(sum(np.array(total_result)==np.array(total_label)))/len(total_result))
np.save("../LSTMdata/predict.npy",total_result)
np.save("../LSTMdata/probability_result.npy",probability_result)

