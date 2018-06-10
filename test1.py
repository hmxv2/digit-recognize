
# coding: utf-8

# In[6]:


import csv
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import math
import pandas as pd
import os
print('imported!')
    


# In[7]:

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,32,3,padding=1)
        self.conv2=nn.Conv2d(32,64,3,padding=1)
        self.conv3=nn.Conv2d(64,128,3,padding=1)
        self.fc1=nn.Linear(1152,500)
        self.fc2=nn.Linear(500,100)
        self.fc3=nn.Linear(100,10)
        self.relu=nn.ReLU()#Sigmoid()#ReLU()#Sigmoid()
        self.softmax=nn.Softmax()
        self.dropout_dot5=nn.Dropout(0.5)
        self.dropout_dot4=nn.Dropout(0.4)
        self.dropout_dot3=nn.Dropout(0.3)
    def forward(self,x):
        x=self.conv1(x)
        x=F.max_pool2d(self.relu(x),(2,2))
        x=self.conv2(x)
        x=F.max_pool2d(self.relu(x),2)
        x=self.conv3(x)
        x=F.max_pool2d(self.relu(x),2)
        x=x.view(-1,self.num_flat_features(x))
        #print(x)
        x=self.dropout(x)
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x
    
    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        return num_features


# In[8]:

#import data for training
path=os.getcwd()+'/test.csv'#/sample_submission.csv'
test_data=pd.read_csv(path)


# In[9]:

test_data=np.array(test_data)
row_num=len(test_data)
#get test data in 4d
test_data_4d=np.ones((row_num,1,28,28))
for idx in range(row_num):
    test_data_4d[idx,0]=test_data[idx].reshape((28,28))
#print(test_data_4d)


# In[10]:

#load model
model=torch.load('./bettermodels/model.pkl')
#test
model.eval()
batch_div=10
batch_len=int(len(test_data_4d)/batch_div)
predict=np.zeros(len(test_data_4d))
for ii in [x*batch_len for x in range(batch_div)]:
    predict_out=model(Variable(torch.from_numpy(test_data_4d[ii:ii+batch_len])).float())
    max_value,max_idx=torch.max(predict_out,1)
    predict[ii:ii+batch_len]=max_idx.data.numpy()

#predict_out=model(Variable(torch.from_numpy(test_data_4d)).float())
#max_value,max_idx=torch.max(predict_out,1)
#predict=max_idx.data.numpy()
predict


# In[11]:

with open("result.csv","w") as csvfile: 
    writer = csv.writer(csvfile)
    #先写入columns_name
    writer.writerow(["ImageId","Label"])
    #写入多行用writerows
    for idx in range(len(predict)):
        writer.writerow([str((idx+1)),str(int(predict[idx]))])

