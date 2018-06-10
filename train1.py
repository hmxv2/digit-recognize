
# coding: utf-8

# In[1]:

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


# In[2]:

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
        x=self.dropout_dot4(x)
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
        


# In[3]:

#import data for training
path=os.getcwd()+'/train.csv'#/sample_submission.csv'
all_data=pd.read_csv(path)


# In[4]:

'''
all_data.columns
all_data.index
all_data.iloc[1]
'''


# In[5]:

'''
img0=all_data.iloc[20000,1:]
img0=img0.reshape((28,28))
#help(plt.imshow)#
plt.imshow(np.array(img0))
plt.show()
'''


# In[ ]:

all_data=np.array(all_data)
#shuffle
print(all_data[0:2,:])
np.random.shuffle(all_data)
print(all_data[0:2,:])
row_num=len(all_data)
train_control=int(row_num*0.96)
valid_control=int(row_num*1)
train_data=all_data[0:train_control]
valid_data=all_data[train_control:valid_control]
#print(row_num)
#get train data in 4d
train_data_4d=np.ones((train_control,1,28,28))
for idx in range(train_control):
    train_data_4d[idx,0]=train_data[idx,1:].reshape((28,28))
    if idx==1:
        plt.imshow(np.array(train_data[idx,1:].reshape((28,28))))
        plt.show()
    
#get valid data in 4d
valid_data_4d=np.ones((valid_control-train_control,1,28,28))
for idx in range(valid_control-train_control):
    valid_data_4d[idx,0]=valid_data[idx,1:].reshape((28,28))
#print(train_data_4d)
#print(valid_data_4d)


# In[ ]:

#model
model=Net()#188,100,50,25,3
LEARNING_RATE=0.0015
num_epochs=50
batch_div=10
#input


criterion=nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,momentum=MOMENTUM)
optimizer = optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr=LEARNING_RATE)
#for plot
all_loss=np.zeros(num_epochs)
all_valid_loss=np.zeros(num_epochs)
all_accuracy_rate=np.zeros(num_epochs)
if os.path.exists('./models'):
    pass
else:
    os.mkdir('./models')
for epoch in range(num_epochs):
    running_loss=0
    batch_len=int(len(train_data_4d)*1.0/batch_div)
    for ii in [batch_len*x for x in range(batch_div)]:
        #print(train_data_4d)
        #print('-------')
        #print(valid_data_4d)
        inputs=Variable(torch.from_numpy(train_data_4d[ii:ii+batch_len])).float()
        targets=Variable(torch.from_numpy(train_data[ii:ii+batch_len,0])).long()
        out=model(inputs)#forward
        loss=criterion(out,targets)#get loss
    
        running_loss+=loss.data[0]
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    all_loss[epoch]=running_loss
    #valid
    valid_loss=0
    
    model.eval()
    #print(valid_data_4d)
    valid_inputs=Variable(torch.from_numpy(valid_data_4d)).float()
    valid_targets=Variable(torch.from_numpy(valid_data[:,0])).long()
    valid_out=model(valid_inputs)#forward
    max_value,max_idx=torch.max(valid_out,1)
    #print(max_idx)
    accuracy_rate=sum(abs(max_idx.data.numpy()==valid_data[:,0]))/len(valid_data)
    valid_loss=criterion(valid_out,valid_targets).data[0]#get loss
    all_valid_loss[epoch]=valid_loss
    all_accuracy_rate[epoch]=accuracy_rate
    #
    model.train()
    #print
    print('epoch: {}/{}, Loss: {}, valid_loss: {}, accuracy_rate: {}'.format(epoch,num_epochs, running_loss, valid_loss, accuracy_rate))
    #model save
    torch.save(model, './models/epoch{}Loss{}valid_loss{}accuracy_rate{}.pkl'.format(epoch, running_loss,valid_loss,accuracy_rate))
    
#plot
plt.plot(range(num_epochs),all_loss,'k',range(num_epochs),10*all_valid_loss,'r',range(num_epochs), 10*all_accuracy_rate, 'b')
plt.grid(True)
plt.show()
#model save
torch.save(model, 'model.pkl')


# In[ ]:



