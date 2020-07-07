import csv
import pandas as pd
import torch
import os
#将文件夹内的图片的名字存放在一个txt文件中。
import os
from pycm import *
from torchvision import transforms,models
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import math
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
PATH='/home/cad429/code/panda/model'
bc=128
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(size=(227, 227)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

train_dataset = ImageFolder(root='/home/cad429/code/data/splitk', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bc, shuffle=True)

# test_dataset = ImageFolder(root='../test', transform=transform)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bc, shuffle=True)

model=models.resnet18(pretrained=True)
for para in model.parameters():
    para.require_grad=False
model.fc=nn.Linear(512,2)
optimizer=optim.SGD(model.parameters(),lr=0.0001)
loss=nn.CrossEntropyLoss()


def train(model,train_loader):
    model.cuda()
    model.train()
    total_loss=0
    for epoch in range(10):
        print("epoch:", epoch)
        s=0
        sum=0
        for i,data in enumerate(train_loader):

            inputs,labels=data
            inputs=Variable(inputs).cuda()
            labels=Variable(labels).cuda()

            optimizer.zero_grad()
            outputs=model(inputs)

            loss1=loss(outputs,labels)
            loss1.backward()
            optimizer.step()
            total_loss += loss1.item()
            #print(loss1)
            _, pred = torch.max(outputs, 1)
            #print(pred)
            pred=pred.cpu().numpy()
            labels=labels.cpu().numpy()

            for j in range(len(pred)):
                #print(len(pred))
                if pred[j] == labels[j]:
                    s+=1
            sum=sum+bc
            if i%bc==0:
                print('i=',i)
                loss2=loss1.cpu().detach().numpy()
                print("loss",loss2)
                # print("s:",s)
                # print("train_loader len:",len(train_loader))
                # sa=len(train_loader)
                acc=s/sum
                print("acc: ",acc)
                s=0
                sum=0


    # model.eval()
    # s1=0
    # for i, data in enumerate(test_loader):
    #     input2, labels2 = data
    #     input2 = Variable(input2).cuda()
    #     labels2 = Variable(labels2).cuda()
    #     outputs2 = model(input2)
    #     _, pred2 = torch.max(outputs2, 1)
    #     optimizer.zero_grad()
    #     loss2 = loss(outputs2, labels2)
    #     pred2 = pred2.cpu().numpy()
    #     labels2 = labels2.cpu().numpy()
    #     print("loss:", loss2)
    #     for aa in range(len(pred2)):
    #         if pred2[aa] == labels2[aa]:
    #             s1 += 1
    # print("s1:",s1)
    # sb=len(test_loader)
    # acc = s1 / (sb*2)
    # print("test acc:", acc)



train(model,train_loader)
torch.save(model.state_dict(),PATH)