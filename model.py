from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.autograd
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d,self).__init__()
        self.conv1=torch.nn.Conv1d(3,64,1)#输入向量的维度是3，输出的维度为64，卷积核的大小为1*1
        self.conv2=torch.nn.Conv1d(64,128,1)
        self.conv3=torch.nn.Conv1d(128,1024,1)
        self.fc1=nn.Linear(1024,512)
        self.fc2=nn.Linear(512, 256)
        self.fc3=nn.Linear(256,9)
        self.relu=nn.ReLU()

        self.bn1=nn.BatchNorm1d(64)
        self.bn2=nn.BatchNorm1d(128)
        self.bn3=nn.BatchNorm1d(1024)
        self.bn4=nn.BatchNorm1d(512)
        self.bn5=nn.BatchNorm1d(256)

    def forward(self,x):
        batchsize=x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x=torch.max(x,2,keepdim=True)[0]#只拿数据，不拿Tensor最大值下标
        x=x.view(-1,1024)#1024列转换成行

        x=F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x=self.fc3(x)

        #print(x)
        #转Tensor格式，重复32次，即(32,9)[1,0,0,0,1,0,0,0,1]
        iden=torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32)).view(1,9).repeat(batchsize,1)

        if x.is_cuda:
            iden=iden.cuda()

        x+=iden
        #print(x)
        x=x.view(-1,3,3)

        return x

class STNkd(nn.Module):
    def __init__(self,k=64):
        super(STNkd,self).__init__()
        self.conv1=torch.nn.Conv1d(k,64,1)
        self.conv2=torch.nn.Conv1d(64,128,1)
        self.conv3 = torch.nn.Conv1d(128, 1024,1)
        self.fc1=nn.Linear(1024,512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k=k

    def forward(self,x):
        batchsize=x.size()[0]
        x=F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x=torch.max(x,2,keepdim=True)[0]
        x=x.view(-1,1024)

        x=F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x=self.fc3(x)

        print(x)
        print(x.shape)

        iden=torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1,self.k*self.k).repeat(batchsize,1)

        print(iden)
        print(iden.shape)

        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x=x.view(-1,self.k,self.k)

        return x

#主体部分
class PointNetfeat(nn.Module):
    #global_feat控制3*3T-net,控制64*64T-net
    def __init__(self,global_feat=True,feature_transform=False):
        super(PointNetfeat,self).__init__()
        self.stn=STN3d()
        self.conv1=torch.nn.Conv1d(3,64,1)
        self.conv2=torch.nn.Conv1d(64,128,1)
        self.conv3=torch.nn.Conv1d(128,1024,1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat=global_feat
        self.feature_transform=feature_transform
        if self.feature_transform:
            self.fstn=STNkd(k=64)

    def forward(self,x):
        n_pts=x.size()[2]
        print(n_pts)
        trans=self.stn(x)
        #print(trans)
        #print(trans.shape)
        x=x.transpose(2,1)
        #print(x)
        #print(x.shape)
        x=torch.bmm(x,trans) # 注意两个tensor的维度必须为3,batch matrix multiply 即乘以T-Net的结果
        print(x)
        print(x.shape)
        x=x.transpose(2,1)
        print(x)
        print(x.shape)
        x=F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat=self.fstn(x)
            x=x.transpose(2,1)
            x=torch.bmm(x,trans_feat)
            x=x.transpose(2,1)

        else:
            trans_feat=None

        pointfeat=x
        x=F.relu(self.bn2(self.conv2(x)))
        x=self.bn3(self.conv3(x))
        x=torch.max(x,2,keepdim=True)[0]
        x=x.view(-1,1024)
        if self.global_feat:
            return x,trans,trans_feat
        else:
            x=x.view(-1,1024,1).repeat(1,1,n_pts)
            return torch.cat([x,pointfeat],1),trans,trans_feat

class PointNetCls(nn.Module):
    def __init__(self,k=2,feature_transform=False):
        super(PointNetCls,self).__init__()
        self.feature_transform=feature_transform
        self.feat=PointNetfeat(global_feat=True,feature_transform=False)
        self.fc1=nn.Linear(1024,512)
        self.fc2=nn.Linear(512,256)
        self.fc3=nn.Linear(256,k)

        #防止过拟合
        self.dropout=nn.Dropout(p=0.3)
        #归一防止梯度爆炸与梯度消失
        self.bn1=nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self,x):
        # 完成网络主体部分
        x,trans,trans_feat=self.feat(x)
        ## 经过三个全连接层（多层感知机）映射成k类
        x=F.relu(self.bn1(self.fc1(x)))
        x=F.relu(self.bn2(self.dropout(self.fc2(x))))
        x=self.fc3(x)

        # 返回的是该点云是第ki类的概率
        return F.log_softmax(x,dim=1),trans,trans_feat

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss





if __name__ == '__main__':
     sim_data = torch.rand(32,3,2500)#32组，3行，2500列的数据
     trans = STN3d()
     # out = trans(sim_data)
     # print('stn', out.size())
     #print('loss', feature_transform_regularizer(out))

     sim_data_64d = torch.rand(32, 64, 2500)
     trans = STNkd(k=64)
     # out = trans(sim_data_64d)
     # print('stn64d', out.size())

     pointfeat = PointNetfeat(global_feat=True)
     out, _, _ = pointfeat(sim_data)
     print('global feat', out.size())