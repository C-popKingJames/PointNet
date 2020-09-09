import numpy as np
import warnings
import os
import torch.utils.data as data
warnings.filterwarnings('ignore')
import torch
from plyfile import PlyData, PlyElement




def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train',
                 data_augmentation=True):#数据增强
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())

        self.cat = {}#创建一个集合
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()# 以空格为分隔符，包含 \n
                #print(ls[0])
                #print(int(ls[1]))
                self.cat[ls[0]] = int(ls[1])#int(ls[1])把字符串转为数字,以键值对的方式存进cat字典里


        #print(self.cat)
        #print(self.cat.keys())
        self.classes = list(self.cat.keys())#列表里面存储字典里面的key
        print(self.classes)

    def __getitem__(self, index):
        fn = self.fns[index]
        print(fn)
        cls = self.cat[fn.split('/')[0]]
        with open(os.path.join(self.root, fn), 'rb') as f:
            plydata = PlyData.read(f)#读取点云.ply文件
        #pts首先获取图的坐标点云点，通过垂直矩阵成3组90714列的数据，然后进行装置处理编程（90714,3）列的数据
        #1.选取区域上的点的坐标
        pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        #2.统一点数#随机降采样，提高模型的鲁棒性
        choice = np.random.choice(len(pts), self.npoints, replace=True)#当前点云的个数为num,希望最后每个点云有2500个点
        #拿了2500个点，此时坐标变成（2500,3）
        point_set = pts[choice, :]

        #3.数据归一化
        #在3d空间中找一个能包含所有点的最小的立方体，这里代码找了0轴，（2500，0）所有数据取平均值，这个为中心点
        # 到这个立方体的下表面的中心点，以这个点作为归一化的参考点。所有点的坐标，减去参考点的坐标
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)#取完归一值后要转为矩阵做减法  # center
        #KNN邻近算法，计算距离
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)#生成一个[0,2π]之间的角度
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation,随机旋转
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        print(point_set.shape)

        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        print(cls)
        return point_set, cls


    def __len__(self):
        return len(self.fns)


if __name__ == '__main__':
    dataset = "modelnet"
    datapath = "E:/NewPython/PointNet/ModelNet40/"

    if dataset == 'modelnet':
        #gen_modelnet_id(datapath)
        d = ModelNetDataset(datapath,split='train')
        DataLoader = torch.utils.data.DataLoader(d, batch_size=12, shuffle=True)
        for point,label in DataLoader:
            print(point.shape)
            print(label.shape)

