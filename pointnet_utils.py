import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

#square_distance函数主要用来在ball query过程中确定每一个点距离采样点的距离
#两组点，N第一组点个数，M第二组点的个数，C为通道数（xyz时C=3），返回的是两组点之间的欧几里得距离
#训练中通常以Mini-Batch的形式输入，Batch数量的维度是B.src[32，2500，3]
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape#N第一组点的个数
    _, M, _ = dst.shape#M第二组点的个数
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))#N×M的矩阵
    #数组的广播机制 dist=sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

#index_points函数是按照输入的点云数据和索引返回由索引的点云数据。
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx也是tensor,维度也为[B,S]，其表示的是每一行对应1个样本，该行中的数据就是在points当中数据的索引。
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]#B为batch
    view_shape = list(idx.shape)#2
    view_shape[1:] = [1] * (len(view_shape) - 1)#view_shape=[B,1],[1]
    # repeat_shape=[B,S]
    repeat_shape = list(idx.shape)
    #repeat_shape=[1,S],[1,512]
    repeat_shape[0] = 1
    # batch_indices的维度[B,S],batch_indices是tensor,维度为[B,S]，表示的是样本索引，即取哪个样本，
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    #points[batch_indices, idx, :]说的就是从points当中取出每个batch对应索引的数据点。
    new_points = points[batch_indices, idx, :]
    return new_points


#FPS(Farthest Point Sampling) 最远点采样法，该方法比随机采样的优势在于它可以尽可能的覆盖空间中的所有点。
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]代表采样点的索引，其维度为[B, N]
    """
    device = xyz.device
    B, N, C = xyz.shape
    #先随机初始化一个centroids矩阵，后面用于存储npoint个采样点的索引位置，大小为B×npoint，其中B为BatchSize的个数，即B个样本;
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    #利用distance矩阵记录某个样本中所有点到某一个点的距离，初始化为B×N矩阵，初值给个比较大的值，后面会迭代更新;
    distance = torch.ones(B, N).to(device) * 1e10
    #利用farthest表示当前最远的点，也是随机初始化，范围为0~N，初始化B个，对应到每个样本都随机有一个初始最远点；
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    #batch_indices初始化为0~(B-1)的数组；
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        # 更新第i个最远点
        centroids[:, i] = farthest
        # 取出这个最远点的xyz坐标,1是把那个点坐标展成1行
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        #计算点集中的所有点到这个最远点的欧式距离
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # 更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
        mask = dist < distance
        distance[mask] = dist[mask]
        # 从更新后的distances矩阵中找出距离最远的点，作为最远点用于下一轮迭代
        # 取出每一行的最大值构成列向量，等价于torch.max(x,2)
        farthest = torch.max(distance, -1)[1]#返回距离最大的值，第二个是距离最大的下标
    return centroids

#query_ball_point函数对应于Grouping layer
#成N'个局部区域
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius//radius为球形领域的半径
        nsample: max sample number in local region//nsample为每个领域中要采样的点
        xyz: all points, [B, N, 3]//xyz为所有的点云；输出为每个样本的每个球形领域的nsample个采样点集的索引[B,S,nsample]
        new_xyz: query points, [B, S, 3]//new_xyz为S个球形领域的中心
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape#[B, S, 3]
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])#[B, S, nsample]
    # sqrdists: [B, S, N] 记录中心点与所有点之间的欧几里德距离(square_distance函数)
    sqrdists = square_distance(new_xyz, xyz)
     # 找到所有距离大于radius^2的，其group_idx直接置为N；其余的保留原来的值
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]#dim=-1说的是最后一维，在源码中指的就是dim=2，反向排序
    # 做升序排列，前面大于radius^2的都是N，会是最大值，所以会直接在剩下的点中取出前nsample个点
    # 考虑到有可能前nsample个点中也有被赋值为N的点（即球形区域内不足nsample个点），这种点需要舍弃，直接用第一个点来代替即可
    # group_first: [B, S, nsample]， 实际就是把group_idx中的第一个点的值复制到[B, S, nsample]的维度，便利于后面的替换
    # 找到group_idx中值等于N的点，会输出0,1构成的三维Tensor，维度为[B,S,nsample][32,512,32]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    # 将这些点的值替换为第一个点的值
    # 考虑到有可能前nsample个点中也有被赋值为N的点（即球形区域内不足nsample个点），这种点需要舍弃，直接用第一个点来代替即可
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:FPS采样点的数量
        radius:球形区域所定义的半径
        nsample:球形区域所能包围的最大的点数量
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]，D不包含坐标数据x,y,z
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3][B, npoint, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    # 从原点云中挑出最远点采样的采样点为new_xyz
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    torch.cuda.empty_cache()## 只有执行完上面这句，显存才会在Nvidia-smi中释放
    # 通过index_points将FPS采样点从原始点中挑出来
    # new_xyz代表中心点，此时维度为[B, S, 3]
    #说的就是从points当中取出每个batch对应索引的数据点。[32, 512, 3]
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    # idx:[B, npoint, nsample] 代表npoint个球形区域中每个区域的nsample个采样点的索引
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    #通过index_points将所有group内的nsample个采样点从原始点中挑出来
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    # grouped_xyz减去中心点：每个区域的点减去区域的中心值
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()
    # 如果每个点上面有新的特征的维度，则用新的特征与旧的特征拼接，否则直接返回旧的特征
    if points is not None:
        # 通过index_points将所有group内的nsample个采样点从原始点中挑出来，得到group内点的除坐标维度外的其他维度的数据
        grouped_points = index_points(points, idx)
        #dim=-1代表按照最后的维度进行拼接，即相当于dim=3
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

#sample_and_group_all直接将所有点作为一个group，即增加一个长度为1的维度而已，当然也存在拼接新的特征的过程。
def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
        直接将所有点作为一个group，即增加一个长度为1的维度而已
    """
    device = xyz.device
    B, N, C = xyz.shape
    #new_xyz代表中心点，用原点表示
    new_xyz = torch.zeros(B, 1, C).to(device)
    # grouped_xyz减去中心点：每个区域的点减去区域的中心值，由于中心点为原点，所以结果仍然是grouped_xyz
    grouped_xyz = xyz.view(B, 1, N, C)
    # 如果每个点上面有新的特征的维度，则用新的特征与旧的特征拼接，否则直接返回旧的特征
    if points is not None:
        #view(B, 1, N, -1)，-1代表自动计算，即结果等于view(B, 1, N, D)
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    #self,points: (batch_size, ndataset, channel)
    '''
                PointNet Set Abstraction (SA) Module
                    Input:
                        xyz: (batch_size, ndataset, 3) <只包含坐标的点>
                        points: (batch_size, ndataset, channel)
                        <Note: 不包含坐标，只包含了每个点经过之前层后提取的除坐标外的其他特征，所以第一层没有>
                        npoint: int32 -- #points sampled in farthest point sampling
                        <Sample layer找512个点作为中心点，这个手工选择的，靠经验或者说靠实验>
                        radius: float32 -- search radius in local region
                        <Grouping layer中ball quary的球半径是0.2，注意这是坐标归一化后的尺度>
                        nsample: int32 -- how many points in each local region
                         <围绕每个中心点在指定半径球内进行采样，上限是32个；半径占主导>
                        mlp: list of int32 -- output size for MLP on each point
                        <PointNet layer有多少层，特征维度变化分别是多少，如（64,64,128）>
                        group_all: bool -- group all points into one PC if set true, OVERRIDE
                            npoint, radius and nsample settings
                    Return:
                        new_xyz: (batch_size, npoint, 3) TF tensor
                        new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
                        idx: (batch_size, npoint, nsample) int32 -- indices for local regions
            '''

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        # 在构造函数__init__中用到list、tuple、dict等对象时，
        # 一定要思考是否应该用ModuleList或ParameterList代替。
        # 如果你想设计一个神经网络的层数作为输入传递
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # permute(dims):将tensor的维度换位，[B, N, 3]
        xyz = xyz.permute(0, 2, 1)
        # 判断点是否有其他特征维度
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            # new_xyz:[B, 1, 3],FPS sampled points position data
            # new_points:[B, 1, N, 3+D], sampled points data
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:

            #new_xyz:[B, npoint, 3], new_points:[B, npoint, nsample, 3+D]
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_points: sampled points data, [B, 3+D, nsample, npoint] OR [B, 3+D, N, 1]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        # 利用1x1的2d的卷积相当于把每个group当成一个通道，共npoint个通道，
        # 对[3+D, nsample]的维度上做逐像素的卷积，结果相当于对单个C+D维度做1d的卷积
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        # 对每个group做一个max pooling得到局部的全局特征,得到的new_points:[B,3+D,npoint]
        new_points = torch.max(new_points, 2)[0]
        #new_xyz:[B, 3, npoint]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

if __name__ == '__main__':
    trans=PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6, mlp=[64, 64, 128],group_all=False)
    #print(sa1)
    sim_data = torch.rand(32, 3, 2500)  # 32组，3行，2500列的数据
    data = torch.rand(32, 3, 2500)
    x,y=trans(sim_data,data)
    print(x,y)

    # trans1=sample_and_group(npoint=512, radius=0.2, nsample=32, xyz=sim_data, points=data, returnfps=False)
    # x1, y1 = trans1(sim_data, data)
    # print(x1, y1)
# if __name__ == '__main__':
#     points=torch.rand(32,2500,3)
#     idx=torch.rand(32,133,100,200)
#     new=index_points(points,idx)
#     print(new)

