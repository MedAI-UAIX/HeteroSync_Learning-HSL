"""
参考：
https://blog.csdn.net/sgyuanshi/article/details/120939816
https://github.com/tomtang110/Multitask/blob/master/Models/mmoe1.py
https://github.com/easezyc/Multitask-Recommendation-Library
"""

import torch
import torch.nn as nn
import torchvision.models as models


# 定义卷积模型，特征提取
class EncodingNet(nn.Module):
    def __init__(self, output_dim=512):
        super(EncodingNet, self).__init__()
        self.net = models.resnet18(pretrained=True)         
        self.myfc = nn.Linear(self.net.fc.in_features, output_dim)
        self.net.fc = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.net(x)
        x = self.myfc(x)
        x = self.relu(x)
        return x

class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        #参数初始化
        torch.nn.init.xavier_uniform_(self.fc1.weight.data)
        torch.nn.init.xavier_uniform_(self.fc2.weight.data)
        # self.log_soft = nn.LogSoftmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        # out = self.log_soft(out)
        return out

class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.softmax = nn.Softmax(dim=1)
        #参数初始化
        torch.nn.init.xavier_uniform_(self.fc1.weight.data)
        torch.nn.init.xavier_uniform_(self.fc2.weight.data)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        # out = self.softmax(out/self.T)
        out = torch.sigmoid(out)
        return out



# MMoE模型
class MMoE(nn.Module):
    def __init__(self, num_experts=12, num_feature=512, experts_out=32, experts_hidden=62, towers_hidden=16, tasks=3,T=3):
        super(MMoE, self).__init__()
        """
        num_experts     专家层里面，专家的个数
        num_feature   每一个任务输出的特征维度, resnet18为512
        experts_hidden  每一个专家内部的神经元数量
        experts_out     每一个专家输出的特征维度
        towers_hidden   分类塔内部的神经元数量
        tasks           任务数量      
        """
        self.num_experts = num_experts
        self.num_feature = num_feature
        self.experts_hidden = experts_hidden
        self.experts_out = experts_out       
        self.towers_hidden = towers_hidden
        self.tasks = tasks
        self.embed_output_dim = self.num_feature*self.tasks  #每个任务输出的维度特征*任务数
        self.T = T

        #特征提取层
        self.encoding_1 = EncodingNet(output_dim=num_feature)
        self.encoding_2 = EncodingNet(output_dim=num_feature)

        self.down1 = nn.Linear(num_feature,experts_out)
        self.down2 = nn.Linear(num_feature,experts_out)

        # 专家层
        self.experts = nn.ModuleList([Expert(self.embed_output_dim, self.experts_out, self.experts_hidden) for i in range(self.num_experts)])

        #分类塔
        self.towers = nn.ModuleList()
        tower1 = Tower(self.experts_out, 1, self.towers_hidden)
        tower2 = Tower(self.experts_out, 1, self.towers_hidden)
        self.towers.append(tower1)
        self.towers.append(tower2)

        #门控层
        self.gates = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.embed_output_dim, self.num_experts), torch.nn.Softmax(dim=1)) for i in range(self.tasks)])

    def forward(self, x_task1, x_task2):
        x_task1 = self.encoding_1(x_task1)
        x_task2 = self.encoding_2(x_task2)
        down1 = self.down1(x_task1)
        down2 = self.down2(x_task2)
        down = [down1,down2]

        emb = torch.cat([x_task1, x_task2], 1).view(-1, self.embed_output_dim)  #将多个任务的特征拼接
        emb = torch.softmax(emb/self.T, dim=-1)

        gate_value = [self.gates[i](emb).unsqueeze(1) for i in range(self.tasks)]   #计算每个任务对应的门控
        #print(gate_value[0].shape)  #torch.Size([8, 1, 6])  （b，1，n）表示每个门控均可调节n个专家

        fea = torch.cat([self.experts[i](emb).unsqueeze(1) for i in range(self.num_experts)], dim = 1)  #将所有专家输出叠加，一层为一个专家，n个专家共n层
        #print(fea.shape)  #torch.Size([8, 6, 16])    （b, n, 16） 16为每个专家输出的特征数，可选参数
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.tasks)]  # m个任务就有m个门控，每个门控均需要与每个专家交互，用m次循环表示m次门控。共产生n*m,此处用矩阵相乘方便计算

        results = [self.towers[i](task_fea[i] + down[i]).squeeze(1) for i in range(self.tasks)]       
        return results

if __name__ == '__main__':
    us1 = torch.rand((8, 3, 128, 128))
    us2 = torch.rand((8, 3, 128, 128))
    model = MMoE(tasks=2)
    output = model(us1, us2)
    print(len(output))
    print(output[0].shape)
    print(output[1].shape)
