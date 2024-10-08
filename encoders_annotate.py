import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np

from set2set import Set2Set
# 增加日志
import logging
# 引用主程序中的日志记录器
logger = logging.getLogger(__name__)

# GCN basic operation
# GCN基础类
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
            dropout=0.0, bias=True):
        """
        :param input_dim : 输入维度
        :param output_dim : 输出维度
        :param add_self : 决定是否在卷积操作中将自身的特征加入邻接特征
        :param normalize_embedding : 是否对输出的嵌入进行归一化
        """    
        super(GraphConv, self).__init__()   # 调用父类nn.Module的初始化方法
        self.add_self = add_self
        self.dropout = dropout
        # 如果dropout大于0.001,则创建一个nn.Dropout层,用于在前向传播时随机丢弃部分输入特征
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 定义了一个可学习的权重矩阵self.weight,用于将输入特征映射到输出特征空间
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):
        """
        :param x : 输入特征矩阵
        :param adj : 邻接矩阵
        """
        # 如果dropout大于0.001,则对输入特征x进行随机丢弃
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y,self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            # p=2表示L2范数,dim=2表示沿着第三个维度进行归一化,即节点特征维度
            y = F.normalize(y, p=2, dim=2)
            #print(y[0][0])
        return y

class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        """
        :param input_dim : 输入维度
        :param hidden_dim : 隐藏维度
        :param embedding_dim : 嵌入维度
        :param label_dim : 输出维度
        :param num_layers : 卷积层数
        :param pred_hidden_dims : 预测模型隐藏层的维度列表
        :param concat : 是否连接所有层的输出
        :param bn : 是否对输出的嵌入进行归一化
        """    
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat   # 表示如果不连接所有层的输出,则在卷积时加上节点自身特征
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs=1

        self.bias = True
        if args is not None:
            self.bias = args.bias

        # 调用build_conv_layers方法,构建卷积层,包括第一层卷积、隐藏层卷积和最后一层卷积
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                input_dim, hidden_dim, embedding_dim, num_layers, 
                add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()    # 使用ReLU作为激活函数
        self.label_dim = label_dim  # 存储标签的维度label_dim

        if concat:
            # 输入维度为隐藏层的维度乘以层数加上嵌入层维度
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            # 否则仅为嵌入层维度
            self.pred_input_dim = embedding_dim
        # 构建预测层网络
        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims, 
                label_dim, num_aggs=self.num_aggs)

        # 对所有子模块进行初始化。如果模块是GraphConv类型,使用Xavier初始化权重并将偏置初始化为0
        for m in self.modules():
            if isinstance(m, GraphConv):
                # m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    # m.bias.data = init.constant(m.bias.data, 0.0)
                    nn.init.constant_(m.bias, 0.0)

    # 构建卷积层
    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
            normalize=False, dropout=0.0):
        # 创建第一层卷积操作
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        
        # 创建中间的卷积层,用ModuleList存储。创建了num_layers-2层,每层输入和输出维度相同
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=self.bias) 
                 for i in range(num_layers-2)])

        # 创建最后一层卷积,将输出维度映射为embedding_dim
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last

    # 构建预测层
    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:  # 如果pred_hidden_dims为空,则直接创建线性层将输入映射到标签维度
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:   # 否则,构建隐藏层并添加激活函数。使用nn.Sequential将各层连接起来,形成预测模型
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    # 构建掩码
    def construct_mask(self, max_nodes, batch_num_nodes): 
        ''' 
        For each num_nodes in batch_num_nodes, the first num_nodes entries of the 
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        max_nodes 是批处理中允许的最大节点数,batch_num_nodes 是一个包含每个样本节点数的列表。
        函数的作用是为每个批次的节点数生成一个掩码矩阵,矩阵的维度为 [batch_size x max_nodes x 1],用于在处理不同数量的节点时,掩盖掉不必要的节点。
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)

        # 将 packed_masks 中对应的掩码填充到 out_tensor 的前 num_nodes 个位置，剩下的元素保持为 0。
        # 如果 batch_num_nodes[i] = 3，那么 out_tensor[i, :3] 会被填充为 1，其余部分保留为 0。
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask

        # unsqueeze()在第二个维度之后 max_nodes 之后增加一个维度,掩码的最终维度为 [batch_size, max_nodes, 1]
        return out_tensor.unsqueeze(2).cuda()
    
    # 批量归一化
    def apply_bn(self, x):
        ''' 
        Batch normalization of 3D tensor x
        对输入特征x进行批量归一化,BatchNorm1d用于归一化特征维度
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]

        x : [batch_size x num_nodes x input_dim],图中每个节点的输入特征
        adj : [batch_size x num_nodes x num_nodes],图中节点的连接关系
        conv_first : 第一层卷积层,输出维度为[batch_size, num_nodes, hidden_dim]
        conv_block : 中间的卷积层,输出维度为[batch_size, num_nodes, hidden_dim]
        conv_last : 最后一层卷积层,输出维度为[batch_size, num_nodes, embedding_dim]
        embedding_mask : 可选的掩码,用于屏蔽无效节点特征
        '''

        x = conv_first(x, adj)
        x = self.act(x) # 激活函数ReLu
        if self.bn:
            x = self.apply_bn(x)
        # 初始化一个列表 x_all,用于存储每一层卷积后的特征
        x_all = [x]
        #out_all = []
        #out, _ = torch.max(x, dim=1)
        #out_all.append(out)
        for i in range(len(conv_block)):
            x = conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x,adj)
        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding] = [batch_size x num_nodes x (hidden_dim * (num_layers - 1) + embedding_dim)]
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
    
        '''
        x : [batch_size x num_nodes x input_dim],图中每个节点的输入特征
        adj : [batch_size x num_nodes x num_nodes],图中节点的连接关系
        batch_num_nodes : 每个图中有效节点的数量,用于生成掩码,以处理图中节点数量不一致的情况
        num_aggs == 2 : 每个图的特征可以同时通过最大池化和求和池化两种方式来聚合
        '''
        # mask
        max_num_nodes = adj.size()[1]   # 每个图的最大节点数量
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []    # 用于存储所有层的池化结果
        out, _ = torch.max(x, dim=1)    # 对卷积后的特征 x 进行最大池化 (torch.max) 操作,沿着节点维度（dim=1）取最大值,得到每个图的全局特征
        out_all.append(out)
        for i in range(self.num_layers-2):
            x = self.conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out,_ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:  # 如果 num_aggs == 2，则在最大池化后还对特征进行求和池化（torch.sum），将结果同样添加到 out_all 中
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x,adj)
        #x = self.act(x)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            # 所有池化结果沿特征维度（dim=1）进行拼接,形成最终的特征表示 output
            output = torch.cat(out_all, dim=1)
        else:
            # 仅使用最后一层卷积的池化结果作为最终输出
            output = out
        
        ypred = self.pred_model(output)
        #print(output.size())
        return ypred

    # 计算预测值与真实标签之间的损失,用于模型的训练和优化
    def loss(self, pred, label, type='softmax'):
        '''
        pred : 预测值，形状为 [batch_size, label_dim]，表示每个样本的预测输出
        label : 真实标签，形状为 [batch_size]，表示每个样本的真实类别
        type : 损失类型，默认为 'softmax'，可选为 'softmax' 或 'margin'
        '''
        # softmax + CE
        if type == 'softmax':
            # 交叉熵损失函数 (F.cross_entropy) 计算损失
            return F.cross_entropy(pred, label, reduction='mean')
        elif type == 'margin':
            # 多标签边缘损失 (margin)：用于多标签分类任务，鼓励正确类别的预测分数比其他类别高。
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1,1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)
            
        #return F.binary_cross_entropy(F.sigmoid(pred[:,0]), label.float())


class GcnSet2SetEncoder(GcnEncoderGraph):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnSet2SetEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                num_layers, pred_hidden_dims, concat, bn, dropout, args=args)
        self.s2s = Set2Set(self.pred_input_dim, self.pred_input_dim * 2)

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        embedding_tensor = self.gcn_forward(x, adj,
                self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        out = self.s2s(embedding_tensor)
        #out, _ = torch.max(embedding_tensor, dim=1)
        ypred = self.pred_model(out)
        return ypred


class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
            pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True,
            assign_input_dim=-1, args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat, args=args)
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True

        # GC
        self.conv_first_after_pool = nn.ModuleList()
        self.conv_block_after_pool = nn.ModuleList()
        self.conv_last_after_pool = nn.ModuleList()
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
                    self.pred_input_dim, hidden_dim, embedding_dim, num_layers, 
                    add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(conv_first2)
            self.conv_block_after_pool.append(conv_block2)
            self.conv_last_after_pool.append(conv_last2)

        # assignment
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = nn.ModuleList()
        self.assign_conv_block_modules = nn.ModuleList()
        self.assign_conv_last_modules = nn.ModuleList()
        self.assign_pred_modules = nn.ModuleList()
        assign_dim = int(max_num_nodes * assign_ratio)
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
                    assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                    normalize=True)
            assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
            assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)


            # next pooling layer
            assign_input_dim = self.pred_input_dim
            assign_dim = int(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(assign_conv_first)
            self.assign_conv_block_modules.append(assign_conv_block)
            self.assign_conv_last_modules.append(assign_conv_last)
            self.assign_pred_modules.append(assign_pred)

        self.pred_model = self.build_pred_layers(self.pred_input_dim * (num_pooling+1), pred_hidden_dims, 
                label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def forward(self, x, adj, batch_num_nodes, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x

        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []

        #self.assign_tensor = self.gcn_forward(x_a, adj, 
        #        self.assign_conv_first_modules[0], self.assign_conv_block_modules[0], self.assign_conv_last_modules[0],
        #        embedding_mask)
        ## [batch_size x num_nodes x next_lvl_num_nodes]
        #self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred(self.assign_tensor))
        #if embedding_mask is not None:
        #    self.assign_tensor = self.assign_tensor * embedding_mask
        # [batch_size x num_nodes x embedding_dim]
        embedding_tensor = self.gcn_forward(x, adj,
                self.conv_first, self.conv_block, self.conv_last, embedding_mask)

        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None

            self.assign_tensor = self.gcn_forward(x_a, adj, 
                    self.assign_conv_first_modules[i], self.assign_conv_block_modules[i], self.assign_conv_last_modules[i],
                    embedding_mask)
            # [batch_size x num_nodes x next_lvl_num_nodes]
            self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred_modules[i](self.assign_tensor))
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask

            # update pooled features and adj matrix
            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x
        
            embedding_tensor = self.gcn_forward(x, adj, 
                    self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                    self.conv_last_after_pool[i])


            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                #out = torch.mean(embedding_tensor, dim=1)
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)


        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        return ypred

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        ''' 
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2) 
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop-1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).cuda())
            #print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
            #print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
            #self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
            self.link_loss = -adj * torch.log(pred_adj+eps) - (1-adj) * torch.log(1-pred_adj+eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print('Warning: calculating link pred loss without masking')
                logging.info('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[(1-adj_mask).bool()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            #print('linkloss: ', self.link_loss)
            # logging.info
            return loss + self.link_loss
        return loss

