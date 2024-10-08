import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np

from set2set import Set2Set
# ������־
import logging
# �����������е���־��¼��
logger = logging.getLogger(__name__)

# GCN basic operation
# GCN������
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
            dropout=0.0, bias=True):
        """
        :param input_dim : ����ά��
        :param output_dim : ���ά��
        :param add_self : �����Ƿ��ھ�������н���������������ڽ�����
        :param normalize_embedding : �Ƿ�������Ƕ����й�һ��
        """    
        super(GraphConv, self).__init__()   # ���ø���nn.Module�ĳ�ʼ������
        self.add_self = add_self
        self.dropout = dropout
        # ���dropout����0.001,�򴴽�һ��nn.Dropout��,������ǰ�򴫲�ʱ�������������������
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        # ������һ����ѧϰ��Ȩ�ؾ���self.weight,���ڽ���������ӳ�䵽��������ռ�
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):
        """
        :param x : ������������
        :param adj : �ڽӾ���
        """
        # ���dropout����0.001,�����������x�����������
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y,self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            # p=2��ʾL2����,dim=2��ʾ���ŵ�����ά�Ƚ��й�һ��,���ڵ�����ά��
            y = F.normalize(y, p=2, dim=2)
            #print(y[0][0])
        return y

class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        """
        :param input_dim : ����ά��
        :param hidden_dim : ����ά��
        :param embedding_dim : Ƕ��ά��
        :param label_dim : ���ά��
        :param num_layers : �������
        :param pred_hidden_dims : Ԥ��ģ�����ز��ά���б�
        :param concat : �Ƿ��������в�����
        :param bn : �Ƿ�������Ƕ����й�һ��
        """    
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat   # ��ʾ������������в�����,���ھ��ʱ���Ͻڵ���������
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs=1

        self.bias = True
        if args is not None:
            self.bias = args.bias

        # ����build_conv_layers����,���������,������һ���������ز��������һ����
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                input_dim, hidden_dim, embedding_dim, num_layers, 
                add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()    # ʹ��ReLU��Ϊ�����
        self.label_dim = label_dim  # �洢��ǩ��ά��label_dim

        if concat:
            # ����ά��Ϊ���ز��ά�ȳ��Բ�������Ƕ���ά��
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            # �����ΪǶ���ά��
            self.pred_input_dim = embedding_dim
        # ����Ԥ�������
        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims, 
                label_dim, num_aggs=self.num_aggs)

        # ��������ģ����г�ʼ�������ģ����GraphConv����,ʹ��Xavier��ʼ��Ȩ�ز���ƫ�ó�ʼ��Ϊ0
        for m in self.modules():
            if isinstance(m, GraphConv):
                # m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    # m.bias.data = init.constant(m.bias.data, 0.0)
                    nn.init.constant_(m.bias, 0.0)

    # ���������
    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
            normalize=False, dropout=0.0):
        # ������һ��������
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        
        # �����м�ľ����,��ModuleList�洢��������num_layers-2��,ÿ����������ά����ͬ
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=self.bias) 
                 for i in range(num_layers-2)])

        # �������һ����,�����ά��ӳ��Ϊembedding_dim
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last

    # ����Ԥ���
    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:  # ���pred_hidden_dimsΪ��,��ֱ�Ӵ������Բ㽫����ӳ�䵽��ǩά��
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:   # ����,�������ز㲢��Ӽ������ʹ��nn.Sequential��������������,�γ�Ԥ��ģ��
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    # ��������
    def construct_mask(self, max_nodes, batch_num_nodes): 
        ''' 
        For each num_nodes in batch_num_nodes, the first num_nodes entries of the 
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        max_nodes ������������������ڵ���,batch_num_nodes ��һ������ÿ�������ڵ������б�
        ������������Ϊÿ�����εĽڵ�������һ���������,�����ά��Ϊ [batch_size x max_nodes x 1],�����ڴ���ͬ�����Ľڵ�ʱ,�ڸǵ�����Ҫ�Ľڵ㡣
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)

        # �� packed_masks �ж�Ӧ��������䵽 out_tensor ��ǰ num_nodes ��λ��,ʣ�µ�Ԫ�ر���Ϊ 0��
        # ��� batch_num_nodes[i] = 3,��ô out_tensor[i, :3] �ᱻ���Ϊ 1,���ಿ�ֱ���Ϊ 0��
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask

        # unsqueeze()�ڵڶ���ά��֮�� max_nodes ֮������һ��ά��,���������ά��Ϊ [batch_size, max_nodes, 1]
        return out_tensor.unsqueeze(2).cuda()
    
    # ������һ��
    def apply_bn(self, x):
        ''' 
        Batch normalization of 3D tensor x
        ����������x����������һ��,BatchNorm1d���ڹ�һ������ά��
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]

        x : [batch_size x num_nodes x input_dim],ͼ��ÿ���ڵ����������
        adj : [batch_size x num_nodes x num_nodes],ͼ�нڵ�����ӹ�ϵ
        conv_first : ��һ������,���ά��Ϊ[batch_size, num_nodes, hidden_dim]
        conv_block : �м�ľ����,���ά��Ϊ[batch_size, num_nodes, hidden_dim]
        conv_last : ���һ������,���ά��Ϊ[batch_size, num_nodes, embedding_dim]
        embedding_mask : ��ѡ������,����������Ч�ڵ�����
        '''

        x = conv_first(x, adj)
        x = self.act(x) # �����ReLu
        if self.bn:
            x = self.apply_bn(x)
        # ��ʼ��һ���б� x_all,���ڴ洢ÿһ�����������
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
        x : [batch_size x num_nodes x input_dim],ͼ��ÿ���ڵ����������
        adj : [batch_size x num_nodes x num_nodes],ͼ�нڵ�����ӹ�ϵ
        batch_num_nodes : ÿ��ͼ����Ч�ڵ������,������������,�Դ���ͼ�нڵ�������һ�µ����
        num_aggs == 2 : ÿ��ͼ����������ͬʱͨ�����ػ�����ͳػ����ַ�ʽ���ۺ�
        '''
        # mask
        max_num_nodes = adj.size()[1]   # ÿ��ͼ�����ڵ�����
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []    # ���ڴ洢���в�ĳػ����
        out, _ = torch.max(x, dim=1)    # �Ծ��������� x �������ػ� (torch.max) ����,���Žڵ�ά�ȣ�dim=1��ȡ���ֵ,�õ�ÿ��ͼ��ȫ������
        out_all.append(out)
        for i in range(self.num_layers-2):
            x = self.conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out,_ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:  # ��� num_aggs == 2,�������ػ��󻹶�����������ͳػ���torch.sum��,�����ͬ����ӵ� out_all ��
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
            # ���гػ����������ά�ȣ�dim=1������ƴ��,�γ����յ�������ʾ output
            output = torch.cat(out_all, dim=1)
        else:
            # ��ʹ�����һ�����ĳػ������Ϊ�������
            output = out
        
        ypred = self.pred_model(output)
        #print(output.size())
        return ypred

    # ����Ԥ��ֵ����ʵ��ǩ֮�����ʧ,����ģ�͵�ѵ�����Ż�
    def loss(self, pred, label, type='softmax'):
        '''
        pred : Ԥ��ֵ,��״Ϊ [batch_size, label_dim],��ʾÿ��������Ԥ�����
        label : ��ʵ��ǩ,��״Ϊ [batch_size],��ʾÿ����������ʵ���
        type : ��ʧ����,Ĭ��Ϊ 'softmax',��ѡΪ 'softmax' �� 'margin'
        '''
        # softmax + CE
        if type == 'softmax':
            # ��������ʧ���� (F.cross_entropy) ������ʧ
            return F.cross_entropy(pred, label, reduction='mean')
        elif type == 'margin':
            # ���ǩ��Ե��ʧ (margin)�����ڶ��ǩ��������,������ȷ����Ԥ��������������ߡ�
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()    # [batch_size, label_dim]
            # long() ����������Ԫ��ת��Ϊ torch.int64 ����
            # ʹ�� scatter_ ������ label �е������Ϣת��Ϊ���ȱ��� (one-hot encoding)
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

# ��GcnEncoderGraph�Ļ����������ͼ�ĳػ�����(Pooling)�Լ��ڵ����ģ��(Assignment Module),�Ա���ͼ��������н�����ػ�������
# ��ػ���Ŀ����ͨ������ͼ�Ľڵ�����������ͼ�ĸ��Ӷ�,ͬʱ����ͼ��ȫ�ֽṹ����
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

        :param max_num_nodes : ÿ��ͼ�����ڵ���,ȷ���ػ�ʱ�������Ĵ�С
        :param input_dim : ��������ά��
        :param hidden_dim : ���ز�����ά��
        :param embedding_dim : Ƕ������ά��
        :param label_dim : �����ǩ��ά��
        :param num_layers : �������
        :param assign_hidden_dim : �����������ز�ά��
        :param assign_ratio : �ػ�ʱ,�ڵ�������ѹ��������Ĭ��ֵΪ 0.25,��ÿ�γػ��ڵ�������Ϊԭ���� 1/4��
        :param assign_num_layers : ����ģ�� Assignment Module �Ĳ���,Ĭ��ֵΪ -1,��ʾ�� num_layers ��ͬ��
        :param num_pooling : �ػ��Ĵ���,��������ģ�ͽ��м��γػ�����
        :param pred_hidden_dims : Ԥ��ģ�͵����ز�ά��
        :param concat : �Ƿ����в���������������һ��
        :param bn : �Ƿ�ʹ��������һ�� Batch Normalization
        :param dropout : �����һ������Ԫ�ر�,�Է�ֹ�����
        :param linkpred : �Ƿ�ʹ����·Ԥ����ʧ
        :param assign_input_dim : ����ģ�������ά��,Ĭ��ֵΪ -1,��ʹ�� input_dim
        '''

        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat, args=args)
        add_self = not concat   # ��ʾ�Ƿ��ھ������������Խڵ�����
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True  # ���Ʒ�����������ʧ���������򻯣�,Ĭ��ֵΪ True

        # GC
        # ����ģ���б�,���ڴ�ųػ���ÿһ��ľ������
        self.conv_first_after_pool = nn.ModuleList()
        self.conv_block_after_pool = nn.ModuleList()
        self.conv_last_after_pool = nn.ModuleList()
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            # ʹ�� build_conv_layers ����Ϊÿһ�γػ���ľ�������µľ������
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
                    self.pred_input_dim, hidden_dim, embedding_dim, num_layers, 
                    add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(conv_first2)
            self.conv_block_after_pool.append(conv_block2)
            self.conv_last_after_pool.append(conv_last2)

        # assignment
        # ��ʼ����������ά�ȺͲ���,��� assign_num_layers �� assign_input_dim û��ָ��,����ֱ�����Ϊ������� num_layers ������ά�� input_dim
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        # ����ģ���б����ڴ洢�������ĸ��������
        self.assign_conv_first_modules = nn.ModuleList()
        self.assign_conv_block_modules = nn.ModuleList()
        self.assign_conv_last_modules = nn.ModuleList()
        self.assign_pred_modules = nn.ModuleList()
        assign_dim = int(max_num_nodes * assign_ratio)  # ����ػ���,ͼ�нڵ������,�����˳ػ����µĽڵ�����
        # Ϊÿ�γػ�������Ӧ�ķ������ģ�飨�� Assignment Module��,����ÿ��ģ��ĵ�һ�������м���������һ������ӵ���Ӧ��ģ���б���
        for i in range(num_pooling):
            assign_dims.append(assign_dim)  # ��¼ÿ�γػ���ͼ�нڵ������ı仯
            assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
                    assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                    normalize=True)
            assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
            # []���б�,��ʾû�����ز� ֱ�ӽ���������ӳ�䵽�������
            assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)


            # next pooling layer
            # ������һ��ػ��������ͷ���ά��
            assign_input_dim = self.pred_input_dim
            assign_dim = int(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(assign_conv_first)
            self.assign_conv_block_modules.append(assign_conv_block)
            self.assign_conv_last_modules.append(assign_conv_last)
            self.assign_pred_modules.append(assign_pred)

        # �������յ�Ԥ��ģ�͡��ػ���ÿһ�����������ƴ����һ��,�γ����յ�������ʾ���뵽Ԥ��ģ����
        self.pred_model = self.build_pred_layers(self.pred_input_dim * (num_pooling+1), pred_hidden_dims, 
                label_dim, num_aggs=self.num_aggs)

        # �����е� GraphConv ģ�����Ȩ�س�ʼ��,ʹ�� Xavier ��ʼ������,����ƫ�ó�ʼ��Ϊ 0
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def forward(self, x, adj, batch_num_nodes, **kwargs):
        if 'assign_x' in kwargs:
            # ����������������x_a,����x_a��x��ͬ
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
        # ��ͨ��ԭʼ GcnEncoderGraph �е�ͼ���ģ��������������о������,�õ�Ƕ������ embedding_tensor
        embedding_tensor = self.gcn_forward(x, adj,
                self.conv_first, self.conv_block, self.conv_last, embedding_mask)

        out, _ = torch.max(embedding_tensor, dim=1) # �ؽڵ�ά��ȡ���ֵ,����ÿ��ͼ�����нڵ��е��������ֵ,��Ϊ���ػ�
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)    # �ؽڵ�ά�ȶ����нڵ��������,��ȡͼ���������ֵ,��˳�Ϊ��ͳػ�
            out_all.append(out)

        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None

            # ʹ�÷������ģ�飨Assignment Module������ͼ���,�õ��ڵ���ػ���Ľڵ�ķ������
            self.assign_tensor = self.gcn_forward(x_a, adj, 
                    self.assign_conv_first_modules[i], self.assign_conv_block_modules[i], self.assign_conv_last_modules[i],
                    embedding_mask)
            # [batch_size x num_nodes x next_lvl_num_nodes]
            # �Է������Ӧ�� Softmax ����,��ʾÿ��ԭʼ�ڵ��ڳػ���Ľڵ��з����Ȩ��
            self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred_modules[i](self.assign_tensor))
            # �Է������Ӧ������,���ε���Ч�Ľڵ����Ȩ��
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask

            # update pooled features and adj matrix
            # ���³ػ��������x���ڽӾ���adj
            # self.assign_tensor ��[batch_size, num_nodes, next_lvl_num_nodes] 
            # embedding_tensor��[batch_size x num_nodes x embedding_dim]
            # torch.transpose(self.assign_tensor, 1, 2)�Ծ���ĵ�1ά�͵ڶ�ά����ת��,Ȼ����embedding_tensor�������
            # x��[batch_size, next_lvl_num_nodes, embedding_dim] 
            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)     # ��Ӧ��ʽ3
            # adj����ǰͼ���ڽӾ���,��״Ϊ [batch_size, num_nodes, num_nodes]
            # �µ��ڽӾ��� adj��״Ϊ[batch_size, next_lvl_num_nodes, next_lvl_num_nodes],��ʾ�ػ�����ڽӹ�ϵ
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor  # ��Ӧ��ʽ4
            x_a = x
        
            # �Գػ��������x��adj�����µ�ͼ�������
            embedding_tensor = self.gcn_forward(x, adj, 
                    self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                    self.conv_last_after_pool[i])


            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                #out = torch.mean(embedding_tensor, dim=1)
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)


        # �����гػ�������ƴ����һ��,�γ����յ�������ʾ output
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        # ʹ��Ԥ��ģ�Ͷ� output ����Ԥ��,�õ����յ���� ypred
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

