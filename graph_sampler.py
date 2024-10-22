import networkx as nx   # 处理图数据结构
import numpy as np
import torch
import torch.utils.data

import util

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft YaHei'
# 增加日志
import logging
logger = logging.getLogger(__name__)

class GraphSampler(torch.utils.data.Dataset):
    # 用于将一组图数据处理成适合神经网络训练的数据格式,并采样
    ''' Sample graphs and nodes in graph
    '''
    def __init__(self, G_list, features='default', normalize=True, assign_feat='default', max_num_nodes=0):
        '''
        G_list:每个图是一个图对象
        normalize:是否对邻接矩阵归一化
        assign_feat:用于分配节点的特征,可以是id或default
        max_num_nodes:设置图中的最大节点数
        '''
        self.adj_all = []
        self.len_all = []
        self.feature_all = []
        self.label_all = []
        
        self.assign_feat_all = []

        if max_num_nodes == 0:
            self.max_num_nodes = max([G.number_of_nodes() for G in G_list])
        else:
            self.max_num_nodes = max_num_nodes

        #if features == 'default':
        self.feat_dim = util.node_dict(G_list[0])[0]['feat'].shape[0]  
        # 获取图第一个图G_list[0]的节点字典,访问节点0的属性字典,获取该节点的feat(特征向量),最后通过.shape[0]获取该特征向量的维度
        # print(f"GraphSampler feat_dim : {self.feat_dim}")
        for G in G_list:
            # 对每个图G,获取邻接矩阵并转换为numpy数组
            adj = np.array(nx.adjacency_matrix(G).todense())
            # adj = np.array(nx.to_numpy_matrix(G)) # network库中使用的to_numpy_matrix已经被弃用
            if normalize:
                sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
                adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
            # 将处理后的邻接矩阵、节点数量、图的标签添加到相应的列表中
            self.adj_all.append(adj)
            self.len_all.append(G.number_of_nodes())
            self.label_all.append(G.graph['label'])
            # feat matrix: max_num_nodes x feat_dim
            # arg_parse feature_type='default'
            # 将每个节点的特征存储到一个矩阵中
            if features == 'default':
                # 创建零矩阵,存储节点特征
                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i,u in enumerate(G.nodes()):
                    f[i,:] = util.node_dict(G)[u]['feat']
                self.feature_all.append(f)
            elif features == 'id':
                self.feature_all.append(np.identity(self.max_num_nodes))
            elif features == 'deg-num':
                degs = np.sum(np.array(adj), 1)
                degs = np.expand_dims(np.pad(degs, [0, self.max_num_nodes - G.number_of_nodes()], 0),
                                      axis=1)
                self.feature_all.append(degs)
            elif features == 'deg':
                self.max_deg = 10
                degs = np.sum(np.array(adj), 1).astype(int)
                degs[degs>max_deg] = max_deg
                feat = np.zeros((len(degs), self.max_deg + 1))
                feat[np.arange(len(degs)), degs] = 1
                feat = np.pad(feat, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                        'constant', constant_values=0)

                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i,u in enumerate(util.node_iter(G)):
                    f[i,:] = util.node_dict(G)[u]['feat']

                feat = np.concatenate((feat, f), axis=1)

                self.feature_all.append(feat)
            elif features == 'struct':
                self.max_deg = 10
                degs = np.sum(np.array(adj), 1).astype(int)
                degs[degs>10] = 10
                feat = np.zeros((len(degs), self.max_deg + 1))
                feat[np.arange(len(degs)), degs] = 1
                degs = np.pad(feat, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                        'constant', constant_values=0)

                clusterings = np.array(list(nx.clustering(G).values()))
                clusterings = np.expand_dims(np.pad(clusterings, 
                                                    [0, self.max_num_nodes - G.number_of_nodes()],
                                                    'constant'),
                                             axis=1)
                g_feat = np.hstack([degs, clusterings])
                if 'feat' in util.node_dict(G)[0]:
                    node_feats = np.array([util.node_dict(G)[i]['feat'] for i in range(G.number_of_nodes())])
                    node_feats = np.pad(node_feats, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                                        'constant')
                    g_feat = np.hstack([g_feat, node_feats])

                self.feature_all.append(g_feat)

            if assign_feat == 'id':
                self.assign_feat_all.append(
                        np.hstack((np.identity(self.max_num_nodes), self.feature_all[-1])) )
            else:
                self.assign_feat_all.append(self.feature_all[-1])
            
        self.feat_dim = self.feature_all[0].shape[1]
        self.assign_feat_dim = self.assign_feat_all[0].shape[1]

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        # use all nodes for aggregation (baseline)
        '''
        adj: (64, 64),
        feats: (64, 10),
        label: 0,
        num_nodes: 60,
        assign_feats: (64, 10)
        '''
        print(f"adj: {adj_padded.shape if hasattr(adj_padded, 'shape') else adj_padded},\n"
            f"feats: {self.feature_all[idx].copy().shape if hasattr(self.feature_all[idx], 'shape') else self.feature_all[idx]},\n"
            f"label: {self.label_all[idx]},\n"  # 去掉 .shape，因为它是一个整数
            f"num_nodes: {num_nodes if isinstance(num_nodes, (int, float)) else num_nodes.shape},\n"
            f"assign_feats: {self.assign_feat_all[idx].copy().shape if hasattr(self.assign_feat_all[idx], 'shape') else self.assign_feat_all[idx]}")
        logging.info(f"adj: {adj_padded.shape if hasattr(adj_padded, 'shape') else adj_padded},\n"
            f"feats: {self.feature_all[idx].copy().shape if hasattr(self.feature_all[idx], 'shape') else self.feature_all[idx]},\n"
            f"label: {self.label_all[idx]},\n"  # 去掉 .shape，因为它是一个整数
            f"num_nodes: {num_nodes if isinstance(num_nodes, (int, float)) else num_nodes.shape},\n"
            f"assign_feats: {self.assign_feat_all[idx].copy().shape if hasattr(self.assign_feat_all[idx], 'shape') else self.assign_feat_all[idx]}")
        
        return {'adj':adj_padded,
                'feats':self.feature_all[idx].copy(),
                'label':self.label_all[idx],
                'num_nodes': num_nodes,
                'assign_feats':self.assign_feat_all[idx].copy()}
    # feat即最终的h0 = Variable(data['feats'].float(), requires_grad=False).cuda()，SoftPoolingGcnEncoder的forward的参数x

