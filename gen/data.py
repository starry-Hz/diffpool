#-*- coding : utf-8-*-
# coding:unicode_escape
import networkx as nx
import numpy as np
import random

import gen.feat as featgen
import util

# Barabási–Albert（BA）模型图
def gen_ba(n_range, m_range, num_graphs, feature_generator=None):
    # n_range 和 m_range 分别定义了生成图的节点数（i）和每个新节点连接到现有节点的边数（j）的范围
    # 定义了生成图的数量
    '''
    n_range: 节点数量的范围
    m_range: 每个新节点连接到现有节点的边数的范围
    num_graphs: 要生成的图的数量
    feature_generator: 用于生成节点特征的生成器
    生成 num_graphs 个 Barabási–Albert 模型图，图的节点数量从 n_range 中选择，边数从 m_range 中选择,使用特性生成器为每个节点生成特征
    Barabási–Albert 模型是一种无尺度网络模型，生成的图符合无尺度特性，适用于模拟现实中的复杂网络（如社交网络）
    '''
    graphs = []
    for i in np.random.choice(n_range, num_graphs):
        for j in np.random.choice(m_range, 1):
            # 生成具有 i 个节点和 j 条边的无向图
            graphs.append(nx.barabasi_albert_graph(i,j))

    if feature_generator is None:
        feature_generator = ConstFeatureGen(0)
    for G in graphs:
        feature_generator.gen_node_features(G)
    return graphs

# Erd?s–Rényi（ER）图
def gen_er(n_range, p, num_graphs, feature_generator=None):
    '''
    n_range: 节点数量的范围
    p: 边的生成概率
    num_graphs: 要生成的图的数量
    feature_generator: 用于生成节点特征的生成器
    使用 networkx 的 erdos_renyi_graph 函数生成随机图，节点数量从 n_range 中选取，边的生成概率为 p
    使用特性生成器为每个节点生成特征
    Erd?s–Rényi 模型用于生成随机图，每对节点之间有固定的概率 p 存在一条边
    '''
    graphs = []
    for i in np.random.choice(n_range, num_graphs):
        graphs.append(nx.erdos_renyi_graph(i,p))

    if feature_generator is None:
        feature_generator = ConstFeatureGen(0)
    for G in graphs:
        feature_generator.gen_node_features(G)
    return graphs

# 社区结构的图
def gen_2community_ba(n_range, m_range, num_graphs, inter_prob, feature_generators):
    ''' Each community is a BA graph.
    Args:
        inter_prob: probability of one node connecting to any node in the other community.
    生成两个社区结构,每个社区都是一个BA图
    inter_prob: 两个社区之间连接的概率。
    feature_generators: 节点特征生成器列表，分别用于每个社区。
    逻辑:分别生成两个BA图graphs0和graphs1,将两个图合并为一个图,使它们之间没有共有节点,在两个社区之间添加边,边的添加概率由inter_prob决定
    '''

    if feature_generators is None:
        mu0 = np.zeros(10)
        mu1 = np.ones(10)
        sigma0 = np.ones(10, 10) * 0.1
        sigma1 = np.ones(10, 10) * 0.1
        fg0 = GaussianFeatureGen(mu0, sigma0)
        fg1 = GaussianFeatureGen(mu1, sigma1)
    else:
        fg0 = feature_generators[0]
        fg1 = feature_generators[1] if len(feature_generators) > 1 else feature_generators[0]

    graphs1 = []
    graphs2 = []
    #for (i1, i2) in zip(np.random.choice(n_range, num_graphs), 
    #                    np.random.choice(n_range, num_graphs)):
    #    for (j1, j2) in zip(np.random.choice(m_range, num_graphs), 
    #                        np.random.choice(m_range, num_graphs)):
    graphs0 = gen_ba(n_range, m_range, num_graphs, fg0)
    graphs1 = gen_ba(n_range, m_range, num_graphs, fg1)
    graphs = []
    for i in range(num_graphs):
        G = nx.disjoint_union(graphs0[i], graphs1[i])
        n0 = graphs0[i].number_of_nodes()
        for j in range(n0):
            if np.random.rand() < inter_prob:
                target = np.random.choice(G.number_of_nodes() - n0) + n0
                G.add_edge(j, target)
        graphs.append(G)
    return graphs

# 生成具有两层次结构的图,每个社区是一个 BA 图
def gen_2hier(num_graphs, num_clusters, n, m_range, inter_prob1, inter_prob2, feat_gen):
    ''' Each community is a BA graph.
    Args:
        inter_prob1: probability of one node connecting to any node in the other community within
            the large cluster.
        inter_prob2: probability of one node connecting to any node in the other community between
            the large cluster.
    inter_prob1: 同一大簇内的不同社区之间连接的概率。
    inter_prob2: 不同大簇之间连接的概率。
    生成多个具有社区结构的簇,每个簇内部生成多个随机图,使用 nx.disjoint_union_all 将每个簇的图合并在一起,
    在同一个簇内和不同簇之间添加边,连接概率分别为inter_prob1和inter_prob2,最终生成一个具有层次化结构的图,适合用于模拟具有复杂层次化组织的网络
    '''
    graphs = []

    for i in range(num_graphs):
        clusters2 = []
        for j in range(len(num_clusters)):
            clusters = gen_er(range(n, n+1), 0.5, num_clusters[j], feat_gen[0])
            G = nx.disjoint_union_all(clusters)
            for u1 in range(G.number_of_nodes()):
                if np.random.rand() < inter_prob1:
                    target = np.random.choice(G.number_of_nodes() - n)
                    # move one cluster after to make sure it's not an intra-cluster edge
                    if target // n >= u1 // n:
                        target += n
                    G.add_edge(u1, target)
            clusters2.append(G)
        G = nx.disjoint_union_all(clusters2)
        cluster_sizes_cum = np.cumsum([cluster2.number_of_nodes() for cluster2 in clusters2])
        curr_cluster = 0
        for u1 in range(G.number_of_nodes()):
            if u1 >= cluster_sizes_cum[curr_cluster]:
                curr_cluster += 1
            if np.random.rand() < inter_prob2:
                target = np.random.choice(G.number_of_nodes() -
                        clusters2[curr_cluster].number_of_nodes())
                # move one cluster after to make sure it's not an intra-cluster edge
                if curr_cluster == 0 or target >= cluster_sizes_cum[curr_cluster - 1]:
                    target += cluster_sizes_cum[curr_cluster]
            G.add_edge(u1, target)
        graphs.append(G)

    return graphs

