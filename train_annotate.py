# -*- coding: utf-8 -*-

import matplotlib
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']     # 显示中文
# # 为了坐标轴负号正常显示。matplotlib默认不支持中文，设置中文字体后，负号会显示异常。需要手动将坐标轴负号设为False才能正常显示负号。
# matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.colors as colors
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft YaHei'
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorboardX
from tensorboardX import SummaryWriter

import argparse
import os
import pickle
import random
import shutil
import time
import seaborn
import seaborn as sns
# 增加日志
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='./log/diffpool_annotate_20241005.log',
                    filemode='a')

# 引入diffpool相关py
import cross_val
import encoders
import gen.feat as featgen
import gen.data as datagen
from graph_sampler import GraphSampler
import load_data
import util

# 评价函数
def evaluate(dataset, model, args, name='Validation', max_num_examples=None):
    """
    :param args : 训练的参数配置
    :param max_num_examples : 限制评估的最大样本数
    """
    # 将模型设置为评估模式,可以关闭模型中的一些特殊层
    model.eval()

    labels = []
    preds = []
    # 遍历数据集,逐批次处理数据
    for batch_idx, data in enumerate(dataset):
        # 从data中提取图的邻接矩阵adj、节点特征h0、标签label、节点数batch_num_nodes、分配矩阵assign_feats
        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        h0 = Variable(data['feats'].float()).cuda()
        labels.append(data['label'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()

        # 进行模型预测,得到预测输出
        ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)    # 获取每个样本的预测类别索引
        preds.append(indices.cpu().data.numpy())

        # 判断最大样本数。当前批次若超过最大样本数则提前终止循环
        if max_num_examples is not None:
            if (batch_idx+1)*args.batch_size > max_num_examples:
                break

    # np.hstack()将多个批次的标签和预测拼接成一个完整的列表
    labels = np.hstack(labels)
    preds = np.hstack(preds)
    
    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro")}
    print(name, " accuracy:", result['acc'])
    logging.info("%s accuracy: %s", name, result['acc'])
    return result

# 生成独特的字符串前缀
def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname
    else:
        name = args.dataset
    # 在name后加上下划线和方法名称
    name += '_' + args.method
    if args.method == 'soft-assign':
        # 添加图卷积层数和池化层数  l表示层数
        name += '_l' + str(args.num_gc_layers) + 'x' + str(args.num_pool)
        # 添加分配比例
        name += '_ar' + str(int(args.assign_ratio*100))
        # 是否开启链路预测linkpred
        if args.linkpred:
            name += '_lp'
    else:
        name += '_l' + str(args.num_gc_layers)
    
    # 隐藏层维度_h  输出层维度_o
    name += '_h' + str(args.hidden_dim) + '_o' + str(args.output_dim)
    if not args.bias:
        name += '_nobias'
    if len(args.name_suffix) > 0:
        name += '_' + args.name_suffix
    return name

# 保存训练图像的文件名路径
def gen_train_plt_name(args):
    return 'results/' + gen_prefix(args) + '.png'

# 将模型的分配张量可视化并记录到Tensorboard中,用于可视化节点分配矩阵assign_tensor
def log_assignment(assign_tensor, writer, epoch, batch_idx):
    """
    :param assign_tensor : 分配张量
    :param epoch : 当前的训练轮次
    :param batch_idx : 批次索引
    """
    plt.switch_backend('agg')   # 将绘图存储为图像文件,不直接显示
    fig = plt.figure(figsize=(8,6), dpi=300)

    # has to be smaller than args.batch_size
    # 可视化节点与簇之间的关系,了解每个节点在不同簇中的分配情况
    for i in range(len(batch_idx)):
        plt.subplot(2, 2, i+1)
        plt.imshow(assign_tensor.cpu().data.numpy()[batch_idx[i]], cmap=plt.get_cmap('BuPu'))
        cbar = plt.colorbar()   # 在每个子图旁边添加颜色条
        cbar.solids.set_edgecolor("face")   # 将颜色条的边界线设置为与面部颜色一致
    plt.tight_layout()
    fig.canvas.draw()

    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # 将 matplotlib 图形对象 fig 转换为一个图像格式，以便写入到 TensorBoard
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('assignment', data, epoch)

# 将图结构的数据邻接矩阵adj可视化并记录到tensorboard中,以便在训练过程中观察图的结构和分配情况
# 用于可视化图的拓扑结构,展示节点之间的连接关系
def log_graph(adj, batch_num_nodes, writer, epoch, batch_idx, assign_tensor=None):
    """
    :param adj : 邻接矩阵
    :param batch_num_nodes : 每个批次中图的节点数量
    :param batch_idx : 批次索引
    """
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8,6), dpi=300)

    # 展示图谱的拓扑结果,图中的节点和边,图的连接关系
    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i+1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()
        # 使用 networkx 的 from_numpy_matrix 函数将邻接矩阵转换为图对象 G
        G = nx.from_numpy_matrix(adj_matrix)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color='#336699',
                edge_color='grey', width=0.5, node_size=300,
                alpha=0.7)
        ax.xaxis.set_visible(False)

    plt.tight_layout()
    fig.canvas.draw()

    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('graphs', data, epoch)

    # log a label-less version
    #fig = plt.figure(figsize=(8,6), dpi=300)
    #for i in range(len(batch_idx)):
    #    ax = plt.subplot(2, 2, i+1)
    #    num_nodes = batch_num_nodes[batch_idx[i]]
    #    adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()
    #    G = nx.from_numpy_matrix(adj_matrix)
    #    nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color='#336699',
    #            edge_color='grey', width=0.5, node_size=25,
    #            alpha=0.8)

    #plt.tight_layout()
    #fig.canvas.draw()

    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #writer.add_image('graphs_no_label', data, epoch)

    # colored according to assignment
    # 用颜色区分节点的分配类别  assign_tensor   便于理解节点在聚类后的分布情况
    assignment = assign_tensor.cpu().data.numpy()
    fig = plt.figure(figsize=(8,6), dpi=300)

    num_clusters = assignment.shape[2]
    all_colors = np.array(range(num_clusters))

    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i+1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()

        # 使用argmax获取每个节点的分配类别,并生成对应的颜色
        label = np.argmax(assignment[batch_idx[i]], axis=1).astype(int)
        label = label[: batch_num_nodes[batch_idx[i]]]
        node_colors = all_colors[label]

        G = nx.from_numpy_matrix(adj_matrix)
        # nx.draw()对节点进行着色   cmap=plt.get_cmap('Set1')选择颜色映射,确保不同类别的节点具有不同的颜色
        nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color=node_colors,
                edge_color='grey', width=0.4, node_size=50, cmap=plt.get_cmap('Set1'),
                vmin=0, vmax=num_clusters-1,
                alpha=0.8)

    plt.tight_layout()
    fig.canvas.draw()

    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('graphs_colored', data, epoch)

# 对模型进行训练+评估
def train(dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
        mask_nodes = True):
    """
    :param args : 训练参数
    :param same_feat : 是否使用相同的特征
    :param val_dataset : 验证数据集
    :param test_dataset : 测试数据集
    :param writer : TensorBoard 的记录器 (SummaryWriter)
    :param mask_nodes : 是否屏蔽节点（用于有无节点数信息的场景）
    """
    writer_batch_idx = [0, 3, 6, 9] # 指定Tensorboard记录的批次索引
    # 使用Adam优化器
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.001)
    iter = 0
    best_val_result = {
            'epoch': 0,
            'loss': 0,
            'acc': 0}
    test_result = {
            'epoch': 0,
            'loss': 0,
            'acc': 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []
    for epoch in range(args.num_epochs):
        total_time = 0
        avg_loss = 0.0
        model.train()   # 用于启用 dropout 和 batchnorm
        print('Epoch: ', epoch)
        logging.info('Epoch: %s', epoch)
        for batch_idx, data in enumerate(dataset):
            # 记录每个批次的训练时间
            begin_time = time.time()
            model.zero_grad()   # 清除前一次迭代中计算的梯度
            # 从 data 中提取出邻接矩阵 (adj)、节点特征 (h0)、标签 (label) 以及节点数量 (batch_num_nodes) 和分配特征 (assign_input)
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            label = Variable(data['label'].long()).cuda()
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()

            ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            # 根据不同的训练方法计算损失函数
            if not args.method == 'soft-assign' or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)
            loss.backward()
            # clip_grad_norm_()进行梯度裁剪,防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()    # 更新模型的参数
            iter += 1
            avg_loss += loss    # 累加损失,用于计算平均损失
            #if iter % 20 == 0:
            #    print('Iter: ', iter, ', loss: ', loss.data[0])
            elapsed = time.time() - begin_time  # 计算每个批次的训练时间
            total_time += elapsed   # 将每个批次的时候叠加到总时间中

            # log once per XX epochs
            # 每10个epoch在训练数据的中间批次时,使用log_assignment()和log_graph()记录节点的分配情况和图的结构
            if epoch % 10 == 0 and batch_idx == len(dataset) // 2 and args.method == 'soft-assign' and writer is not None:
                log_assignment(model.assign_tensor, writer, epoch, writer_batch_idx)
                if args.log_graph:
                    log_graph(adj, batch_num_nodes, writer, epoch, writer_batch_idx, model.assign_tensor)
        # 计算每个epoch的平均损失,并记录损失变化趋势
        avg_loss /= batch_idx + 1
        if writer is not None:
            writer.add_scalar('loss/avg_loss', avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar('loss/linkpred_loss', model.link_loss, epoch)
        print('Avg loss: ', avg_loss, '; epoch time: ', total_time)
        logging.info('Avg loss: %s; epoch time: %s', avg_loss, total_time)

        # 评估模型在训练集中的性能
        result = evaluate(dataset, model, args, name='Train', max_num_examples=100)
        train_accs.append(result['acc'])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model, args, name='Validation')
            val_accs.append(val_result['acc'])
        
        # 更新最佳验证结果：如果当前验证集准确率高于之前记录的最佳验证准确率（考虑浮点数的微小误差），则更新最佳验证结果
        if val_result['acc'] > best_val_result['acc'] - 1e-7:
            best_val_result['acc'] = val_result['acc']
            best_val_result['epoch'] = epoch
            best_val_result['loss'] = avg_loss

        # 测试集评估：如果提供了测试集，则调用 evaluate() 函数评估模型在测试集上的性能。
        if test_dataset is not None:
            test_result = evaluate(test_dataset, model, args, name='Test')
            test_result['epoch'] = epoch
        if writer is not None:
            writer.add_scalar('acc/train_acc', result['acc'], epoch)
            writer.add_scalar('acc/val_acc', val_result['acc'], epoch)
            writer.add_scalar('loss/best_val_loss', best_val_result['loss'], epoch)
            if test_dataset is not None:
                writer.add_scalar('acc/test_acc', test_result['acc'], epoch)

        print('Best val result: ', best_val_result)
        logging.info('Best val result: %s', best_val_result)
        best_val_epochs.append(best_val_result['epoch'])
        best_val_accs.append(best_val_result['acc'])

        # 若提供测试集,衡量模型
        if test_dataset is not None:
            print('Test result: ', test_result)
            logging.info('Test result: %s', test_result)
            test_epochs.append(test_result['epoch'])
            test_accs.append(test_result['acc'])

    # matplotlib.style.use('seaborn')
    # sns.set()
    plt.style.use('seaborn')
    plt.switch_backend('agg')
    plt.figure()
    plt.plot(train_epochs, util.exp_moving_avg(train_accs, 0.85), '-', lw=1)
    # 如果提供了测试数据集，则绘制验证集和测试集的准确率曲线
    if test_dataset is not None:
        plt.plot(best_val_epochs, best_val_accs, 'bo', test_epochs, test_accs, 'go')
        plt.legend(['train', 'val', 'test'])
    else:
        plt.plot(best_val_epochs, best_val_accs, 'bo')
        plt.legend(['train', 'val'])
    
    # 获取当前文件夹,并将结果保持至results中
    current_dir = os.getcwd()
    save_path = os.path.join(current_dir,'results')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(gen_train_plt_name(args), dpi=600)
    plt.close()
    # matplotlib.style.use('default')
    plt.style.use('default')

    return model, val_accs

# 将图数据划分为训练、验证和测试集
def prepare_data(graphs, args, test_graphs=None, max_nodes=0):
    """
    :param graphs : 图数据    
    :param args : 训练参数
    :param test_graphs : 测试数据集
    :param max_nodes : 每个图中的最大节点数
    """
    random.shuffle(graphs)  # 随机打乱图数据,避免数据顺序对训练结果产生影响
    # 训练集划分
    if test_graphs is None:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1-args.test_ratio))
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx: test_idx]
        test_graphs = graphs[test_idx:]
    else:
        train_idx = int(len(graphs) * args.train_ratio)
        train_graphs = graphs[:train_idx]
        val_graphs = graph[train_idx:]
    print('Num training graphs: ', len(train_graphs), 
          '; Num validation graphs: ', len(val_graphs),
          '; Num testing graphs: ', len(test_graphs))
    logging.info('Num training graphs: %d; Num validation graphs: %d; Num testing graphs: %d', 
             len(train_graphs), len(val_graphs), len(test_graphs))

    print('Number of graphs: ', len(graphs))
    logging.info('Number of graphs: %d', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    logging.info('Number of edges: %d', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ', 
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))
    logging.info('Max, avg, std of graph size: %d, %.2f, %.2f', 
             max([G.number_of_nodes() for G in graphs]), 
             np.mean([G.number_of_nodes() for G in graphs]), 
             np.std([G.number_of_nodes() for G in graphs]))

    # minibatch
    # 对训练集,验证集,测试集创建数据采样
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(test_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    test_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, test_dataset_loader, \
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim

# 生成两类图结构数据,创建不同类型的GNN模型,最终对这些数据进行训练
def syn_community1v2(args, writer=None, export_graphs=False):
    """
    :param export_graphs : 是否导出图结构的可视化图像    
    """
    # data
    # 生成两种图结构的数据 graphs1 和 graphs2
    # 生成一组 BA 图
    graphs1 = datagen.gen_ba(range(40, 60), range(4, 5), 500, 
            featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float)))
    for G in graphs1:
        G.graph['label'] = 0
    if export_graphs:
        util.draw_graph_list(graphs1[:16], 4, 4, 'figs/ba')

    # 生成了另一组具有两类社区结构的 BA 图
    graphs2 = datagen.gen_2community_ba(range(20, 30), range(4, 5), 500, 0.3, 
            [featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))])
    for G in graphs2:
        G.graph['label'] = 1
    if export_graphs:
        util.draw_graph_list(graphs2[:16], 4, 4, 'figs/ba2')

    graphs = graphs1 + graphs2
    
    # max_num_nodes图中节点的最大数目, input_dim输入维度, assign_input_dim分配输入维度
    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = prepare_data(graphs, args)
    # soft-assign适用于对图进行池化
    if args.method == 'soft-assign':
        print('Method: soft-assign')
        logging.info('Method: soft-assign')
        model = encoders.SoftPoolingGcnEncoder(
                max_num_nodes, 
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                bn=args.bn, linkpred=args.linkpred, assign_input_dim=assign_input_dim).cuda()
    elif args.method == 'base-set2set':
        print('Method: base-set2set')   # Set2Set,从图中提取全局信息
        logging.info('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(input_dim, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn).cuda()
    else:
        print('Method: base')
        logging.info('Method: base')
        model = encoders.GcnEncoderGraph(input_dim, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn).cuda()

    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
            writer=writer)

# 生成了3类图结构,并对图进行训练
def syn_community2hier(args, writer=None):

    # data
    feat_gen = [featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))]  # 创建常量特征生成器
    # graphs1 两层次结构图,图中有 1000 个节点，两个层次分别有 2 和 4 个子社区，10 表示每个节点的边连接数量，
    # range(4,5) 表示每个新节点连接的已有节点数目，0.1 和 0.03 分别表示不同层次社区之间的边缘连接概率。
    graphs1 = datagen.gen_2hier(1000, [2,4], 10, range(4,5), 0.1, 0.03, feat_gen)
    graphs2 = datagen.gen_2hier(1000, [3,3], 10, range(4,5), 0.1, 0.03, feat_gen)
    graphs3 = datagen.gen_2community_ba(range(28, 33), range(4,7), 1000, 0.25, feat_gen)

    # 将每类图的数据标签设置为不同的类比
    for G in graphs1:
        G.graph['label'] = 0
    for G in graphs2:
        G.graph['label'] = 1
    for G in graphs3:
        G.graph['label'] = 2

    graphs = graphs1 + graphs2 + graphs3    # 合并为一个包含所有图的列表

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = prepare_data(graphs, args)

    if args.method == 'soft-assign':
        print('Method: soft-assign')
        logging.info('Method: soft-assign')
        model = encoders.SoftPoolingGcnEncoder(
                max_num_nodes, 
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                bn=args.bn, linkpred=args.linkpred, args=args, assign_input_dim=assign_input_dim).cuda()
    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        logging.info('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(input_dim, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn, args=args, assign_input_dim=assign_input_dim).cuda()
    else:
        print('Method: base')
        logging.info('Method: base')
        model = encoders.GcnEncoderGraph(input_dim, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn, args=args).cuda()
    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
            writer=writer)

# 对 .pkl 文件中存储的图数据进行训练和评估
def pkl_task(args, feat=None):
    # 读取.pkl文件中的数据
    with open(os.path.join(args.datadir, args.pkl_fname), 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    graphs = data[0]
    labels = data[1]
    test_graphs = data[2]
    test_labels = data[3]

    # 对训练、测试数据中的每个图,为图设置对应的标签
    for i in range(len(graphs)):
        graphs[i].graph['label'] = labels[i]
    for i in range(len(test_graphs)):
        test_graphs[i].graph['label'] = test_labels[i]

    #  使用常量特征生成器生成节点特征
    if feat is None:
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        # 为每个图的节点生成特征
        for G in graphs:
            featgen_const.gen_node_features(G)
        for G in test_graphs:
            featgen_const.gen_node_features(G)

    train_dataset, test_dataset, max_num_nodes = prepare_data(graphs, args, test_graphs=test_graphs)
    # 创建图卷积模型,进行训练和评估
    model = encoders.GcnEncoderGraph(
            args.input_dim, args.hidden_dim, args.output_dim, args.num_classes, 
            args.num_gc_layers, bn=args.bn).cuda()
    train(train_dataset, model, args, test_dataset=test_dataset)
    evaluate(test_dataset, model, args, 'Validation')

# 加载图数据,准备节点特征,使用指定的图神经网络GNN模型对图数据进行训练和评估
def benchmark_task(args, writer=None, feat='node-label'):
    # 读取数据
    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)
    
    # 根据参数feat指定的节点特征类型进行节点特征的生成
    if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:    # 如果图中包含节点特征,则直接使用
        print('Using node features')
        logging.info('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in graphs[0].node[0]:    # 如果图中节点具有标签,则使用标签作为节点的特征
        print('Using node labels')
        logging.info('Using node labels')
        for G in graphs:
            for u in G.nodes():
                G.node[u]['feat'] = np.array(G.node[u]['label'])
    else:   # 如果图中既没有特征也没有标签,使用常量特征生成器        
        print('Using constant labels')
        logging.info('Using constant labels')
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    # 准备数据
    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = \
            prepare_data(graphs, args, max_nodes=args.max_nodes)

    # 根据args.method指定的训练方法选择不同的模型
    if args.method == 'soft-assign':
        print('Method: soft-assign')
        logging.info('Method: soft-assign')
        model = encoders.SoftPoolingGcnEncoder(
                max_num_nodes, 
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                assign_input_dim=assign_input_dim).cuda()
    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        logging.info('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(
                input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()
    else:
        print('Method: base')
        logging.info('Method: base')
        model = encoders.GcnEncoderGraph(
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, 
                args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()

    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
            writer=writer)
    evaluate(test_dataset, model, args, 'Validation')

# 用于图神经网络模型的交叉验证的基准任务
def benchmark_task_val(args, writer=None, feat='node-label'):
    """
    :param args : 训练参数
    :param writer : TensorBoard 的记录器 (SummaryWriter)
    :param feat : 指定使用哪种类型的节点特征
    """
    all_vals = []
    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)  # 从文件中读取图数据

    example_node = util.node_dict(graphs[0])[0] # 从第一个图中获取第一个节点的字典信息,用于判断节点特征类型
    
    # 根据feat指定的节点特征类型进行节点特征的生成
    if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        logging.info('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in example_node:
        print('Using node labels')
        logging.info('Using node labels')
        for G in graphs:
            for u in G.nodes():
                util.node_dict(G)[u]['feat'] = np.array(util.node_dict(G)[u]['label'])
    else:
        print('Using constant labels')
        logging.info('Using constant labels')
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    # 进行10次交叉验证,通过cross_val.prepare_val_data对数据集进行划分
    for i in range(10):
        train_dataset, val_dataset, max_num_nodes, input_dim, assign_input_dim = \
                cross_val.prepare_val_data(graphs, args, i, max_nodes=args.max_nodes)
        if args.method == 'soft-assign':
            print('Method: soft-assign')
            logging.info('Method: soft-assign')
            model = encoders.SoftPoolingGcnEncoder(
                    max_num_nodes, 
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                    args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                    bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                    assign_input_dim=assign_input_dim).cuda()
        elif args.method == 'base-set2set':
            print('Method: base-set2set')
            logging.info('Method: base-set2set')
            model = encoders.GcnSet2SetEncoder(
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                    args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()
        else:
            print('Method: base')
            logging.info('Method: base')
            model = encoders.GcnEncoderGraph(
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, 
                    args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()

        # 进行训练并将准确率添加到 all_vals 中
        _, val_accs = train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=None,
            writer=writer)
        all_vals.append(np.array(val_accs))
    all_vals = np.vstack(all_vals)  # 将每次交叉验证的结果按列堆叠起来
    all_vals = np.mean(all_vals, axis=0)
    print(all_vals)
    print(np.max(all_vals))
    print(np.argmax(all_vals))  # 验证准确率在哪一轮交叉验证中达到了最大值
    logging.info(all_vals)
    logging.info(np.max(all_vals))
    logging.info(np.argmax(all_vals))
    
# 用于解析命令行参数   
def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')    # 创建一个ArgumentParser实例
    io_parser = parser.add_mutually_exclusive_group(required=False) # 创建一个互斥组 io_parser,即此组内的参数不能同时存在
    io_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--bmname', dest='bmname',
            help='Name of the benchmark dataset')   # 指定基准数据集的名称,与 bename 参数互斥
    io_parser.add_argument('--pkl', dest='pkl_fname',
            help='Name of the pkl data file')   # 指定pkl格式数据文件的名称,与 dataset 和 bename 参数互斥

    softpool_parser = parser.add_argument_group()   # 创建参数组
    softpool_parser.add_argument('--assign-ratio', dest='assign_ratio', type=float,
            help='ratio of number of nodes in consecutive layers')  # 定义 --assign-ratio 参数，表示连续层中节点数的比例
    softpool_parser.add_argument('--num-pool', dest='num_pool', type=int,
            help='number of pooling layers')    # 定义 --num-pool 参数，表示池化层的数量
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
            const=True, default=False,
            help='Whether link prediction side objective is used')  # 是否使用链路预测作为辅助目标（布尔类型）


    parser.add_argument('--datadir', dest='datadir',
            help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
            help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda',
            help='CUDA.')   # GPU 设备编号
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
            help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
            help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
            help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
            help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
            help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
            help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
            const=False, default=True,
            help='Whether batch normalization is used') # 用于控制是否使用批量归一化
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
            const=False, default=True,
            help='Whether to add bias. Default to True.')   # 控制是否在网络中加入偏置项
    parser.add_argument('--no-log-graph', dest='log_graph', action='store_const',
            const=False, default=True,
            help='Whether disable log graph')   # 控制是否禁用图的日志记录

    parser.add_argument('--method', dest='method',
            help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--name-suffix', dest='name_suffix',
            help='suffix added to the output filename')

    """
    bmname : 基准数据集的名称（与 `dataset` 参数互斥）
    pkl_fname : `pkl` 格式数据文件的名称（与 `dataset` 和 `bmname` 参数互斥）
    linkpred : 是否使用链路预测作为辅助目标
    cuda : 使用的 CUDA 设备编号（例如 `'0'`、`'1'` 表示 GPU 设备编号）
    bn : 是否使用批量归一化（布尔类型）
    bias : 是否在网络中加入偏置项（布尔类型）
    log_graph : 是否禁用图日志记录（布尔类型）
    nobn : 是否禁用批量归一化   如果设置该参数，则 `bn` 为 `False`
    nobias : 是否禁用偏置项 如果设置该参数，则 `bias` 为 `False`
    no-log-graph : 是否禁用日志记录 如果设置该参数，则 `log_graph` 为 `False`
    """

    # 参数的默认值
    """
    datadir : 
    dataset : 数据集名称
    logdir : TensorBoard 日志文件保存的目录路径
    max_nodes : 图中允许的最大节点数量（超过该节点数的图将被忽略）
    feature_type : 使用的特征类型，可选值为 "id" 或 "deg"
    lr : 学习率
    clip : 设置梯度裁剪的阈值
    batch_size : 每个批次的样本数量
    num_epochs : 训练的迭代次数
    train_ratio : 训练集占所有图数据的比例
    test_ratio : 测试集占所有图数据的比例
    num_workers : 加载数据的工作线程数
    input_dim : 输入特征的维度
    hidden_dim : 隐藏层的维度
    output_dim : 输出特征的维度
    num_classes : 标签类别的数量
    num_gc_layers : 每次池化前的图卷积层数
    dropout : Dropout 的比例，用于防止过拟合
    method : 使用的模型方法，例如 "base", "base-set2set", "soft-assign" 等
    name_suffix : 添加到输出文件名的后缀
    assign_ratio : 连续层中节点数的比例（池化层中节点的比例）
    num_pool : 池化层数量
    datadir : 数据存储的目录路径
    """
    parser.set_defaults(datadir='data',
                        logdir='log',
                        dataset='syn1v2',
                        max_nodes=1000,
                        cuda='1',
                        feature_type='default',
                        lr=0.001,
                        clip=2.0,
                        batch_size=20,
                        num_epochs=1000,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=1,
                        input_dim=10,
                        hidden_dim=20,
                        output_dim=20,
                        num_classes=2,
                        num_gc_layers=3,
                        dropout=0.0,
                        method='base',
                        name_suffix='',
                        assign_ratio=0.1,
                        num_pool=1
                       )
    # 返回所有的参数配置值（即 args）
    return parser.parse_args()

def main():
    # 解析命令行参数
    prog_args = arg_parse()

    # export scalar data to JSON for external processing
    path = os.path.join(prog_args.logdir, gen_prefix(prog_args))
    if os.path.isdir(path):
        print('Remove existing log dir: ', path)
        logging.info('Remove existing log dir: %s', path)
        shutil.rmtree(path)
    writer = SummaryWriter(path)
    #writer = None

    os.environ['CUDA_VISIBLE_DEVICES'] = prog_args.cuda
    print('CUDA', prog_args.cuda)
    logging.info('CUDA: %s', prog_args.cuda)

    # 根据数据集不同执行不同的数据处理和训练任务
    if prog_args.bmname is not None:
        benchmark_task_val(prog_args, writer=writer)
    elif prog_args.pkl_fname is not None:
        pkl_task(prog_args)
    elif prog_args.dataset is not None:
        if prog_args.dataset == 'syn1v2':
            syn_community1v2(prog_args, writer=writer)
        if prog_args.dataset == 'syn2hier':
            syn_community2hier(prog_args, writer=writer)

    writer.close()

if __name__ == "__main__":
    main()

