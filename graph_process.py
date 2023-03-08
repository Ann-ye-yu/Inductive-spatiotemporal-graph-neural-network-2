import operator
import random
from collections import defaultdict
import torch
# import dglgo as dgl
from torch_geometric.data import Data
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse as ssp
from matrix_indexer import SparseRowIndexer,SparseColIndexer


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    if not fringe:
        return set([])
    return set(A[list(fringe)].indices)


def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x


# 参数的选择
# 计算每个点到其k近个点的距离，之后将距离从小到大排序
def select_MinPts(data, k):
    k_dist = []
    for i in range(data.shape[0]):
        # dist1 = (data[i] - data) ** 2
        # dist2 = dist1.sum(axis=1) ** 0.5
        dist = (((data[i] - data) ** 2).sum(axis=1) ** 0.5)
        dist.sort()
        k_dist.append(dist[k])
    return np.array(k_dist)


def divide_son_by_cluster(T, ind, v_fringe, v_dist, max_son_length):
    u_centre = int(ind[0])
    v_centre = int(ind[1])
    # 这里的时间戳统一乘以1e-9转换为0-1之间的数字
    v_timestamp_map = [[x*1e-2, T[u_centre, x]*1e-9] for x in v_fringe]
    v_timestamp_map = np.array(v_timestamp_map)
    # kmeans聚类法
    # kmeans = KMeans(n_clusters=3)  # 创建一个K-均值聚类对象
    # kmeans.fit(v_timestamp)
    # v_label = kmeans.predict(v_timestamp)  # 获取聚类分配

    # DBSCAN算法聚类
    # 获取拐点获取eps参数,10是可调参数
    # k_dist = select_MinPts(v_timestamp_map, 3)
    # k_dist.sort()
    # eps = k_dist[::-1][15]  # 倒数15个
    # plt.plot(np.arange(k_dist.shape[0]), k_dist[::-1])
    # plt.savefig('results/参数选择拐点图.jpg')
    # plt.scatter(15,eps,color="r")
    # plt.plot([0,15],[eps,eps],linestyle="--",color = "r")
    # plt.plot([15,15],[0,eps],linestyle="--",color = "r")
    # plt.savefig("results/拐点.jpg")
    pred_label = DBSCAN(eps=0.9, min_samples=20).fit_predict(v_timestamp_map)
    v_class = list(set(pred_label))
    # 画出聚类结果图
    # 这里将噪声点分开处理了
    if -1 in v_class:
        v_class.remove(-1)
    #     v_class_list = [[] for j in v_class]
    # for index, value in enumerate(pred_label):
    #     for i in v_class:
    #         if value == i:
    #             v_class_list[i].append(index)
    #             break
    #         elif value == -1:
    #             noise_list.append(index)
    #             break
    # color_list = ['g', 'b', 'y', 'm', 'k', 'c']
    # for i in range(len(v_class_list)):
    #     plt.scatter(v_timestamp[v_class_list[i], 0], v_timestamp[v_class_list[i], 1], color=color_list[i],
    #                 label="class" + str(i))
    # plt.scatter(v_timestamp[noise_list, 0], v_timestamp[noise_list, 1], color='r', label="noise")
    # plt.legend()
    # plt.savefig(f"cluster result.jpg")
    # plt.clf()
    son_list = [[] for i in v_class]
    son_timstamp_map = [[] for j in v_class]
    son_v_dist = [[] for k in v_class]
    for i in v_class:
        son_list[i].append(v_centre)
        son_timstamp_map[i].append(T[u_centre, v_centre])
        son_v_dist[i].append(v_dist[0])
    v_dist = v_dist[1:]
    for i in range(len(pred_label)):
        if pred_label[i] != -1:
            label = pred_label[i]
            for k in v_class:
                if label == k:
                    son_list[k].append(int(v_timestamp_map[i][0]*1e2))
                    son_timstamp_map[k].append(v_timestamp_map[i][1]*1e9)
                    son_v_dist[k].append(v_dist[i])
                    break
    # 计算每个子图与中心item之间的时间距离，时间相隔越近，影响越大。
    centre_timestamp = T[u_centre, v_centre]
    son_time_dis = []
    for index, timestamp_list in enumerate(son_timstamp_map):
        time_dis_li = []
        for stamp in timestamp_list:
            time_dis_li.append(abs(stamp-centre_timestamp))
        son_time_dis.append({'time_dis': sum(time_dis_li), 'son_index': index})
    # 按照时间距离进行升序排列
    son_dis_dict = sorted(son_time_dis, key=operator.itemgetter('time_dis'))

    son_list_sorted = []
    son_timestamp = []
    son_v_dist_sorted = []
    for i in son_dis_dict:
        index_of_son = i['son_index']
        son_list_sorted.append(son_list[index_of_son])
        son_timestamp.append(son_timstamp_map[index_of_son])
        son_v_dist_sorted.append(son_v_dist[index_of_son])
    if len(son_list_sorted) > max_son_length:
        son_list_sorted = son_list_sorted[:max_son_length]
        son_timestamp = son_timestamp[:max_son_length]
        son_v_dist_sorted= son_v_dist_sorted[:max_son_length]
    return son_list_sorted, len(son_list_sorted), son_timestamp, son_v_dist_sorted

def processing_songraph(son_graph,u_dist,v_dist,son_u_nodes,son_v_nodes):
    subgraph = son_graph
    u_nodes = son_u_nodes
    v_nodes = son_v_nodes
    # remove link between target nodes
    subgraph[0, 0] = 0
    # prepare pyg graph constructor input. 返回矩阵非零元素的行索引，列索引，以及对应的值
    u, v, r = ssp.find(subgraph)
    v += len(u_nodes)
    r = r - 1
    num_nodes = len(u_nodes) + len(v_nodes)
    '''
    首先将标签0和1分别赋予目标用户和目标项目
    u_dist = [0,1,...1],长度为user个数101，v_dist = [0,1,1,1...]，长度为item的个数101，这样就可以使得目标用户和目标项目标签为0，1
    对于封闭子图中的其他节点，我们根据其在子图中的哪一跳来确定它们的标签，
    如果用户类型的节点包含在第i跳，我们将给它一个标签2i。
    如果项目类型的节点包括在第i跳，我们将给它2i+1
    '''
    # 将中心user标记为0，中心item标记为1，一跳user标记为2，一跳item标记为3
    node_labels = [x * 2 for x in u_dist] + [x * 2 + 1 for x in v_dist]
    _dict = {
        'u': u,
        'v': v,
        'r': r,
        'node_labels': node_labels,
    }
    return _dict

def construct_pyq_graph(graph_message_list, max_node_label, y, temporal_table,temporal_label,centre_index):
    data_list = []
    for i in range(len(graph_message_list)):
        u, v, r = graph_message_list[i]['u'], graph_message_list[i]['v'], graph_message_list[i]['r']
        u, v, r = torch.LongTensor(u), torch.LongTensor(v), torch.LongTensor(r)
        node_labels = graph_message_list[i]['node_labels']
        edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], 0)  # torch.stack([],0)，按行堆叠
        edge_type = torch.cat([r, r])
        x = torch.FloatTensor(one_hot(node_labels, max_node_label + 1))
        y = torch.FloatTensor([y])
        data = Data(x, edge_index, edge_type=edge_type, y=y, pos=centre_index
                    , temporal_table=temporal_table,temporal_label=temporal_label)
        data_list.append(data)
    # 为了方便进行子图上的batch_size,进行子图扩充,当数量不足5，在列表中随机选一个子图进行扩充
    len_subgraph = len(data_list)
    if len_subgraph < 5:
        # pad_graph = random.choice(data_list)
        pad_graph = data_list[0]
        for i in range(5-len_subgraph):
            data_list.append(pad_graph)
    return tuple(data_list)
