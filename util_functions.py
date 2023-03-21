from __future__ import print_function
import numpy as np
import random
import os, sys, pdb, math, time
import networkx as nx
import scipy.sparse as ssp
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import warnings
from graph_process import divide_son_by_cluster, construct_pyq_graph,processing_songraph
from collections import defaultdict

warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
import torch.multiprocessing

from matrix_indexer import SparseRowIndexer, SparseColIndexer
torch.multiprocessing.set_sharing_strategy('file_system')

class MyDynamicDataset(Dataset):
    # root = "data/ml_1m_mnph100/testmode/train"
    def __init__(self, root, A, T, links, labels, h, sample_ratio, subgraph_num, eps, cluster_samples,max_nodes_per_hop,
                 u_features, v_features, class_values, max_num=None):
        super(MyDynamicDataset, self).__init__(root)
        self.T = T
        self.Arow = SparseRowIndexer(A)  # A.shape(6040,3706),(num_users,num_items),存储每个用户对电影的评分:1-5
        self.Acol = SparseColIndexer(A.tocsc())
        self.links = links  # links = train_indices=(train_u_indices,train_v_indices)
        self.labels = labels
        self.h = h
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
        self.u_features = u_features
        self.v_features = v_features
        self.class_values = class_values
        self.subgraph_num = subgraph_num
        self.eps = eps
        self.cluser_samples = cluster_samples
        if max_num is not None:
            np.random.seed(123)
            num_links = len(links[0])
            perm = np.random.permutation(num_links)  # 将num_links随机排列
            perm = perm[:max_num]
            self.links = (links[0][perm], links[1][perm])
            self.labels = labels[perm]

    def len(self):
        return self.__len__()

    def __len__(self):
        return len(self.links[0])

    def get(self, idx):
        i, j = self.links[0][idx], self.links[1][idx]
        g_label = self.labels[idx]
        tmp = subgraph_extraction_labeling(
            (i, j), self.Arow, self.Acol, self.T, self.h, self.sample_ratio, self.max_nodes_per_hop,
            self.u_features, self.v_features, self.class_values,self.subgraph_num,self.eps,self.cluser_samples, g_label
        )
        results = construct_pyq_graph(*tmp)
        return results

def build_temporal_order(maxlen, sequence):
    order = []
    index_bool = []
    last_node = sequence[-1]
    k = len(sequence)
    if k >= maxlen:
        order.extend(sequence[k-maxlen:])  # recent maxlen items
        index_bool.extend([1]*maxlen)
    else:
        order.extend(sequence[:k] + [last_node] * (maxlen - k))  # padding unexist item id
        index_bool.extend([1]*k + [0]*(maxlen - k))
    return order, index_bool

def subgraph_extraction_labeling(ind, Arow, Acol, T, h=1, sample_ratio=1.0, max_nodes_per_hop=None,
                                 u_features=None, v_features=None, class_values=None, subgraph_num=5,eps =0.9, cluster_samples=20,
                                 y=1):
    # extract the h-hop enclosing subgraph around link 'ind',ind=(uid,vid)
    u_nodes, v_nodes = [ind[0]], [ind[1]]  # v_nodes: [318], u_nodes: [162]
    u_dist, v_dist = [0], [0]
    u_visited, v_visited = set([ind[0]]), set([ind[1]])
    u_fringe, v_fringe = set([ind[0]]), set([ind[1]])
    for dist in range(1, h + 1):
        # v_fringe = set(A[list(u_fringe)].indices)
        # v_fringe存储的是u_fringe所在行非零列的列索引,u_fringe存储的是v_fringe所在列非零行的行索引
        # v_fringe存储的是(u,v)中u的交互item的索引，u_fringe存储的是(u,v)中v的交互user的索引
        v_fringe, u_fringe = neighbors(u_fringe, Arow), neighbors(v_fringe, Acol)
        u_fringe = u_fringe - u_visited
        v_fringe = v_fringe - v_visited
        u_visited = u_visited.union(u_fringe)
        v_visited = v_visited.union(v_fringe)
        # 原论文中的采样方法
        if sample_ratio < 1.0:
            u_fringe = random.sample(u_fringe, int(sample_ratio * len(u_fringe)))
            v_fringe = random.sample(v_fringe, int(sample_ratio * len(v_fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(u_fringe):
                u_fringe = random.sample(u_fringe, max_nodes_per_hop)
            if max_nodes_per_hop < len(v_fringe):
                v_fringe = random.sample(v_fringe, max_nodes_per_hop)
        if len(u_fringe) == 0 and len(v_fringe) == 0:
            break
        u_nodes = u_nodes + list(u_fringe)
        v_nodes = v_nodes + list(v_fringe)
        u_dist = u_dist+[dist] * len(u_fringe)
        v_dist = v_dist+[dist] * len(v_fringe)

    # 获取大图的(u,v)的时序邻居序列
    u_numbers = Acol.shape[0]
    user_items_times = defaultdict(list)
    for o_iid in v_nodes:
        user_id = ind[0]
        item_id = o_iid
        time = T[user_id,item_id]
        item_id_map = item_id+u_numbers
        user_items_times[user_id].append((item_id_map, time))
    item_users_times = defaultdict(list)
    for o_uid in u_nodes:
        item_id = ind[1]
        user_id = o_uid
        time = T[user_id,item_id]
        item_id_map = item_id+u_numbers
        item_users_times[item_id_map].append((user_id, time))
    user_items_dict = defaultdict(list)
    item_users_dict = defaultdict(list)
    for key in user_items_times:
        sorted_user_items_times = sorted(user_items_times[key], key=lambda x: x[1])
        for item, timestamp in sorted_user_items_times:
            user_items_dict[key].append(item)
    for key in item_users_times:
        sorted_item_users_times = sorted(item_users_times[key], key=lambda x: x[1])
        for user, timestamp in sorted_item_users_times:
            item_users_dict[key].append(user)
    user_items = sorted(user_items_dict.items(), key=lambda d: d[0])
    item_users = sorted(item_users_dict.items(), key=lambda d: d[0])
    temporal_orders = defaultdict(list)
    node_index = defaultdict(list)
    for user_id, item_list in user_items:
        order, index = build_temporal_order(maxlen=200, sequence=item_list)
        temporal_orders[user_id].extend(order)
        node_index[user_id].extend(index)
    for item_id, user_list in item_users:
        order, index = build_temporal_order(maxlen=200, sequence=user_list)
        temporal_orders[item_id].extend(order)
        node_index[item_id].extend(index)
    temporal_table = [temporal_orders[key] for key in sorted(temporal_orders.keys())]
    node_index = [node_index[key] for key in sorted(node_index.keys())]
    temporal_table = torch.tensor(temporal_table)  # (2,200)
    node_index = torch.tensor(node_index)  # (2,200)

    # 获得待预测用户所交互的item,按照时间分三部分
    son_list, num_son, son_timestamp, son_v_dist_list = divide_son_by_cluster(T, ind, v_fringe, v_dist, eps, cluster_samples, subgraph_num)
    subgraph_message_list = []
    if num_son > 0:
        for i in range(num_son):
            son_v_nodes = son_list[i]
            son_graph = Arow[u_nodes][:, son_v_nodes]
            son_u_dst = u_dist
            son_v_dist = son_v_dist_list[i]
            # 处理子图
            son_graph_message = processing_songraph(son_graph, son_u_dst, son_v_dist, u_nodes, son_v_nodes)
            subgraph_message_list.append(son_graph_message)
    else:
        subgraph = Arow[u_nodes][:, v_nodes]
        graph_message = processing_songraph(subgraph, u_dist, v_dist, u_nodes, v_nodes)
        subgraph_message_list.append(graph_message)
    max_node_label = 2 * h + 1
    y = class_values[y]
    centre_index = torch.tensor([ind[0], ind[1]+u_numbers])
    return subgraph_message_list, max_node_label, y, temporal_table, node_index, centre_index
    # return u, v, r, node_labels, max_node_label, y, node_features

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

def PyGGraph_to_nx(data):
    edges = list(zip(data.edge_index[0, :].tolist(), data.edge_index[1, :].tolist()))
    g = nx.from_edgelist(edges)
    g.add_nodes_from(range(len(data.x)))  # in case some nodes are isolated
    # transform r back to rating label
    edge_types = {(u, v): data.edge_type[i].item() for i, (u, v) in enumerate(edges)}
    nx.set_edge_attributes(g, name='type', values=edge_types)
    node_types = dict(zip(range(data.num_nodes), torch.argmax(data.x, 1).tolist()))
    nx.set_node_attributes(g, name='type', values=node_types)
    g.graph['rating'] = data.y.item()
    return g
