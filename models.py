import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch_geometric.nn import GCNConv, RGCNConv, global_sort_pool, global_add_pool
from torch_geometric.utils import dropout_adj
from util_functions import *
import time
import torch

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class GNN(torch.nn.Module):
    # a base GNN class, GCN message passing + sum_pooling
    def __init__(self, dataset, gconv=GCNConv, latent_dim=[32, 32, 32, 1],
                 regression=False, adj_dropout=0.2, force_undirected=False):
        super(GNN, self).__init__()
        self.regression = regression
        self.adj_dropout = adj_dropout
        self.force_undirected = force_undirected
        self.convs = torch.nn.ModuleList()
        # dataset.num_features = 4, latent_dim = [32,32,32,32]
        self.convs.append(gconv(dataset.num_features, latent_dim[0]))
        for i in range(0, len(latent_dim) - 1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i + 1]))
        self.lin1 = Linear(sum(latent_dim), 128)
        if self.regression:
            self.lin2 = Linear(128, 1)
        else:
            self.lin2 = Linear(128, dataset.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout,
                force_undirected=self.force_undirected, num_nodes=len(x),
                training=self.training
            )
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)
        x = global_add_pool(concat_states, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0]
        else:
            return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class STAM(torch.nn.Module):
    def __init__(self, input_dim, n_heads, input_length, hidden_dim, batch_size,**kwargs):
        super(STAM, self).__init__()
        self.n_heads = n_heads
        self.input_batch = batch_size
        self.input_dim = input_dim  # 64
        self.input_length = input_length  # 200
        self.hidden_dim = hidden_dim  # 64
        self.attn_drop_rate = 0.5
        self.attention_head_size = int(hidden_dim / n_heads)
        self.attn_softmax = nn.Softmax(dim=3)
        self.attn_dropout = nn.Dropout(p=self.attn_drop_rate)
        self.attn_con1v = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=1, bias=True)
        self.position_embedding = nn.Embedding(num_embeddings=self.input_length, embedding_dim=self.input_dim)
        self.Q_embedding = nn.init.normal_(torch.Tensor(self.input_dim, self.hidden_dim), std=0.01).to(device)
        self.K_embedding = nn.init.normal_(torch.Tensor(self.input_dim, self.hidden_dim), std=0.01).to(device)
        self.V_embedding = nn.init.normal_(torch.Tensor(self.input_dim, self.hidden_dim), std=0.01).to(device)
        self.position_inputs = torch.repeat_interleave(torch.unsqueeze(torch.arange(0, 200), 0), 2*self.input_batch, 0).to(device)  # (2*batch,200)
        self.input_mask = torch.ones((2*self.input_batch, self.input_length), dtype=torch.int32).to(device)  # (2*batch,200)
        self.broadcast_ones = torch.ones(size=[2*self.input_batch, self.input_length, 1]).float().to(device)    # (2*batch,200,1)
    def __call__(self, inputs):  # (2*batch_size,200,64)

        position_embedding = self.position_embedding(self.position_inputs)  # (2*batch,200,64)
        temporal_inputs = inputs + position_embedding  # (2*batch,200,64)
        q = torch.matmul(temporal_inputs, self.Q_embedding)  # (2*batch,200,64)
        k = torch.matmul(temporal_inputs, self.K_embedding)  # (2*batchi,200,64)
        v = torch.matmul(temporal_inputs, self.V_embedding)  # (2*batchi,200,64)

        batch_size = inputs.size()[0]
        q = torch.reshape(q, (batch_size, self.input_length, self.n_heads, self.attention_head_size)).transpose(1,
                                                                                                                2)  # (2*batch,4,200,16)
        k = torch.reshape(k, (batch_size, self.input_length, self.n_heads, self.attention_head_size)).transpose(1,
                                                                                                                2)  # (2*batch,4,200,16)
        v = torch.reshape(v, (batch_size, self.input_length, self.n_heads, self.attention_head_size)).transpose(1,
                                                                                                                2)  # (2*batch,4,200,16)

        # scaled dot-product attention
        outputs = torch.matmul(q, torch.transpose(k, 2, 3))  # (2*batch,4,200,200)
        outputs = outputs / (self.input_length ** 0.5)  # (2*batch,4,200,200)

        to_mask = torch.reshape(self.input_mask, (batch_size, 1, self.input_length)).float()  # (2*batch,1,200)
        mask = self.broadcast_ones * to_mask  # (2*batch,200,200)
        attention_mask = torch.unsqueeze(mask, 1).float()  # (2*batch,1,200,200)
        adder = (1.0 - attention_mask) * (-10000.0)  # (2*batch,1,200,200)
        outputs += adder

        att_weights = self.attn_softmax(outputs)  # (2*batch,4,200,200)
        att_weights = self.attn_dropout(att_weights)  # (2*batch,4,200,200)

        h = torch.matmul(att_weights, v)  # (2,4,200,16)
        h = torch.transpose(h, 1, 2)  # (2,200,4,16)
        h = torch.reshape(h, [batch_size, self.input_length, self.input_dim])  # (2*batch,200,64)
        Tem_emb = self.feedforward(h)  # (2*batch,200,64)
        Tem_emb = temporal_inputs + Tem_emb
        Tem_emb = F.normalize(Tem_emb, p=2.0, dim=2)  # (2,200,64)
        return Tem_emb

    def feedforward(self, inputs):  # (2*batch,200,64)
        input = torch.reshape(inputs, [-1, self.input_length, self.input_dim])  # (2*batch,200,64)
        outputs = self.attn_con1v(input)  # (2*batch,200,64)
        outputs += inputs
        return outputs


class IGMC(GNN):
    # The GNN model of Inductive Graph-based Matrix Completion. 
    # Use RGCN convolution + center-nodes readout.
    def __init__(self, dataset, num_u, num_v,gconv=RGCNConv, latent_dim=[32, 32, 32, 32],
                 num_relations=5, num_bases=2, regression=False, adj_dropout=0.2,
                 force_undirected=False, side_features=False, n_side_features=0,
                 multiply_by=1):
        super(IGMC, self).__init__(
            dataset, GCNConv, latent_dim, regression, adj_dropout, force_undirected
        )
        self.batch_size = 50
        self.multiply_by = multiply_by
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0], num_relations, num_bases))
        for i in range(0, len(latent_dim) - 1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i + 1], num_relations, num_bases))
        temp_emb_dim = 5*2*64
        meta_emb_dim = 5*2*4*32
        self.lin1 = Linear(temp_emb_dim+meta_emb_dim, 128)
        self.lin2 = Linear(128,1)
        # self.lin_meta = Linear(5 * 8 * latent_dim[0], 1)
        self.side_features = side_features
        self.latent_dim = latent_dim[0]
        # if side_features:
        #     self.lin1 = Linear(2 * sum(latent_dim) + n_side_features, 128)
        self.emb_dim = 4 * latent_dim[0] * 2
        self.meta_dim = self.emb_dim * 2
        self.att_net = torch.nn.Sequential(torch.nn.Linear(self.emb_dim, self.emb_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(self.emb_dim, 1, False))
        self.att_softmax = torch.nn.Softmax(dim=1)
        self.meta_net = torch.nn.Sequential(torch.nn.Linear(self.emb_dim, self.meta_dim), torch.nn.ReLU(),
                                            torch.nn.Linear(self.meta_dim, 5 * 4 * 2 * latent_dim[0]))
        self.num_users = num_u  # 6040
        self.num_items = num_v  # 3706
        self.temp_num_layers = 4
        self.temp_softmax = nn.Softmax(dim=1)
        # (9746,64)
        self.u_v_init_embedding = nn.Embedding(num_embeddings=self.num_users + self.num_items, embedding_dim=64)
        self.init_emb_indices = torch.arange(self.num_users + self.num_items).to(device)
        self.stam_layer = STAM(input_dim=64, n_heads=4, input_length=200, hidden_dim=64,batch_size=self.batch_size).to(device=device)
        self.tem_table_indices = torch.reshape(torch.repeat_interleave(torch.arange(0, 2*self.batch_size), 200), (-1, 1)).to(device)

    def forward(self, data_list):
        start = time.time()
        x_list = []
        graph_tem_emb_batch = []
        for data in data_list:
            '''
            x=[26901,4] :图的节点特征,维度为四的one_hot向量，代表着user,item,user的一跳邻居，item的一跳邻居
            edge_index:边的信息：u,v对儿
            edge_type: 边的类型：评分
            temporal_tabel list:30,30个(2,200)
            node_index :list,30个(2,200)
            '''
            x, edge_index, edge_type, centre_index, temporal_table, temporal_label, batch \
                = data.x, data.edge_index, data.edge_type, data.pos, data.temporal_table, data.temporal_label, data.batch
            if self.adj_dropout > 0:
                edge_index, edge_type = dropout_adj(
                    edge_index, edge_type, p=self.adj_dropout,
                    force_undirected=self.force_undirected, num_nodes=len(x),
                    training=self.training
            )
            concat_states = []
            for conv in self.convs:  # 根据conv输出维度为n(32)
                x = torch.tanh(conv(x, edge_index, edge_type))
                concat_states.append(x)
            # (28595,128)
            concat_states = torch.cat(concat_states, 1)  # 4层conv，最后输出4*n,即：(4*32),这里的n是隐藏层的维度

            users = data.x[:, 0] == 1
            items = data.x[:, 1] == 1
            x = torch.cat([concat_states[users], concat_states[items]], 1)  # (batch_size,4*n*2):(30,256)
            # if self.side_features:
            #     x = torch.cat([x, data.u_feature, data.v_feature], 1)
            x_list.append(x)  # list :5个(30,256)

            # 获取时空表征
            tem_inputs = self.u_v_init_embedding(temporal_table)  # (2*batch,200,64),tensor,float32
            tem_outs = self.stam_layer(tem_inputs)  # (2*batch,200,64)
            # min = torch.min(centre_index)
            # max = torch.max(centre_index)
            W_1 = torch.unsqueeze(self.u_v_init_embedding(centre_index), 1)  # (2*batch,1,64), centre_index:(2*batchsize,)
            tem_weight = self.temp_softmax(torch.squeeze(torch.matmul(tem_outs, W_1.permute(0, 2, 1))))  # (2*batch,200)
            adj_mask = torch.mul(tem_weight, temporal_label)  # (2*batch,200)
            adj_sum = torch.div(adj_mask, torch.sum(adj_mask, dim=1, keepdim=True))  # (2*batch,200)
            adj_sum = torch.where(torch.isinf(adj_sum), torch.full_like(adj_sum, 0), adj_sum)
            adj_sum = torch.where(torch.isnan(adj_sum), torch.full_like(adj_sum, 0), adj_sum)   # (2*batch,200)
            adj = torch.masked_select(adj_sum, temporal_label.bool())  # (？,)
            tem_table_seq = torch.reshape(temporal_table, (-1, 1))      # (2*batch*200,1)
            indices = torch.squeeze(torch.stack((self.tem_table_indices, tem_table_seq), 1))  # (2*batch*200,2)
            adj_indices = torch.reshape(
                torch.masked_select(indices, torch.repeat_interleave(temporal_label.reshape(-1, 1), 2, 1).bool()), (-1, 2)).T # (2*batch,?)
            # (2*batchsize,9746)
            stam_weights = torch.sparse_coo_tensor(indices=adj_indices, values=adj,
                                                   size=[2*self.batch_size, self.num_users + self.num_items])  # (2*batchsize,9746)

            ego_emb = self.u_v_init_embedding(self.init_emb_indices)  # (9746,64)
            _emb = torch.sparse.mm(stam_weights, ego_emb)  # (2*batch,64)
            temp_emb_batch = torch.reshape(_emb, (self.batch_size, -1))  # (batch,128)
            graph_tem_emb_batch.append(temp_emb_batch)  # list ,5个(30,128)

        graph_tem_emb = torch.cat(graph_tem_emb_batch, 1)  # (30,640),(30,128*5=640)

        # 聚合子图法一：利用元学习学习子图之间的关系
        # 获取batchsize个用户每个子图的embedding
        graph_emb = torch.cat(x_list, 1)  # (batch_size,1,5*8*n) (30,1280)
        subgraph_fea = torch.stack(x_list, dim=1)  # (batchsize, 5, 4*n*2)
        att_K = self.att_net(subgraph_fea)  # (batchsize,5,1)
        att = self.att_softmax(att_K)  # (batchsize,5,1)
        graph_fea = torch.sum(att * subgraph_fea, 1)  # (batch_size, 4*n*2)
        # 输入到meta_net
        output = self.meta_net(graph_fea).squeeze(1)  # (batch_size, 5*8*n)
        x = graph_emb * output  # (batch_size,5*8*n) (30,1280)

        x = torch.cat((graph_tem_emb, x), 1)  # (30, 1920)
        x = F.relu(self.lin1(x))  # (batch_size,1)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        # 法2：将所有图的embedding进行concat
        # x = torch.cat(x_list, 1)   # x:(batch_size,len(list)*8*n)
        # x = F.relu(self.lin1(x))    # lin1:(len(list)*8*n,128)，x :(batch_size,128)
        # x = F.dropout(x, p=0.5, training=self.training)   # x:(batch_size,128)
        # x = self.lin2(x)  # (batch_size,1)

        if self.regression:
            return x[:, 0] * self.multiply_by  # (batch_size,)
        else:
            return F.log_softmax(x, dim=-1)
