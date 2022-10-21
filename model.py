import torch.nn as nn

from layers.GNN import GraphAttention
from utils import *

class MLP(nn.Module):
    def __init__(self, units: list):
        super(MLP, self).__init__()
        self.units = units # contain the input_dim
        self.hidden_numbers = len(self.units) - 1

        layers = []
        for i in range(self.hidden_numbers):
            layers.extend([nn.Linear(self.units[i], self.units[i + 1]), nn.BatchNorm1d(units[i + 1]), nn.Tanh()])
        self.backbone_net = nn.ModuleList(layers)
        self.backbone_net = nn.Sequential(*self.backbone_net)

    def forward(self, x):
        z = self.backbone_net(x)
        return z


class GCNEncoder(nn.Module):
    def __init__(self, input_dim, generate_dim, edge_dim, dropout=0.5, beta=1) -> None:
        super(GCNEncoder, self).__init__()
        self.input_dim = input_dim
        self.generate_dim = generate_dim
        self.edge_dim = edge_dim
        self.beta = beta

        self.gcn = GraphAttention(self.input_dim, self.generate_dim, self.edge_dim, dropout)

    def forward(self, param_dict: dict):
        recons = self.gcn(**param_dict) # unpack the param-pairs dict
        return recons * self.beta 


class KNNGenerator(nn.Module):
    def __init__(self, K=10, dis_type='L2'):
        super(KNNGenerator, self).__init__()
        self.K = K
        self.dis_type = dis_type

    def distance(self, mat1, mat2, type='cosine'):
        if type == 'cosine':
            mat1_norm = F.normalize(mat1)
            mat2_norm = F.normalize(mat2)
            sim = mat1_norm.mm(mat2_norm)
            dis = -sim
        elif type == 'L2':
            dis = torch.cdist(mat1, mat2, p=2)
        else:
            dis = None
        return dis

    def forward(self, feat, anchor, target_anchor):
        dis = self.distance(feat, anchor, type=self.dis_type)
        index = torch.argsort(dis, dim=1, descending=False)
        index_topk = index[:, :self.K]
        recons = torch.mean(target_anchor[index_topk], dim=1)
        return recons


class FFNGenerator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FFNGenerator, self).__init__()
        self.mlp = MLP(units=[input_dim, 2048, output_dim])

    def forward(self, feat):
        recons = self.mlp(feat)
        return recons


class Fusion(nn.Module):
    def __init__(self, fusion_dim=1024, nbit=64) -> None:
        super(Fusion, self).__init__()
        self.hash = nn.Sequential(
            nn.Linear(fusion_dim, nbit),
            nn.BatchNorm1d(nbit),
            nn.Tanh(),)

    def forward(self, x, y):
        hash_code = self.hash(x + y)
        return hash_code


class SelfAttention(nn.Module):
    def __init__(self, Q_dim, K_dim, V_dim, d_model) -> None:
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.Q_layer = nn.Linear(Q_dim, d_model)
        self.K_layer = nn.Linear(K_dim, d_model)
        self.V_layer = nn.Linear(V_dim, d_model)

    def forward(self, Q, K, V, label_graph):
        Q = self.Q_layer(Q)
        # K = self.K_layer(K)
        K = self.Q_layer(K)
        V = self.V_layer(V)
        attention_score = (Q @ K.T) / math.sqrt(self.d_model)
        if label_graph != None:
            attention_score = label_graph.mul(attention_score)
        attention_prob = F.softmax(attention_score, dim=-1)
        context = attention_prob @ V

        return context, attention_prob


class TransformerEncoder(nn.Module):
    def __init__(self, Q_dim, K_dim, V_dim, d_model=1024, dim_feedforward=2048, dropout=0.5) -> None:
        super(TransformerEncoder, self).__init__()

        self.self_attn = SelfAttention(Q_dim, K_dim, V_dim, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, V_dim),
            nn.BatchNorm1d(V_dim),
            nn.Tanh(),)

        self.activation = nn.Tanh()

    def forward(self, src, anchor_1, anchor_2, label_graph):
        src2, _ = self.self_attn(src, anchor_1, anchor_2, label_graph)
        src = src2 # without short-cut
        
        src = self.norm1(src) # d_model
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src)))) # d_model
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # d_model -> V_dim
        src = self.decoder(src)
        return src

