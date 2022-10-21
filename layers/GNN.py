import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    '''
        GCN layer.
    '''
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        output = torch.mm(input, self.weight)
        output = torch.mm(adj, output)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphAttention(nn.Module):
    '''
        GAT layer.
    '''
    def __init__(self, in_features, out_features, edge_features, dropout=0.5, alpha=0.1, concat=True):
        super(GraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_features = edge_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        nn.init.xavier_normal_(self.weight, gain=1.414)
        self.att = Parameter(torch.FloatTensor(2 * self.edge_features, 1))
        nn.init.xavier_normal_(self.att, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, feat_edge, feat_edge_a, feat_node_a, adj=None, sup=False):
        Wh = torch.mm(feat_node_a, self.weight)
        e = self._prepare_attentional_mechanism_input(feat_edge, feat_edge_a)
        if sup:
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
        else:
            attention = e
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

        return

    def _prepare_attentional_mechanism_input(self, feat_edge, feat_edge_a):
        '''
        self.att.shape (2 * edge_features, 1)
        Wh1/Wh2_a.shape (N, 1)
        e.shape (N, N_a)
        :param Wh: shape (N, out_feature)
        :param Wh_a: shape (N_a, out_feature)
        :return:
        '''
        Afeat_edge = torch.matmul(feat_edge, self.att[:self.edge_features, :])
        Afeat_edge_a = torch.matmul(feat_edge_a, self.att[self.edge_features:, :])
        # broadcast add
        e = Afeat_edge + Afeat_edge_a.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
