import torch, torch_sparse, sys, time
from torch_geometric.nn import JumpingKnowledge
import torch.nn.functional as F
from sklearn import decomposition
from scipy.sparse import coo_matrix
import numpy as np
from torchkge.utils import Trainer, MarginLoss
from torchkge.models import TransEModel
from torchkge.models.bilinear import ComplExModel, DistMultModel

class GNN(torch.nn.Module):
    def __init__(self, Conv, in_channels, hidden_channels, num_layers,
                 dropout, act=torch.nn.ReLU(), positional=False, num_nodes=None, jk_mode="mean", use_bn=False, label=False, bn_stats=True):
        super(GNN, self).__init__()

        assert jk_mode in ["max","sum","mean","lstm","cat","none"]

        self.positional = positional
        if positional and num_nodes is None: raise Exception("positional requires num_nodes")
        if positional:
            self.pos_embedding = torch.nn.Embedding(num_nodes, hidden_channels)
            in_channels += hidden_channels

        self.label = label
        if label:
            self.label_embedding = torch.nn.Embedding(2,hidden_channels)
            in_channels += hidden_channels
        
        Conv[1][0] = in_channels
        self.convs = torch.nn.ModuleList()
        self.convs.append(Conv[0](*Conv[1]))
        for _ in range(num_layers - 1):
            self.convs.append(
                Conv[0](*Conv[2]))
        self.use_bn = use_bn
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_channels, track_running_stats=bn_stats) for _ in range(num_layers)])
        self.jk_mode = jk_mode
        if jk_mode in ["cat","lstm","max"]: self.jk = JumpingKnowledge(mode=self.jk_mode, channels=hidden_channels, num_layers=num_layers)
        self.jk_mean_weight = torch.nn.Parameter(torch.randn(num_layers))

        self.act = act
        self.dropout = dropout

    def jk_mean(self, x_lst):
        weights = F.softmax(self.jk_mean_weight, dim=0)
        for i in range(len(x_lst)):
            x_lst[i] = x_lst[i] * weights[i]
        return torch.stack(x_lst, dim=-1).sum(dim=-1)

    def reset_parameters(self):
        if self.positional: torch.nn.init.xavier_uniform_(self.pos_embedding.weight)
        if self.label: torch.nn.init.xavier_uniform_(self.label_embedding.weight)
        self.jk_mean_weight = torch.nn.Parameter(torch.randn(self.jk_mean_weight.data.size(0)))
        for conv in self.convs:
            conv.reset_parameters()
        if self.jk_mode in ["cat","lstm","max"]: self.jk.reset_parameters()

    def forward(self, x, adj_t, batch_nodes, undirected=True, edge_weight=None, edge_index=None):
        x_lst = []
        if self.positional: x = torch.cat([x,self.pos_embedding.weight],dim=1)
        if self.label:
            node_label = torch.zeros(x.size(0)).long()
            node_label[batch_nodes] = 1
            x = torch.cat([x,self.label_embedding.weight[node_label]],dim=1)
        for conv, bn in zip(self.convs, self.bns):
            if edge_weight is None:
                x = conv(x, adj_t)
            else:
                x = conv(x=x, edge_index=edge_index, edge_weight=edge_weight)
            if self.use_bn: x = bn(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_lst += [x]
        if self.jk_mode == "none":
            x = x_lst[-1]
        elif self.jk_mode == "mean":
            x = self.jk_mean(x_lst)
        elif self.jk_mode == "sum":
            x = torch.stack(x_lst, dim=-1).sum(dim=-1)
        else:
            x = self.jk(x_lst)
        if undirected: return x
        return x,x


class NMF(torch.nn.Module):
    def __init__(self, edge_index, hidden_channels, num_nodes, device):
        super(NMF, self).__init__()
        self.hidden_channels = hidden_channels
        self.coo_matrix = coo_matrix((np.ones(edge_index.size(1)), (edge_index[1].numpy().astype(int), edge_index[0].numpy().astype(int))), shape=(num_nodes,num_nodes))
        self.m = decomposition.NMF(n_components=hidden_channels, init='random', max_iter=1)
        self.embedding1 = torch.from_numpy(self.m.fit_transform( self.coo_matrix )).float().to(device)
        self.embedding2 = torch.from_numpy( self.m.components_ ).float().to(device).t()
        self.device = device
    def reset_parameters(self):
        self.m = decomposition.NMF(n_components=self.hidden_channels, init='random', max_iter=1)
        self.embedding1 = torch.from_numpy(self.m.fit_transform( self.coo_matrix )).float().to(self.device)
        self.embedding2 = torch.from_numpy( self.m.components_ ).float().to(self.device).t()
    def forward(self, x, adj_t, batch_nodes, undirected=True):
        if undirected: return self.embedding1
        return self.embedding1, self.embedding2

class SVD(torch.nn.Module):
    def __init__(self, adj_t, hidden_channels):
        super(SVD, self).__init__()
        self.adj_t = adj_t
        self.hidden_channels = hidden_channels
        self.embedding1,_, self.embedding2 = torch.svd_lowrank(self.adj_t.to_torch_sparse_coo_tensor(), q=self.hidden_channels)
    def reset_parameters(self):
        self.embedding1,_,self.embedding2 = torch.svd_lowrank(self.adj_t.to_torch_sparse_coo_tensor(), q=self.hidden_channels)
    def forward(self, x, adj_t, batch_nodes, undirected=True, edge_weight=None, edge_index=None):
        if undirected: return self.embedding1
        return self.embedding1, self.embedding2


class KGE(torch.nn.Module):
    def __init__(self, data, hidden_channels, kge_model, device):
        super(KGE, self).__init__()
        self.data = data
        self.device = device
        self.hidden_channels = hidden_channels
        self.kge_model = kge_model
        self.models = {  "TransE" : TransEModel, "ComplEx": ComplExModel, "DistMult": DistMultModel }
        model = self.models[kge_model](hidden_channels, data.n_ent, data.n_rel)
        criterion = MarginLoss(0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
        trainer = Trainer(model, criterion, data, 2000, data.n_ent,
                      optimizer=optimizer, sampling_type='bern', use_cuda='all',)
        trainer.run()
        if self.kge_model == "ComplEx":
            self.embedding1, self.embedding2,_,_ = model.get_embeddings()
        else:
            self.embedding1,_ = model.get_embeddings()
            self.embedding2 = self.embedding1
        self.embedding1 = self.embedding1.detach().to(self.device)
        self.embedding2 = self.embedding2.detach().to(self.device)
    def reset_parameters(self):
        model = self.models[self.kge_model](self.hidden_channels, self.data.n_ent, self.data.n_rel)
        criterion = MarginLoss(0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
        trainer = Trainer(model, criterion, self.data, 2000, self.data.n_ent,
                      optimizer=optimizer, sampling_type='bern', use_cuda='all',)
        trainer.run()
        if self.kge_model == "ComplEx":
            self.embedding1, self.embedding2,_,_ = model.get_embeddings()
        else:
            self.embedding1,_ = model.get_embeddings()
            self.embedding2 = self.embedding1
        self.embedding1 = self.embedding1.detach().to(self.device)
        self.embedding2 = self.embedding2.detach().to(self.device)
    def forward(self, x, adj_t, batch_nodes, undirected=True):
        if undirected: return self.embedding1
        return self.embedding1, self.embedding2

class MCSVD(torch.nn.Module):
    def __init__(self, adj_t, hidden_channels, num_nodes, act=torch.nn.ReLU, nsamples=1):
        super(MCSVD, self).__init__()
        self.adj_t = adj_t
        self.hidden_channels = hidden_channels
        self.num_nodes = num_nodes
        self.nsamples = nsamples
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.act = act
    def reset_parameters(self):
        pass
    def forward(self, x, adj_t, batch_nodes):
        x = 0
        for _ in range(self.nsamples):
            perm = torch.randperm(self.num_nodes)
            adj_t = torch_sparse.permute(adj_t, perm)
            embedding,_,_ = torch.svd_lowrank(adj_t.to_torch_sparse_coo_tensor(), q=self.hidden_channels, niter=1)
            inv_perm = [None]*self.num_nodes
            for i,j in enumerate(perm):
                inv_perm[j.item()] = i
            embedding = embedding[inv_perm]
            embedding = self.act(self.lin1(embedding))
            x += embedding
        x = x/self.nsamples
        x = self.act(self.lin2(x))
        return x