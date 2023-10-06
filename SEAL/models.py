import math

import torch
import torch.nn.functional as F
from torch.nn import Conv1d, MaxPool1d, ModuleList, Linear, Embedding
from torch_geometric.nn import GCNConv, global_sort_pool

class SimpleModel(torch.nn.Module):
    def __init__(self, node_embedding, predictor):
        super(SimpleModel, self).__init__()
        self.node_embedding = node_embedding
        self.predictor = predictor

    def forward(self, data):
        h = self.node_embedding(data.x, data.adj_t)
        out = self.predictor(h[data.edge[0]], h[data.edge[1]])
        return out

    def reset_parameters(self):
        self.node_embedding.reset_parameters()
        self.predictor.reset_parameters()


# An end-to-end deep learning architecture for graph classification, AAAI-18.
class DGCNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, max_z=1000, k=0.6, train_dataset=None,
                 dynamic_train=False, GNN=GCNConv, use_feature=False,
                 node_embedding=None, edge_channels=None):
        super(DGCNN, self).__init__()

        self.use_feature = use_feature
        self.node_embedding = node_embedding

        if edge_channels is not None:
            self.edge = torch.nn.Embedding(edge_channels, hidden_channels)
        else:
            self.edge = None
        
        if k <= 1:  # Transform percentile to number.
            if train_dataset is None:
                k = 30
            else:
                if dynamic_train:
                    sampled_train = train_dataset[:1000]
                else:
                    sampled_train = train_dataset
                num_nodes = sorted([g.num_nodes for g in sampled_train])
                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)
        self.k = int(k)

        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += in_channels
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim

        self.convs.append(GNN(initial_channels, hidden_channels))
        for i in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        if edge_channels is None:
            self.lin1 = Linear(dense_dim, 128)
        else:
            self.lin1 = Linear(dense_dim+hidden_channels, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, data, edge=None):
        z, edge_index, batch, edge_weight = data.z, data.edge_index, data.batch, data.edge_weight
        x = data.x if self.use_feature else None
        node_id = None

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        xs = [x]

        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index, edge_weight))]
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        if edge is not None: x = torch.cat([x,self.edge(edge)],1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

    def reset_parameters(self):
        self.z_embedding.reset_parameters()
        if self.node_embedding is not None:
            self.node_embedding.reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()
        if self.edge is not None: torch.nn.init.xavier_uniform_(self.edge.weight)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
