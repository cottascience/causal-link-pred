import torch
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, act=torch.nn.ReLU(), regression=False ):
        super(MLP, self).__init__()

        self.act = act
        self.regression = regression
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j, embedding=False):
        x = x_i * x_j
        
        for lin in self.lins[:-1]:
            x = lin(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if embedding:
            return x
        x = self.lins[-1](x)
        if self.regression: return x
        return torch.sigmoid(x)

class InnerProduct(torch.nn.Module):
    def __init__(self):
        super(InnerProduct, self).__init__()
        self.dummy = torch.nn.Parameter(torch.rand(1))
    def reset_parameters(self):
        pass
    def forward(self, x_i, x_j):
        return torch.sigmoid((x_i * x_j).sum(1).unsqueeze(1))

class KG(torch.nn.Module):
    def __init__(self, edge_channels, in_channels , hidden_channels, out_channels, num_layers,
                 dropout, act=torch.nn.ReLU() ):
        super(KG, self).__init__()

        self.edge = torch.nn.Embedding(edge_channels, hidden_channels)
        self.act = act
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(2*in_channels+hidden_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        torch.nn.init.xavier_uniform_(self.edge.weight)

    def forward(self, x_i, x_j, edge):
        
        x = torch.cat( [ x_i , x_j, self.edge(edge) ] ,1)
        
        for lin in self.lins[:-1]:
            x = lin(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)