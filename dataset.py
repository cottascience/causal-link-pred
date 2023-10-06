from statistics import variance
import torch,sys
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import torch_geometric.transforms as T
from torch_geometric.utils import degree, negative_sampling, to_networkx, from_networkx, subgraph
from torch_geometric.utils.loop import *
from torch_geometric.data import Data
from pprint import pprint
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from SEAL.dataset import SEALDataset
from torchkge.data_structures import KnowledgeGraph
from typing import Optional, Tuple, Union
from torch_geometric.typing import OptTensor
from torch import Tensor


class Dataset:
    def __init__(self, name, device, eval_method, seal, use_train_in_valid=False):
        self.name = name
        self.device = device
        self.eval_method = eval_method
        if self.name[:4] == "ogbl":
            self.build_ogb()
            ratio_per_hop = 0.2 if self.name == "obgl-ddi" else 1.0
            if seal: self.build_seal(0.1, ratio_per_hop)
        elif self.name == "electronics" or self.name == "music":
            self.build_recsys()
            if seal: self.build_seal(100.0, 1.0, 1)
        elif self.name == "family-tree":
            self.build_family_tree()
            if seal: self.build_seal(100.0, 1.0, 3)
        elif self.name == "covariance":
            self.build_covariance()
        else:
            raise Exception("dataset not implemented")
        if use_train_in_valid:
            self.split_edge["valid"]["edge"] = torch.cat( [ self.split_edge["valid"]["edge"], self.split_edge["train"]["edge"] ] )
            self.split_edge["valid"]["edge_neg"] = torch.cat( [ self.split_edge["valid"]["edge_neg"], self.split_edge["train"]["edge_neg"] ] )
        self.move_to_device()
    
    def get_self_loop_attr(self, edge_index: Tensor, edge_attr: OptTensor = None,
                       num_nodes: Optional[int] = None) -> Tensor:
        loop_mask = edge_index[0] == edge_index[1]
        loop_index = edge_index[0][loop_mask]

        if edge_attr is not None:
            loop_attr = edge_attr[loop_mask]
        else:
            loop_attr = torch.ones_like(loop_index, dtype=torch.float)
        num_nodes = edge_index.max() + 1
        full_loop_attr = loop_attr.new_zeros((num_nodes, ) + loop_attr.size()[1:])
        full_loop_attr[loop_index] = loop_attr
        return full_loop_attr
    
    def build_covariance(self):

        data = torch.load(self.name+"/covariance-dataset.tch")

        stds = torch.sqrt( self.get_self_loop_attr( data.edge_index, data.edge_attr ) )
        nodes_to_keep = torch.arange( stds.size(0) )[ stds != 0]
        num_nodes = data.edge_index.max() + 1
        data.edge_index, data.edge_attr = subgraph(nodes_to_keep, data.edge_index, data.edge_attr, relabel_nodes=False, num_nodes=num_nodes)
        data.int_edge_index_train, data.int_edge_attr_train = subgraph(nodes_to_keep, data.int_edge_index_train, data.int_edge_attr_train, relabel_nodes=False, num_nodes=num_nodes)
        data.int_edge_index_test, data.int_edge_attr_test = subgraph(nodes_to_keep, data.int_edge_index_test, data.int_edge_attr_test, relabel_nodes=False, num_nodes=num_nodes)

        for i in range(data.edge_index.size(1)):
            data.edge_attr[i] = data.edge_attr[i]/(stds[data.edge_index[0,i]]*stds[data.edge_index[1,i]])
        for i in range( data.int_edge_index_train.size(1) ):
            data.int_edge_attr_train[i] = data.int_edge_attr_train[i]/(stds[data.int_edge_index_train[0,i]]*stds[data.int_edge_index_train[1,i]])
        for i in range( data.int_edge_index_test.size(1) ):
            data.int_edge_attr_test[i] = data.int_edge_attr_test[i]/(stds[data.int_edge_index_test[0,i]]*stds[data.int_edge_index_test[1,i]] )
        
        data.edge_weight = data.edge_attr
        data = T.ToSparseTensor(remove_edge_index=False)(data)

        self.num_nodes = data.edge_index.max() + 1
        self.x = torch.ones( self.num_nodes, 1 )
        self.edge_index = data.edge_index
        self.adj_t = data.adj_t
        self.edge_weight = data.edge_attr
        self.split_edge = { "train": {}, "small_train": {}, "valid": {}, "test": {} }

        train_size = int(data.int_edge_attr_train.size(0)*0.9)
        perm = torch.randperm(data.int_edge_attr_train.size(0))

        self.split_edge["train"]["edge"] = { "index": data.int_edge_index_train.t()[perm][:train_size], "weight": data.int_edge_attr_train[perm][:train_size] }
        self.split_edge["valid"]["edge"] = { "index": data.int_edge_index_train.t()[perm][train_size:], "weight": data.int_edge_attr_train[perm][train_size:] }

        int_edge_attr_train_neg = data.int_edge_attr_train[torch.randperm(data.int_edge_attr_train.size(0))]
        self.split_edge["train"]["edge_neg"] = { "index": data.int_edge_index_train.t()[perm][:train_size], "weight": int_edge_attr_train_neg[perm][:train_size] }
        self.split_edge["valid"]["edge_neg"] = { "index": data.int_edge_index_train.t()[perm][train_size:], "weight": int_edge_attr_train_neg[perm][train_size:] }

        int_edge_attr_test_neg = data.int_edge_attr_test[torch.randperm( data.int_edge_attr_test.size(0))]
        self.split_edge["test"]["edge"] = { "index": data.int_edge_index_test.t(), "weight": data.int_edge_attr_test }
        self.split_edge["test"]["edge_neg"] = { "index": data.int_edge_index_test.t(), "weight": int_edge_attr_test_neg }
        self.observational = False
        self.KG = None
        self.split_edge["small_train"] = self.split_edge["train"]

    def build_family_tree(self):
        data = torch.load(self.name+"/data.pt")
        data = T.ToSparseTensor(remove_edge_index=False)(data)
        self.x = torch.zeros(( data.x.size(0), 2 ))
        for i in range( data.size(0)):
            self.x[ i, data.x[i] ] = 1
        self.edge_index = data.edge_index
        self.adj_t = data.adj_t
        self.edge_weight = None
        self.num_nodes = data.x.size(0)
        self.split_edge = { "train": {}, "small_train": {}, "valid": {}, "test": {} }

        train_edge = data.iso_left_edge_index.t()
        train_attr = data.iso_left_edge_attr
        train_neg_tails = torch.from_numpy(np.random.choice( torch.unique(data.iso_left_edge_index[1]).tolist(), replace=True, size=data.iso_left_edge_index[0].size(0) )).long()
        train_edge_neg = torch.cat( [ data.iso_left_edge_index[0].unsqueeze(1) ,\
            train_neg_tails.unsqueeze(1) ] , 1 )
        perm = torch.randperm( train_edge.size(0) )
        train_size = int(len(perm)*0.95)

        self.split_edge["train"]["edge"] = train_edge[perm][ : train_size ]
        self.split_edge["train"]["attr"] = train_attr[perm][ : train_size ]
        self.split_edge["train"]["edge_neg"] = train_edge_neg[perm][ : train_size ]
        self.split_edge["valid"]["edge"] = train_edge[perm][ train_size : ]
        self.split_edge["valid"]["attr"] = train_attr[perm][ train_size : ]
        self.split_edge["valid"]["edge_neg"] = train_edge_neg[perm][ train_size : ]

        self.split_edge["test"]["edge"] = data.iso_right_edge_index.t()
        self.split_edge["test"]["attr"] = data.iso_right_edge_attr
        self.split_edge["test"]["edge_neg"] = torch.cat( [ data.iso_right_edge_index[0].unsqueeze(1) ,\
            data.iso_left_edge_index[1].unsqueeze(1) ] , 1 )

        self.split_edge["small_train"]["edge"] = self.split_edge["train"]["edge"]
        self.split_edge["small_train"]["attr"] = self.split_edge["train"]["attr"]
        self.split_edge["small_train"]["edge_neg"] = self.split_edge["train"]["edge_neg"]

        ent2ix = {}
        rel2ix = { 0: 0 }
        for i in range(self.x.size(0)):
            ent2ix[i] = i
        self.KG = KnowledgeGraph( kg = {"heads": self.edge_index[0], "tails": self.edge_index[1], "relations": torch.zeros(self.edge_index.size(1),dtype=torch.long ) }, ent2ix=ent2ix,rel2ix=rel2ix)
        self.observational = False

    def build_recsys(self):
        data = torch.load(self.name+"/data.pt")
        data = T.ToSparseTensor(remove_edge_index=False)(data)
        self.x = data.x
        self.edge_index = data.edge_index
        self.adj_t = data.adj_t
        self.edge_weight = None
        self.num_nodes = data.num_nodes
        self.split_edge = torch.load(self.name+"/split_edge.pt")
        self.split_edge["small_train"] = {}
        self.split_edge["small_train"]["edge"] = self.split_edge["train"]["edge"]
        self.split_edge["small_train"]["edge_neg"] = self.split_edge["train"]["edge_neg"]
        self.observational = False
        self.KG = None

    def build_ogb(self):
        dataset = PygLinkPropPredDataset(name=self.name, root=self.name, transform=T.ToSparseTensor(remove_edge_index=False))
        data = dataset[0]
        if data.edge_weight is not None:
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)
            data = T.ToSparseTensor(remove_edge_index=False)(data)
        self.edge_weight = data.edge_weight
        self.split_edge = dataset.get_edge_split()
        self.adj_t = data.adj_t
        self.num_nodes = self.adj_t.sparse_sizes()[0]
        if data.x is None: 
            self.x = torch.ones((self.num_nodes,1))
        else:
            self.x = data.x.float()
        neg_edge_index = negative_sampling(data.edge_index, self.num_nodes, num_neg_samples=self.split_edge["train"]["edge"].size(0), method='sparse')
        self.split_edge["train"]["edge_neg"] = neg_edge_index.t()
        self.split_edge["small_train"] = {}
        self.split_edge["small_train"]["edge"] = self.subsample_edge( self.split_edge["train"]["edge"], 0.01 )
        self.split_edge["small_train"]["edge_neg"] = self.subsample_edge( self.split_edge["train"]["edge_neg"], 0.01 )
        data.num_nodes = self.num_nodes
        self.edge_index = data.edge_index
        self.observational = True
        self.KG = None
    
    def build_seal(self, train_percent, ratio_per_hop, num_hops=1, node_label="drnl"):

        self.SEALdata = { "train": {}, "valid": {}, "test": {}, "small_train": {} }
        seal_train = SEALDataset( self.name, Data(x=self.x, edge_index=self.edge_index, edge_weight=self.edge_weight ),\
             self.split_edge, num_hops=num_hops, percent=train_percent, split = "train", ratio_per_hop = ratio_per_hop, max_nodes_per_hop=None, node_label=node_label )
        self.SEALdata["train"]["edge"] = [ data for data in seal_train if data.y.item() == 1 ]
        self.SEALdata["train"]["edge_neg"] = [ data for data in seal_train if data.y.item() == 0 ]
        del seal_train        

        self.split_edge["valid"]["edge"] = self.split_edge["valid"]["edge"]
        self.split_edge["valid"]["edge_neg"] = self.split_edge["valid"]["edge_neg"]
        self.split_edge["test"]["edge"] = self.split_edge["test"]["edge"]
        self.split_edge["test"]["edge_neg"] = self.split_edge["test"]["edge_neg"]

        seal_valid = SEALDataset( self.name, Data(x=self.x, edge_index=self.edge_index, edge_weight=self.edge_weight ),\
             self.split_edge, num_hops=num_hops, percent=100.0, split = "valid", ratio_per_hop = ratio_per_hop, max_nodes_per_hop=None, node_label=node_label )
        self.SEALdata["valid"]["edge"] = [ data for data in seal_valid if data.y.item() == 1 ]
        self.SEALdata["valid"]["edge_neg"] = [ data for data in seal_valid if data.y.item() == 0 ]
        del seal_valid

        seal_test = SEALDataset( self.name, Data(x=self.x, edge_index=self.edge_index, edge_weight=self.edge_weight ),\
             self.split_edge, num_hops=num_hops, percent=100.0, split = "test", ratio_per_hop = ratio_per_hop, max_nodes_per_hop=None, node_label=node_label )
        self.SEALdata["test"]["edge"] = [ data for data in seal_test if data.y.item() == 1 ]
        self.SEALdata["test"]["edge_neg"] = [ data for data in seal_test if data.y.item() == 0 ]
        del seal_test

        self.split_edge["train"]["edge"] = self.subsample_edge( self.split_edge["train"]["edge"], train_percent/100 )
        self.split_edge["train"]["edge_neg"] = self.subsample_edge( self.split_edge["train"]["edge_neg"], train_percent/100 )
        self.SEALdata["small_train"]["edge"] = self.subsample_seal( self.SEALdata["train"]["edge"], 0.01 )
        self.SEALdata["small_train"]["edge_neg"] = self.subsample_seal( self.SEALdata["train"]["edge_neg"], 0.01 )

    def subsample_edge(self, edge, ratio):
        perm = torch.randperm( edge.size(0) )
        size = int(len(perm)*ratio)
        return edge[perm[:size]]
    
    def subsample_seal(self, edge, ratio):
        perm = torch.randperm( len(edge) )
        size = int(len(perm)*ratio)
        return [ edge[i] for i in range(len(edge)) if i in perm[:size].tolist() ]

    def remove_edges(self, edge):
        all_ids = self.num_nodes * self.edge_index[0] + self.edge_index[1]
        remove_ids = self.num_nodes * edge[0] + edge[1]
        to_keep = torch.isin(all_ids, remove_ids, invert=True)
        edge_weight = self.edge_weight[to_keep] if self.edge_weight is not None else None
        data = T.ToSparseTensor()(Data(x=self.x, edge_index=self.edge_index[:, to_keep], edge_weight = edge_weight))
        return data.adj_t

    def move_to_device(self):
        for key1 in self.split_edge:
            for key2 in self.split_edge[key1]:
                if isinstance(self.split_edge[key1][key2],dict):
                    for key3 in self.split_edge[key1][key2]:
                        self.split_edge[key1][key2][key3] = self.split_edge[key1][key2][key3].to(self.device)
                else:
                    self.split_edge[key1][key2] = self.split_edge[key1][key2].to(self.device)
        self.adj_t = self.adj_t.to(self.device)
        self.x = self.x.to(self.device)
        self.edge_index = self.edge_index.to(self.device)
        if self.edge_weight is not None: self.edge_weight = self.edge_weight.to(self.device)

    def evaluate(self, pos, neg):
        if self.eval_method[:4] == "Hits":
            evaluator = Evaluator(name='ogbl-ppa')
            K = int(self.eval_method.split("@")[1])
            evaluator.K = K
            result = evaluator.eval({
            'y_pred_pos': pos,
            'y_pred_neg': neg,
            })[f'hits@{K}']
        elif self.eval_method == "MRR":
            evaluator = Evaluator(name='ogbl-citation2')
            return evaluator.eval({
            'y_pred_pos': pos,
            'y_pred_neg': neg.unsqueeze(1),
            })['mrr_list'].mean().item()
        elif self.eval_method == "AUC":
            result = roc_auc_score( torch.cat( [torch.ones(len(pos)), torch.zeros(len(neg)) ] ).unsqueeze(1), torch.cat( [pos,neg] ).unsqueeze(1)  )
        elif self.eval_method == "Acc":
            result = accuracy_score( torch.cat( [torch.ones(len(pos)), torch.zeros(len(neg)) ] ).unsqueeze(1), torch.cat( [pos,neg] ).unsqueeze(1).round()  )
        else:
            raise Exception("eval_method not implemented")
        return result

