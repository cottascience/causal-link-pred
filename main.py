import argparse, torch, sys, scheduler, time, math
from turtle import pos
from posixpath import split
import numpy as np
import scipy.sparse as ssp
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, SAGEConv, TransformerConv
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from dataset import Dataset
from embedding import *
from link import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.contrib import tzip
from SEAL.models import *
from NeoGNN.models import *

" Family Tree "
" nohup python -u main.py --dataset family-tree --node_embedding NMF --device 0 > family-tree-NMF.txt & "
" nohup python -u main.py --dataset family-tree --node_embedding SVD --device 1 > family-tree-SVD.txt & "
" nohup python -u main.py --dataset family-tree --node_embedding ComplEx --device 2 > family-tree-ComplEx.txt & "
" nohup python -u main.py --dataset family-tree --node_embedding TransE --device 3 > family-tree-TransE.txt & "
" nohup python -u main.py --dataset family-tree --node_embedding DistMult --device 4 > family-tree-DistMult.txt &"
" nohup python -u main.py --dataset family-tree --node_embedding GCN --device 5 --positional --batch_size 1024 > family-tree-GCN-pos.txt &"
" nohup python -u main.py --dataset family-tree --node_embedding GCN --device 6 > family-tree-GCN.txt &"
" nohup python -u main.py --dataset family-tree --node_embedding SAGE --device 7 > family-tree-SAGE.txt &"
" nohup python -u main.py --dataset family-tree --node_embedding TRANSFORMER --device 7 > family-tree-TRANSFORMER.txt &"
" nohup python -u main.py --dataset family-tree --node_embedding GCN --label --batch_size 8 --device 0 > family-tree-GCN-label.txt & "
" nohup python -u main.py --dataset family-tree --seal --device 1 > family-tree-seal.txt & "
" nohup python -u main.py --dataset family-tree --neo --device 2 --use_A3  > family-tree-neo.txt & "

"Electronics"
" nohup python -u main.py --dataset electronics --device 0 --eval Hits@500 --hidden_channels 8 --num_layers 2 --node_embedding NMF --warmup_epochs 10  > electronics-NMF.txt & "
" nohup python -u main.py --dataset electronics --device 0 --eval Hits@500 --hidden_channels 8 --num_layers 2 --node_embedding SVD --warmup_epochs 10 > electronics-SVD.txt & "
" nohup python -u main.py --dataset electronics --device 1 --eval Hits@500 --hidden_channels 8 --num_layers 2 --node_embedding GCN --warmup_epochs 10  --positional > electronics-GCN-pos.txt & "
" nohup python -u main.py --dataset electronics --device 2 --eval Hits@500 --hidden_channels 8 --num_layers 2 --node_embedding GCN --warmup_epochs 10 > electronics-GCN.txt & "
" nohup python -u main.py --dataset electronics --device 3 --eval Hits@500 --hidden_channels 8 --num_layers 2 --node_embedding TRANSFORMER --warmup_epochs 10 > electronics-TRANSFORMER.txt & "
" nohup python -u main.py --dataset electronics --eval Hits@500 --hidden_channels 32 --num_layers 2 --seal --device 4 --warmup_epochs 10 --weight_decay 1e-5 > electronics-seal.txt & "
" 67.57249918540242 2.470315506 "
" nohup python -u main.py --dataset electronics --device 5 --label --batch_size 8 --eval Hits@500 --hidden_channels 8 --num_layers 2 --warmup_epochs 10 > electronics-GCN-label.txt & "

" Music "
" nohup python -u main.py --dataset music --device 0  --lr 0.001 --warmup_epochs 10 --node_embedding NMF --eval Hits@50 > music-NMF.txt & "
" nohup python -u main.py --dataset music --device 1  --lr 0.001 --warmup_epochs 10 --node_embedding SVD --eval Hits@50 > music-SVD.txt & "
" nohup python -u main.py --dataset music --device 2  --lr 0.001 --warmup_epochs 10 --eval Hits@50 > music-GCN.txt & "
" nohup python -u main.py --dataset music --device 3  --lr 0.001 --warmup_epochs 10 --pos --eval Hits@50 > music-GCN-pos.txt & "
" nohup python -u main.py --dataset music --device 4  --lr 0.001 --warmup_epochs 10 --node_embedding TRANSFORMER --eval Hits@50 > music-TRANSFORMER.txt & "
" nohup python -u main.py --dataset music --batch_size 10 --label --lr 0.001 --warmup_epochs 10 --eval Hits@50 > music-GCN-label.txt & "
" nohup python -u main.py --dataset music --device 6 --seal --hidden_channels 16 --lr 0.00001 --eval Hits@50 --warmup_epochs 10 > music-seal.txt & "

" Covariance "

" nohup python -u main.py --dataset covariance --device 0 --lr 0.001 --use_bn --hidden_channels 64 --batch_size 16  > covariance-GCN.txt & "
" nohup python -u main.py --dataset covariance --device 3 --lr 0.001 --use_bn --hidden_channels 64 --node_embedding SVD --batch_size 16  > covariance-SVD.txt & "
" nohup python -u main.py --dataset covariance --lr 0.001 --use_bn --hidden_channels 64  --label --label_exact --batch_size 16  > covariance-GCN-label-exact.txt & "

" nohup python -u main.py --dataset covariance --lr 0.001 --use_bn --hidden_channels 64  --label --label_exact --batch_size 16 --dropout 0.2 --mlp_dropout 0.2  > covariance-GCN-label-exact-2.txt & "
" nohup python -u main.py --dataset covariance --lr 0.001 --use_bn --hidden_channels 64  --label --label_exact --batch_size 16 --dropout 0.2 --mlp_dropout 0.0 --device 2  > covariance-GCN-label-exact-3.txt & "
" nohup python -u main.py --dataset covariance --lr 0.001 --use_bn --hidden_channels 64  --label --label_exact --batch_size 16 --dropout 0.0 --mlp_dropout 0.2 --device 3 > covariance-GCN-label-exact-4.txt & "
" nohup python -u main.py --dataset covariance --lr 0.001 --use_bn --hidden_channels 64  --label --label_exact --batch_size 16 --jk_mode cat --device 4 > covariance-GCN-label-exact-5.txt & "
" nohup python -u main.py --dataset covariance --lr 0.001 --use_bn --hidden_channels 64  --label --label_exact --batch_size 16 --jk_mode sum --device 5 > covariance-GCN-label-exact-6.txt & "
" nohup python -u main.py --dataset covariance --lr 0.001 --use_bn --hidden_channels 64  --label --label_exact --batch_size 16 --jk_mode none --device 6 > covariance-GCN-label-exact-7.txt & "

" nohup python -u main.py --device 1 --dataset covariance --lr 0.001 --use_bn --hidden_channels 64  --label --label_exact --batch_size 16 --jk_mode none --device 6 --dropout 0.2 --mlp_dropout 0 > covariance-GCN-label-exact-8.txt & "


parser = argparse.ArgumentParser(description="Link prediction tasks")
parser.add_argument("--dataset", type=str, default="behance")
parser.add_argument("--node_embedding", type=str, default="GCN", choices=["GCN", "SAGE", "SVD", "NMF", "TRANSFORMER", "TransE", "ComplEx", "DistMult"])
parser.add_argument("--eval_method", type=str, default="Hits@100") #Hits@K, AUC
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--hidden_channels", type=int, default=256)
" GNN params "
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--use_bn", action="store_true")
parser.add_argument("--jk_mode", type=str, default="mean",choices=["max","cat","mean","lstm","sum","none"])
parser.add_argument("--positional", action="store_true") # Force symmetric model (GNN) to be positional
" Label-GNN params "
parser.add_argument("--label", action="store_true")
parser.add_argument("--label_exact", action="store_true")
" SEAL params "
parser.add_argument("--seal", action="store_true")
parser.add_argument("--use_feature", action="store_true")
" NeoGNN params "
parser.add_argument("--neo", action="store_true")
parser.add_argument("--use_A3", action="store_true")
parser.add_argument("--use_A1", action="store_true")
parser.add_argument('--alpha', type=float, default=-1)
parser.add_argument('--beta', type=float, default=0.1)
" Predictor params "
parser.add_argument("--pred", type=str, default="MLP", choices=["MLP", "inner_product"])
parser.add_argument("--mlp_layers", type=int, default=1)
parser.add_argument('--mlp_dropout', type=float, default=0.5)
" Learning params "
parser.add_argument("--use_train_in_valid", action="store_true")
parser.add_argument("--epoch_ratio", type=float, default=1.0)
parser.add_argument("--warmup_epochs", type=int, default=0)
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--patience", type=int, default=15)
parser.add_argument("--epochs", type=int, default=20000)
parser.add_argument("--lr_scheduler", type=str, default="none", choices=["sgdr", "cos", "zigzag", "none"])
parser.add_argument("--runs", type=int, default=5)

parser.add_argument("--log_embeds", action="store_true")

args = parser.parse_args()
print(args)

if args.node_embedding == "SVD" and args.pred == "inner_product":
    raise Exception("SVD+inner_product not implemented")

device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
data = Dataset(args.dataset, device, args.eval_method, args.seal, use_train_in_valid=args.use_train_in_valid)

my_act = torch.nn.ELU()
edge_dim = None
gcn_norm = True
bn_stats = True
regression = False
if args.dataset == "covariance":
    my_act = torch.nn.ReLU6()
    edge_dim = 1
    gcn_norm = False
    bn_stats = False
    regression = True

if args.node_embedding in ["GCN", "SAGE", "TRANSFORMER"]:
    Conv = { "GCN" : ( GCNConv,[data.x.size(1), args.hidden_channels, False, False, True, gcn_norm], [args.hidden_channels, args.hidden_channels, False, False, True, gcn_norm] ), \
        "SAGE" : ( SAGEConv,[data.x.size(1), args.hidden_channels, True, True, False], [args.hidden_channels, args.hidden_channels, True, True, False] ), \
            "TRANSFORMER" : ( TransformerConv,[data.x.size(1), args.hidden_channels, 3, False, False, 0,edge_dim], [args.hidden_channels, args.hidden_channels,3, False, False,0,edge_dim] ), }[args.node_embedding]
    node_embedding = GNN(Conv, data.x.size(1), args.hidden_channels, args.num_layers,
                    args.dropout, act=my_act, positional=args.positional, num_nodes=data.num_nodes, \
                        jk_mode=args.jk_mode, use_bn=args.use_bn, label=args.label, bn_stats=bn_stats).to(device)
elif args.node_embedding == "SVD":
    node_embedding = SVD( data.adj_t, args.hidden_channels ).to(device)
elif args.node_embedding == "NMF":
    node_embedding = NMF( data.edge_index.cpu(), args.hidden_channels, data.num_nodes, device )
elif args.node_embedding == "MCSVD":
    node_embedding = MCSVD( data.adj_t, args.hidden_channels, data.num_nodes, act=my_act, nsamples=1 ).to(device)
elif args.node_embedding in ["TransE", "ComplEx", "DistMult"]:
    node_embedding = KGE( data.KG, args.hidden_channels, args.node_embedding , device=device ).to(device)
else:
    raise Exception("node_embedding not implemented")

if args.seal:
    if data.KG is None:
        seal_model = DGCNN(data.x.size(1), args.hidden_channels, args.num_layers, \
            train_dataset=data.SEALdata["train"]["edge"]+data.SEALdata["train"]["edge_neg"],
                            dynamic_train=False, use_feature=args.use_feature).to(device)
    else:
        seal_model = DGCNN(data.x.size(1), args.hidden_channels, args.num_layers, \
            train_dataset=data.SEALdata["train"]["edge"]+data.SEALdata["train"]["edge_neg"],
                            dynamic_train=False, use_feature=args.use_feature, edge_channels=data.split_edge["train"]["attr"].max()+1).to(device)
else:
    seal_model = DGCNN(data.x.size(1), args.hidden_channels, args.num_layers, \
        train_dataset=None,
                        dynamic_train=False, use_feature=args.use_feature).to(device)

edge_num_attr = 0 if data.KG is None else data.split_edge["train"]["attr"].max()+1

if data.KG is None:
    neo_model = model = NeoGNN(data.x.size(1), args.hidden_channels,
                        args.hidden_channels, args.num_layers,
                        args.dropout, f_edge_dim=8, f_node_dim=128, g_phi_dim=128, act=torch.nn.ReLU(), jk_mode=args.jk_mode).to(device)
else:
    neo_model = model = NeoGNN(data.x.size(1), args.hidden_channels,
                        args.hidden_channels, args.num_layers,
                        args.dropout, f_edge_dim=8, f_node_dim=128, g_phi_dim=128, \
                            edge_size=data.split_edge["train"]["attr"].max()+1, act=my_act ,jk_mode=args.jk_mode).to(device)

pred_input_channels = args.num_layers*args.hidden_channels if args.jk_mode == "cat" else args.hidden_channels

neo_pred = LinkPredictor(pred_input_channels, args.hidden_channels, 1,
                              args.num_layers, dropout=0.0).to(device)
                              
if data.KG is not None:
    neo_pred = KG( data.split_edge["train"]["attr"].max()+1, args.hidden_channels, args.hidden_channels, 1,
                                1, dropout=0.5, act=my_act ).to(device)

neo_params = { "A": None, "degree": None }

if args.neo:
    edge_weight = torch.ones(data.edge_index.size(1), dtype=float)
    A = ssp.csr_matrix((edge_weight, (data.edge_index[0].cpu(), data.edge_index[1].cpu())), shape=(data.num_nodes, data.num_nodes))
    A2 = A * A
    if args.use_A3:
        A3 = A2 * A
    else:
        A3 = 0
    if not args.use_A1: A = A + args.beta*A2 + (args.beta*args.beta)*A3
    neo_params["degree"] = torch.from_numpy(A.sum(axis=0)).squeeze()
    neo_params["A"] = A
    del edge_weight

if data.KG is not None:
   predictor =  KG( data.split_edge["train"]["attr"].max()+1, data.x.size(1)+args.hidden_channels, args.hidden_channels, 1,
                                1, dropout=0.5, act=my_act ).to(device)
elif args.pred == "MLP":
    pred_input_channels = args.num_layers*args.hidden_channels if args.jk_mode == "cat" else args.hidden_channels
    predictor = MLP(pred_input_channels, args.hidden_channels, 1,
                                args.mlp_layers, dropout=args.mlp_dropout, act=my_act, regression=regression).to(device)
elif args.pred == "inner_product":
    predictor = InnerProduct()
else:
    raise Exception("predictor not implemented")

if args.seal: 
    data.device = torch.device("cpu")
    data.move_to_device()

def train(node_embedding, predictor, seal_model, neo_model, neo_pred, optims):
    seal_model.train()
    node_embedding.train()
    predictor.train()
    neo_model.train()
    neo_pred.train()

    total_loss = total_examples = 0
    if data.KG is None and args.dataset != "covariance":
        train_edge = data.subsample_edge(data.split_edge["train"]["edge"], args.epoch_ratio)
        train_edge_neg = data.subsample_edge(data.split_edge["train"]["edge_neg"], args.epoch_ratio)
    elif args.dataset == "covariance":
        train_edge = data.split_edge["train"]["edge"]["index"]
        train_edge_neg = data.split_edge["train"]["edge_neg"]["index"]
        train_edge_weight = data.split_edge["train"]["edge"]["weight"]
        train_edge_neg_weight = data.split_edge["train"]["edge_neg"]["weight"]
    else:
        train_edge = data.split_edge["train"]["edge"]
        train_edge_neg = data.split_edge["train"]["edge_neg"]
        train_attr = data.split_edge["train"]["attr"]

    for perm1,perm2 in tzip(DataLoader(range(train_edge.size(0)), args.batch_size, shuffle=True),\
         DataLoader(range(train_edge_neg.size(0)), args.batch_size, shuffle=True)):
        _ = optims.update_lr(args.lr)
        optims.zero_grad()

        if args.dataset == "covariance":
            pos_weight = train_edge_weight[perm1].unsqueeze(1)
            neg_weight = train_edge_neg_weight[perm2].unsqueeze(1)
            edge_weight = data.edge_weight
        else:
            edge_weight = None

        if args.neo:
            pos_edge = train_edge[perm1].t()
            neg_edge = train_edge_neg[perm2].t()
            if data.KG is None:
                pos_out, pos_out_struct, pos_out_feat = neo_model(pos_edge, data, neo_params["A"], neo_pred)
                neg_out, neg_out_struct, neg_out_feat = neo_model(neg_edge, data, neo_params["A"], neo_pred)
            else:
                pos_out, pos_out_struct, pos_out_feat = neo_model(pos_edge, data, neo_params["A"], neo_pred, edge_attr=train_attr[perm1])
                neg_out, neg_out_struct, neg_out_feat = neo_model(neg_edge, data, neo_params["A"], neo_pred, edge_attr=train_attr[perm2])
            pos_loss = -torch.log(pos_out_struct + 1e-15).mean()
            neg_loss = -torch.log(1 - neg_out_struct + 1e-15).mean()
            loss1 = pos_loss + neg_loss
            pos_loss = -torch.log(pos_out_feat + 1e-15).mean()
            neg_loss = -torch.log(1 - neg_out_feat + 1e-15).mean()
            loss2 = pos_loss + neg_loss
            pos_loss = -torch.log(pos_out + 1e-15).mean()
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
            loss3 = pos_loss + neg_loss
            loss = loss1 + loss2 + loss3

        elif args.seal:
            pos_edge = Batch().from_data_list([ data.SEALdata["train"]["edge"][i] for i in range(len( data.SEALdata["train"]["edge"] )) if i in perm1.tolist() ]).to(device)
            neg_edge = Batch().from_data_list([ data.SEALdata["train"]["edge_neg"][i] for i in range(len( data.SEALdata["train"]["edge_neg"] )) if i in perm2.tolist() ]).to(device)
            if data.KG is None:
                pos_out = seal_model(pos_edge)
                neg_out = seal_model(neg_edge)
            else:
                pos_out = seal_model(pos_edge,train_attr[perm1].to(device))
                neg_out = seal_model(neg_edge,train_attr[perm2].to(device))
            loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
            loss = loss_fn( torch.cat([pos_out,neg_out]), torch.cat([torch.ones(len(pos_out)),torch.zeros(len(neg_out))]).reshape(torch.cat([pos_out,neg_out]).size()).to(device) )

        else:
            pos_edge = train_edge[perm1].t()
            adj_t = data.remove_edges(pos_edge) if args.label and data.observational else data.adj_t
            neg_edge = train_edge_neg[perm2].t()
            batch_nodes = torch.unique( torch.cat([ pos_edge, neg_edge ], dim=1) )
            if data.KG is None:
                if args.label_exact:
                    pos_out_list = []
                    for idx in range(pos_edge.size(1)):
                        curr_edge = pos_edge[:, idx]
                        h = node_embedding(data.x, adj_t, curr_edge, True, edge_weight=edge_weight, edge_index=data.edge_index)
                        pos_out_list.append(predictor(h[curr_edge[0]], h[curr_edge[1]]))
                    pos_out = torch.cat(pos_out_list).unsqueeze(1)
                else:
                    h = node_embedding(data.x, adj_t, batch_nodes, True, edge_weight=edge_weight, edge_index=data.edge_index)
                    pos_out = predictor(h[pos_edge[0]], h[pos_edge[1]])
                    neg_out = predictor(h[neg_edge[0]], h[neg_edge[1]])
            else:
                h1,h2 = node_embedding(data.x, adj_t, batch_nodes, False)
                pos_out = predictor( torch.cat([data.x[pos_edge[0]],h1[pos_edge[0]]],1), torch.cat([data.x[pos_edge[1]],h2[pos_edge[1]]],1), train_attr[perm1])
                neg_out = predictor( torch.cat([data.x[neg_edge[0]],h1[neg_edge[0]]],1), torch.cat([data.x[neg_edge[1]],h2[neg_edge[1]]],1), train_attr[perm2] )
            
            if args.dataset == "covariance":
                #lf = torch.nn.L1Loss()
                lf = torch.nn.MSELoss()
                pos_loss =  lf(pos_out, pos_weight)
                neg_loss =  0
            else:
                pos_loss = -torch.log(pos_out + 1e-15).mean()
                neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
            loss = pos_loss + neg_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(node_embedding.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(neo_model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(neo_pred.parameters(), 1.0)

        optims.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples

@torch.no_grad()
def test(node_embedding, predictor, seal_model, neo_model, neo_pred, split):
    seal_model.eval()
    node_embedding.eval()
    predictor.eval()
    neo_model.eval()
    neo_pred.eval()

    if args.dataset == "covariance":
        pos_edge = data.split_edge[split]["edge"]["index"]
        neg_edge = data.split_edge[split]["edge_neg"]["index"]
        pos_weight = data.split_edge[split]["edge"]["weight"].detach().cpu()
        neg_weight = data.split_edge[split]["edge_neg"]["weight"].detach().cpu()
        edge_weight = data.edge_weight
    elif args.seal:
        pos_edge = data.SEALdata[split]["edge"]
        neg_edge = data.SEALdata[split]["edge_neg"]
    else:
        pos_edge = data.split_edge[split]["edge"]
        neg_edge = data.split_edge[split]["edge_neg"]
        edge_weight = None    

    neo_params_ = { "alpha": None, "h": None, "edge_weight": None, "edge_index": None }
    if args.neo:
        neo_params_["edge_weight"] = torch.from_numpy(neo_params["A"].data).to(device)
        neo_params_["edge_weight"] = neo_model.f_edge(neo_params_["edge_weight"].unsqueeze(-1))
        neo_params_["alpha"] = torch.softmax(neo_model.alpha, dim=0).cpu()
        neo_params_["h"] = neo_model.get_gnn_rep(data.x, data.adj_t)
        row, col = neo_params["A"].nonzero()
        neo_params_["edge_index"] = torch.stack([torch.from_numpy(row), torch.from_numpy(col)]).type(torch.LongTensor).to(device)
        row, col = neo_params_["edge_index"][0], neo_params_["edge_index"][1]
        deg = scatter_add(neo_params_["edge_weight"], col, dim=0, dim_size=data.num_nodes)
        deg =  neo_model.f_node(deg).squeeze()
        deg = deg.cpu().numpy()
        neo_params_["A_"] = neo_params["A"].multiply(deg).tocsr()

    pos_preds = []
    pos_embs =[]
    for perm in tzip(DataLoader(range(len(pos_edge)), shuffle=False, batch_size=args.batch_size)):
        perm = perm[0]
        if args.neo:
            edge = pos_edge[perm].t()
            if data.KG is None:
                gnn_scores = neo_pred(neo_params_["h"][edge[0]], neo_params_["h"][edge[1]] ).squeeze().cpu()
            else:
                gnn_scores = neo_pred(neo_params_["h"][edge[0]], neo_params_["h"][edge[1]], data.split_edge[split]["attr"][perm] ).squeeze().cpu()
            src, dst = pos_edge[perm].t().cpu()
            cur_scores = torch.from_numpy(np.sum(neo_params_["A_"][src].multiply(neo_params_["A_"][dst]), 1)).to(device)
            if data.KG is None:
                cur_scores = torch.sigmoid(neo_model.g_phi(cur_scores).squeeze().cpu())  
            else:
                cur_scores = torch.sigmoid(neo_model.g_phi(torch.cat([cur_scores,neo_model.edge(data.split_edge[split]["attr"][perm].to(device))],1)).squeeze().cpu())
            cur_scores = neo_params_["alpha"][0]*cur_scores + neo_params_["alpha"][1] * gnn_scores
            pos_preds += [cur_scores]
        if args.seal:
            edge = Batch().from_data_list([ pos_edge[i] for i in range(len(pos_edge)) if i in perm.tolist() ]).to(device)
            if data.KG is None:
                pos_preds += [seal_model(edge).squeeze().cpu()]
            else:
                pos_preds += [seal_model(edge, data.split_edge[split]["attr"][perm].to(device) ).squeeze().cpu()]
        else:
            edge = pos_edge[perm].t()
            batch_nodes = torch.unique( edge )
            if data.KG is None:
                if args.label_exact:
                    pos_out_list = []
                    for idx in range(edge.size(1)):
                        curr_edge = edge[:, idx]
                        h = node_embedding(data.x, data.adj_t, curr_edge, True, edge_weight=edge_weight, edge_index=data.edge_index)
                        pos_out_list.append(predictor(h[curr_edge[0]], h[curr_edge[1]]))
                    pos_out = torch.cat(pos_out_list)
                    to_append = [pos_out.squeeze().cpu()]
                    if len(to_append) > 0: pos_preds += to_append
                else:
                    h = node_embedding(data.x, data.adj_t, batch_nodes, True, edge_weight=edge_weight, edge_index=data.edge_index)
                    to_append = [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
                    if len(to_append) > 0:
                        pos_preds += to_append
                        pos_embs += [predictor(h[edge[0]], h[edge[1]], embedding=True).squeeze().cpu()]
            else:
                h1,h2 = node_embedding(data.x, data.adj_t, batch_nodes, False)
                pos_preds += [ predictor( torch.cat([data.x[edge[0]],h1[edge[0]]],1), torch.cat([data.x[edge[1]],h2[edge[1]]],1), data.split_edge[split]["attr"][perm] ).squeeze().cpu()]
    for i in range(len(pos_preds)):
        if len(pos_preds[i].size()) == 0: pos_preds[i] = pos_preds[i].unsqueeze(0)
    pos_pred = torch.cat(pos_preds, dim=0)

    if args.dataset == "covariance":
        #lf = torch.nn.L1Loss()
        lf = torch.nn.MSELoss()
        return lf(pos_pred.unsqueeze(1), pos_weight.unsqueeze(1)).item()

    neg_preds = []
    neg_embs = []
    for perm in tzip(DataLoader(range(len(neg_edge)), args.batch_size)):
        perm = perm[0]
        if args.neo:
            edge = neg_edge[perm].t()
            if data.KG is None:
                gnn_scores = neo_pred(neo_params_["h"][edge[0]], neo_params_["h"][edge[1]] ).squeeze().cpu()
            else:
                gnn_scores = neo_pred(neo_params_["h"][edge[0]], neo_params_["h"][edge[1]], data.split_edge[split]["attr"][perm] ).squeeze().cpu()
            src, dst = neg_edge[perm].t().cpu()
            cur_scores = torch.from_numpy(np.sum(neo_params_["A_"][src].multiply(neo_params_["A_"][dst]), 1)).to(device)
            if data.KG is None:
                cur_scores = torch.sigmoid(neo_model.g_phi(cur_scores).squeeze().cpu())  
            else:
                cur_scores = torch.sigmoid(neo_model.g_phi(torch.cat([cur_scores,neo_model.edge(data.split_edge[split]["attr"][perm].to(device))],1)).squeeze().cpu())
            cur_scores = neo_params_["alpha"][0]*cur_scores + neo_params_["alpha"][1] * gnn_scores
            neg_preds += [cur_scores]
        if args.seal:
            edge = Batch().from_data_list([ neg_edge[i] for i in range(len(neg_edge)) if i in perm.tolist() ]).to(device)
            if data.KG is None:
                neg_preds += [seal_model(edge).squeeze().cpu()]
            else:
                neg_preds += [seal_model(edge, data.split_edge[split]["attr"][perm].to(device)).squeeze().cpu()]
        else:
            edge = neg_edge[perm].t()
            batch_nodes = torch.unique( edge )
            if data.KG is None:
                h = node_embedding(data.x, data.adj_t, batch_nodes)
                neg_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
                neg_embs += [predictor(h[edge[0]], h[edge[1]], embedding=True).squeeze().cpu()]
            else:
                h1,h2 = node_embedding(data.x, data.adj_t, batch_nodes, False)
                neg_preds += [ predictor( torch.cat([data.x[edge[0]],h1[edge[0]]],1), torch.cat([data.x[edge[1]],h2[edge[1]]],1), data.split_edge[split]["attr"][perm] ).squeeze().cpu() ]
    neg_pred = torch.cat(neg_preds, dim=0)

    if args.log_embeds and split == "test":
        torch.save(torch.cat(pos_embs, dim=0), 'positive.pt')
        torch.save(torch.cat(neg_embs, dim=0), 'negative.pt')
    
    return data.evaluate(pos_pred, neg_pred)*100

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

results = []
for run in range(args.runs):
    seal_model.reset_parameters()
    node_embedding.reset_parameters()
    predictor.reset_parameters()
    neo_model.reset_parameters()
    neo_pred.reset_parameters()
    optimizer = torch.optim.Adam(
            list(node_embedding.parameters()) + list(seal_model.parameters()) + \
                list(neo_model.parameters()) + list(neo_pred.parameters()) +
            list(predictor.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    optims = scheduler.MultipleOptimizer(args.lr_scheduler, optimizer)
    train_result = valid_result = test_result = best_valid = final_result = 0
    if regression: best_valid = -100000000000
    cnt_wait = 0
    for epoch in range(1, 1 + args.epochs):
        stime = time.time()
        print( "Train:" )
        loss = train( node_embedding, predictor, seal_model, neo_model, neo_pred, optims )
        if  epoch > args.warmup_epochs:
            if regression:
                train_result = loss
            else:
                train_result = test( node_embedding, predictor, seal_model, neo_model, neo_pred, "small_train" )
            print("Valid:")
            valid_result = test( node_embedding, predictor, seal_model, neo_model, neo_pred, "valid" )
            print("Test:")
            test_result = test( node_embedding, predictor, seal_model, neo_model, neo_pred, "test" )
        print( "Epoch:\t", epoch, "Loss:\t", loss, \
            "Train:\t", train_result, "Valid:\t", valid_result, \
            "Test:\t", test_result, "LR:\t", get_lr(optimizer), \
                "Patience:\t", cnt_wait, "Time:\t", time.time()-stime )
        if regression: valid_result = -valid_result
        if valid_result > best_valid or epoch < args.warmup_epochs:
            cnt_wait = 0
            best_valid = valid_result
            final_result = test_result
        else:
            cnt_wait += 1
        if cnt_wait == args.patience: break
    results.append ( final_result )
    print( "RUN\t", run, results )

print( "Final result:\t", np.array(results).mean(), np.array(results).std() )