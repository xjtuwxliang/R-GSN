import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_geometric.data import NeighborSampler
from torch_geometric.utils.hetero import group_hetero_graph
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import ast
import os
import argparse
import numpy as np
from tqdm import tqdm
from logger import Logger
from models import RGNN
from attacks import flag
from utils import gen_features, get_n_params, args_print, EarlyStopping


parser = argparse.ArgumentParser(description='OGBN-MAG (R-GSN)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.004)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--opt', type=str, default='adamw')
parser.add_argument('--early_stop', type=int, default=1)
parser.add_argument('--feat_dir', type=str, default='feat', help='origin feature dir')

parser.add_argument('--conv_name', type=str, default='rgsn', help='rgcn or rgsn')
parser.add_argument('--Norm4', type=ast.literal_eval, default=True) # 1+
parser.add_argument('--FDFT', type=ast.literal_eval, default=True)     # 2+
parser.add_argument('--use_attack', type=ast.literal_eval, default=False)  # 3+
args = parser.parse_args()
args_print(args)

dataset = PygNodePropPredDataset(name='ogbn-mag')
data = dataset[0]
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-mag')
logger = Logger(args.runs, args)

# We do not consider those attributes for now.
data.node_year_dict = None
data.edge_reltype_dict = None

print(data)
edge_index_dict = data.edge_index_dict

# We need to add reverse edges to the heterogeneous graph.
r, c = edge_index_dict[('author', 'affiliated_with', 'institution')]
edge_index_dict[('institution', 'to', 'author')] = torch.stack([c, r])

r, c = edge_index_dict[('author', 'writes', 'paper')]
edge_index_dict[('paper', 'to', 'author')] = torch.stack([c, r])

r, c = edge_index_dict[('paper', 'has_topic', 'field_of_study')]
edge_index_dict[('field_of_study', 'to', 'paper')] = torch.stack([c, r])

# Convert to undirected paper <-> paper relation.
edge_index = to_undirected(edge_index_dict[('paper', 'cites', 'paper')])
edge_index_dict[('paper', 'cites', 'paper')] = edge_index


if not os.path.exists(args.feat_dir):
    os.mkdir(args.feat_dir)
    ###### for field_of_study
    print('###### for field_of_study')
    rows = edge_index_dict[('field_of_study', 'to', 'paper')][0]
    cols = edge_index_dict[('field_of_study', 'to', 'paper')][1]
    v = torch.ones(rows.size())
    m, n = data.num_nodes_dict['field_of_study'], data.num_nodes_dict['paper']
    y = data.x_dict['paper']
    out = gen_features(rows, cols, v, m, n, y)
    np.save(f'{args.feat_dir}/field_of_study_FEAT.npy', out)

    ###### for author
    print('###### for author')
    rows = edge_index_dict[('author', 'writes', 'paper')][0]
    cols = edge_index_dict[('author', 'writes', 'paper')][1]
    v = torch.ones(rows.size())
    m, n = data.num_nodes_dict['author'], data.num_nodes_dict['paper']
    y = data.x_dict['paper']
    out = gen_features(rows, cols, v, m, n, y)
    np.save(f'{args.feat_dir}/author_FEAT.npy', out)

    ###### for institution
    print('###### for institution')
    rows = edge_index_dict[('institution', 'to', 'author')][0]
    cols = edge_index_dict[('institution', 'to', 'author')][1]
    v = torch.ones(rows.size())
    m, n = data.num_nodes_dict['institution'], data.num_nodes_dict['author']
    y = np.load(f'{args.feat_dir}/author_FEAT.npy')
    out = gen_features(rows, cols, v, m, n, y)
    np.save(f'{args.feat_dir}/institution_FEAT.npy', out)
print("preprocess finished")

# We convert the individual graphs into a single big one, so that sampling
# neighbors does not need to care about different edge types.
# This will return the following:
# * `edge_index`: The new global edge connectivity.
# * `edge_type`: The edge type for each edge.
# * `node_type`: The node type for each node.
# * `local_node_idx`: The original index for each node.
# * `local2global`: A dictionary mapping original (local) node indices of
#    type `key` to global ones.
# `key2int`: A dictionary that maps original keys to their new canonical type.
out = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)
edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out

# Map informations to their canonical type.
x_dict = {}
for key, x in data.x_dict.items():
    x_dict[key2int[key]] = x

num_nodes_dict = {}
for key, N in data.num_nodes_dict.items():
    num_nodes_dict[key2int[key]] = N

paper_idx = local2global['paper']
paper_train_idx = paper_idx[split_idx['train']['paper']]
paper_val_test_idx = torch.cat([paper_idx[split_idx['valid']['paper']], paper_idx[split_idx['test']['paper']]])

train_loader = NeighborSampler(edge_index, node_idx=paper_train_idx,
                               sizes=[25, 20], batch_size=args.batch_size, shuffle=True, num_workers=12)

infer_train_loader = NeighborSampler(edge_index, node_idx=paper_train_idx,
                               sizes=[25, 20], batch_size=4096, shuffle=True, num_workers=12)

infer_val_test_loader = NeighborSampler(edge_index, node_idx=paper_val_test_idx,
                               sizes=[-1, -1], batch_size=args.test_batch_size, shuffle=True, num_workers=12)

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

model = RGNN(128, args.hidden_channels, dataset.num_classes, args.num_layers,
             args.dropout, num_nodes_dict, list(x_dict.keys()),
             len(edge_index_dict.keys()), args).to(device)

print('Model #Params: %d' % get_n_params(model))

# Create global label vector.
y_global = node_type.new_full((node_type.size(0), 1), -1)
y_global[local2global['paper']] = data.y_dict['paper']

# Move everything to the GPU.
x_dict = {k: v.to(device) for k, v in x_dict.items()}
edge_type = edge_type.to(device)
node_type = node_type.to(device)
local_node_idx = local_node_idx.to(device)
y_global = y_global.to(device)

def train_vanilla(epoch):
    model.train()
    pbar = tqdm(total=paper_train_idx.size(0))
    pbar.set_description(f'Vanilla Epoch {epoch:02d}')
    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        n_id = n_id.to(device)
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out = model(n_id, x_dict, adjs, edge_type, node_type, local_node_idx)
        y = y_global[n_id][:batch_size].squeeze()
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_size
        pbar.update(batch_size)
        train_acc = evaluator.eval({
            'y_true': y.cpu().detach().numpy().reshape(-1,1),
            'y_pred': out.cpu().detach().numpy().argmax(1).reshape(-1,1),
        })['acc']
        pbar.set_description(f'Vanilla Epoch {epoch:02d}, train acc: {100 * train_acc:.2f}')
    pbar.close()
    loss = total_loss / paper_train_idx.size(0)
    return loss


def train_attack(epoch, model, optimizer):
    pbar = tqdm(total=paper_train_idx.size(0))
    pbar.set_description(f'Attack Epoch {epoch:02d}')
    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        n_id = n_id.to(device)
        adjs = [adj.to(device) for adj in adjs]
        forward = lambda perturb: model(n_id, x_dict, adjs, edge_type, node_type, local_node_idx, perturb)
        model_forward = (model, forward)
        perturb_shape = ((node_type[n_id] == 3).sum(), 128)
        y = y_global[n_id][:batch_size].squeeze()
        loss, _ = flag(model_forward, perturb_shape, y, args, optimizer, device, F.nll_loss)
        total_loss += loss.item() * batch_size
        pbar.update(batch_size)
    pbar.close()
    loss = total_loss / paper_train_idx.size(0)
    return loss


@torch.no_grad()
def infer():
    model.eval()
    y_true = data.y_dict['paper']
    y_pred = data.y_dict['paper'].new_full((data.y_dict['paper'].size(0), 1), -1)

    ### approximate for train data
    pbar = tqdm(total=paper_train_idx.size(0) + paper_val_test_idx.size(0))
    pbar.set_description('* infer train approximate')
    for batch_size, n_id, adjs in infer_train_loader:
        n_id = n_id.to(device)
        adjs = [adj.to(device) for adj in adjs]
        out = model(n_id, x_dict, adjs, edge_type, node_type, local_node_idx)
        y_pred[n_id[:batch_size]-local2global['paper'][0]] = out.argmax(dim=-1, keepdim=True).cpu()
        pbar.update(batch_size)

    ### exact for valid and test data
    pbar.set_description('* infer valid_test exact ')
    for batch_size, n_id, adjs in infer_val_test_loader:
        n_id = n_id.to(device)
        adjs = [adj.to(device) for adj in adjs]
        out = model(n_id, x_dict, adjs, edge_type, node_type, local_node_idx)
        y_pred[n_id[:batch_size]-local2global['paper'][0]] = out.argmax(dim=-1, keepdim=True).cpu()
        pbar.update(batch_size)
    pbar.close()

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']['paper']],
        'y_pred': y_pred[split_idx['train']['paper']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']['paper']],
        'y_pred': y_pred[split_idx['valid']['paper']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']['paper']],
        'y_pred': y_pred[split_idx['test']['paper']],
    })['acc']

    return train_acc, valid_acc, test_acc


@torch.no_grad()
def test():
    model.eval()
    out = model.inference(x_dict, edge_index_dict, key2int)
    out = out[key2int['paper']]
    y_pred = out.argmax(dim=-1, keepdim=True).cpu()
    y_true = data.y_dict['paper']

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']['paper']],
        'y_pred': y_pred[split_idx['train']['paper']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']['paper']],
        'y_pred': y_pred[split_idx['valid']['paper']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']['paper']],
        'y_pred': y_pred[split_idx['test']['paper']],
    })['acc']

    return train_acc, valid_acc, test_acc


for run in range(args.runs):
    model.reset_parameters()
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-06)

    es = EarlyStopping(args.early_stop)
    for epoch in range(1, 1 + args.epochs):
        if args.use_attack:
            loss = train_attack(epoch, model, optimizer)  # flag
        else:
            loss = train_vanilla(epoch)
        result = infer()
        logger.add_result(run, result)
        train_acc, valid_acc, test_acc = result
        print(f'Run: {run + 1:02d}, '
              f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}%, '
              f'Test: {100 * test_acc:.2f}%')
        es(valid_acc)
        if es.early_stop:
            break

    logger.print_statistics(run)
logger.print_statistics()

