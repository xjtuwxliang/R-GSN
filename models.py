import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_geometric.nn import MessagePassing

import os
import numpy as np
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot
from utils import MsgNorm


class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_types,
                 num_edge_types, args):
        super(RGCNConv, self).__init__(aggr='mean', node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.args = args

        self.rel_lins = ModuleList([Linear(in_channels, out_channels, bias=False) for _ in range(num_edge_types)])
        self.root_lins = ModuleList([Linear(in_channels, out_channels, bias=True) for _ in range(num_node_types)])

        if self.args.Norm4:
            self.msg_norm = ModuleList([MsgNorm(True) for _ in range(num_node_types)])
            self.layer_norm = ModuleList([torch.nn.LayerNorm(out_channels) for _ in range(num_node_types)])

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins:
            lin.reset_parameters()
        for lin in self.root_lins:
            lin.reset_parameters()
        if self.args.Norm4:
            for n in self.msg_norm:
                n.reset_parameters()
            for n in self.layer_norm:
                n.reset_parameters()

    def forward(self, x, edge_index, edge_type, target_node_type, src_node_type):
        x_src, x_target = x
        out = x_target.new_zeros(x_target.size(0), self.out_channels)

        for i in range(self.num_edge_types):
            mask = edge_type == i
            if self.args.Norm4:
                out.add_(F.normalize(self.propagate(edge_index[:, mask], x=x, edge_type=i, src_node_type = src_node_type)))
            else:
                out.add_(self.propagate(edge_index[:, mask], x=x, edge_type=i))

        for i in range(self.num_node_types):
            mask = target_node_type == i
            if self.args.Norm4:
                x = self.root_lins[i](x_target[mask])
                out[mask] = x + self.msg_norm[i](x, out[mask])
                out[mask] = self.layer_norm[i](out[mask])
            else:
                out[mask] += self.root_lins[i](x_target[mask])

        return out

    def message(self, x_j, edge_type: int):
        return self.rel_lins[edge_type](x_j)


class RGSNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_types,
                 num_edge_types, args):
        super(RGSNConv, self).__init__(aggr='add', node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.args = args

        self.rel_lins = ModuleList([Linear(in_channels, out_channels, bias=False) for _ in range(num_edge_types)])
        self.root_lins = ModuleList([Linear(in_channels, out_channels, bias=True) for _ in range(num_node_types)])

        if self.args.Norm4:
            self.msg_norm = ModuleList([MsgNorm(True) for _ in range(num_node_types)])
            self.layer_norm = ModuleList([torch.nn.LayerNorm(out_channels) for _ in range(num_node_types)])

        self.intra_attn_l = Parameter(torch.Tensor(num_edge_types, 1, in_channels))
        self.intra_attn_r = Parameter(torch.Tensor(num_edge_types, 1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins:
            lin.reset_parameters()
        for lin in self.root_lins:
            lin.reset_parameters()
        if self.args.Norm4:
            for n in self.msg_norm:
                n.reset_parameters()
            for n in self.layer_norm:
                n.reset_parameters()

        glorot(self.intra_attn_l)
        glorot(self.intra_attn_r)

    def forward(self, x, edge_index, edge_type, target_node_type, src_node_type):
        x_src, x_target = x
        out = x_target.new_zeros(x_target.size(0), self.out_channels)

        ######### Intra-Relation AGGR:  SIM-ATTN AGGR
        out_list = []
        nx = F.normalize(x[0])
        for i in range(self.num_edge_types):
            mask = edge_type == i
            ei = edge_index[:, mask]
            r, c = ei

            msg_from_i = F.normalize(self.propagate(ei, x=x, edge_type=i, src_node_type = src_node_type))
            a_l = (nx * self.intra_attn_l[i]).sum(-1)[r]
            a_r = (msg_from_i * self.intra_attn_r[i]).sum(-1)[c]
            a = a_l + a_r

            msg_from_i = F.normalize(self.propagate(ei, x=x, edge_type=i, src_node_type = src_node_type, a=a))
            out_list.append(msg_from_i)

        ######### Inter-Relation AGGR:  SIM AGGR
        for j in range(self.num_node_types):
            mask = target_node_type == j
            out_list_for_j = [o[mask] for o in out_list]

            out_j = torch.stack(out_list_for_j, dim=0)  # [7, r, 128]
            rep1 = torch.mean(out_j, dim=1)     # [7, 128]
            rep2 = rep1.mean(0, keepdim=True)   # [1, 128]
            rep1_norm = rep1 / (rep1.norm(dim=-1, keepdim=True) + 1e-5)
            rep2_norm = rep2 / (rep2.norm(dim=-1, keepdim=True) + 1e-5)

            b = (rep1_norm * rep2_norm).sum(dim=-1)
            b = F.softmax(b, dim=0).unsqueeze(-1).unsqueeze(-1)

            out_j = torch.sum(b * out_j, dim=0)
            out[mask] = out_j

        ########## Status Update
        for i in range(self.num_node_types):
            mask = target_node_type == i
            if self.args.Norm4:
                x = self.root_lins[i](x_target[mask])
                out[mask] = x + self.msg_norm[i](x, out[mask])
                out[mask] = self.layer_norm[i](out[mask])
            else:
                out[mask] += self.root_lins[i](x_target[mask])

        return out

    def message(self, edge_index_i, x_i, x_j, src_node_type_j,  edge_type: int , a=None):
        if a == None:
            res = x_j
        else:
            if x_i.size(0) == 0:
                return self.rel_lins[edge_type](x_j)
            a = softmax(a, edge_index_i)
            res = a.unsqueeze(-1) * self.rel_lins[edge_type](x_j)  ######## Message Transform
        return res


class RGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_nodes_dict, x_types, num_edge_types, args):
        super(RGNN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.args = args

        node_types = list(num_nodes_dict.keys())
        num_node_types = len(node_types)

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # Create embeddings for all node types that do not come with features.
        self.emb_dict = ParameterDict({
            f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })

        I, H, O = in_channels, hidden_channels, out_channels  # noqa

        if self.args.conv_name == 'rgcn':
            self.convs = ModuleList()
            self.convs.append(RGCNConv(I, H, num_node_types, num_edge_types, self.args))
            for _ in range(num_layers - 2):
                self.convs.append(RGCNConv(H, H, num_node_types, num_edge_types, self.args))
            self.convs.append(RGCNConv(H, O, self.num_node_types, num_edge_types, self.args))
        else:
            self.convs = ModuleList()
            self.convs.append(RGSNConv(I, H, num_node_types, num_edge_types, self.args))
            for _ in range(num_layers - 2):
                self.convs.append(RGSNConv(H, H, num_node_types, num_edge_types, self.args))
            self.convs.append(RGSNConv(H, O, self.num_node_types, num_edge_types, self.args))

        if self.args.Norm4:
            self.norm = torch.nn.LayerNorm(I)

        self.reset_parameters()

    def reset_parameters(self):
        root = self.args.feat_dir
        Feat_list = [os.path.join(root, i) for i in ['./author_FEAT.npy', './field_of_study_FEAT.npy', './institution_FEAT.npy']]

        for emb, Feat_path in zip(self.emb_dict.values(), Feat_list):
            if self.args.FDFT:
                emb.data = torch.Tensor(np.load(Feat_path)).to(self.args.device)
                emb.requires_grad = True
            else:
                torch.nn.init.xavier_uniform_(emb)

        for conv in self.convs:
            conv.reset_parameters()

        if self.args.Norm4:
            self.norm.reset_parameters()

    def group_input(self, x_dict, node_type, local_node_idx, n_id=None, perturb=None):
        # Create global node feature matrix.
        if n_id is not None:
            node_type = node_type[n_id]
            local_node_idx = local_node_idx[n_id]

        h = torch.zeros((node_type.size(0), self.in_channels),
                        device=node_type.device)

        for key, x in x_dict.items():
            mask = node_type == int(key)
            h[mask] = x[local_node_idx[mask]] if perturb is None else x[local_node_idx[mask]] + perturb

        for key, emb in self.emb_dict.items():
            mask = node_type == int(key)
            h[mask] = emb[local_node_idx[mask]]

        return h

    def forward(self, n_id, x_dict, adjs, edge_type, node_type,
                local_node_idx, perturb=None):

        x = self.group_input(x_dict, node_type, local_node_idx, n_id, perturb)

        if self.args.FDFT:
            x = F.dropout(x, p=0.5, training=self.training)

        if self.args.Norm4:
            x = self.norm(x)

        node_type = node_type[n_id]

        for i, (edge_index, e_id, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target node embeddings.
            src_node_type = node_type
            node_type = node_type[:size[1]]  # Target node types.

            conv = self.convs[i]
            x = conv((x, x_target), edge_index, edge_type[e_id], node_type, src_node_type)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x.log_softmax(dim=-1)



