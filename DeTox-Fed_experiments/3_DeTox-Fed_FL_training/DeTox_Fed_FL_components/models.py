"""Node-classification models used by DeTox-Fed."""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GraphSAGE(torch.nn.Module):
    """Two-layer GraphSAGE."""

    supports_edge_weight = False

    def __init__(self, in_dim, hid=128, out_dim=2, drop=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid)
        self.conv2 = SAGEConv(hid, out_dim)
        self.drop = drop

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        return self.conv2(x, edge_index)


class GAT(torch.nn.Module):
    """Two-layer graph attention network."""

    supports_edge_weight = False

    def __init__(self, in_dim, hid=128, out_dim=2, heads=4, drop=0.2):
        super().__init__()
        self.conv1 = GATConv(in_dim, hid, heads=heads, dropout=drop)
        self.conv2 = GATConv(hid * heads, out_dim, heads=1, concat=False, dropout=drop)
        self.drop = drop

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        return self.conv2(x, edge_index)


class GCN(torch.nn.Module):
    """Two-layer graph convolutional network (unweighted edges)."""

    supports_edge_weight = False

    def __init__(self, in_dim, hid=128, out_dim=2, drop=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, out_dim)
        self.drop = drop

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        return self.conv2(x, edge_index)


class WeightedGCN(torch.nn.Module):
    """Two-layer GCN that uses the backbone edge scores as edge weights."""

    supports_edge_weight = True

    def __init__(self, in_dim, hid=128, out_dim=2, drop=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, out_dim)
        self.drop = drop

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.drop, training=self.training)
        return self.conv2(x, edge_index, edge_weight=edge_weight)


MODELS = ("sage", "gat", "gcn", "weighted_gcn")


def build_model(model_name, in_dim, out_dim=2, args=None):
    """
    Build a model by name.

    Parameters
    ----------
    model_name : one of MODELS
    in_dim : input feature dimension
    out_dim : number of classes
    args : optional namespace with ``hidden_dim``, ``dropout`` and ``heads``
    """
    hid = int(getattr(args, "hidden_dim", 128))
    drop = float(getattr(args, "dropout", 0.2))
    heads = int(getattr(args, "heads", 4))

    name = model_name.lower()
    if name == "sage":
        return GraphSAGE(in_dim, hid=hid, out_dim=out_dim, drop=drop)
    if name == "gat":
        return GAT(in_dim, hid=hid, out_dim=out_dim, heads=heads, drop=drop)
    if name == "gcn":
        return GCN(in_dim, hid=hid, out_dim=out_dim, drop=drop)
    if name == "weighted_gcn":
        return WeightedGCN(in_dim, hid=hid, out_dim=out_dim, drop=drop)

    raise ValueError(f"Unknown model '{model_name}'. Choose from: {', '.join(MODELS)}.")
