"""
Federated data loading.

Each Fediverse instance is one FL client and holds one graph:

* nodes are the conversations observed on that instance;
* edges connect conversations of the same instance that share users
  (backbone edges produced by ``2_conversation_graph_backboning/noise_corrected_backboning.ipynb``);
* node features are the per-conversation vectors produced by the feature
  pipeline notebooks in ``1_creating_conversation_graph``.

Conversation labels come from one of two sources:

1. **Rule-based labels** (default) computed from the per-conversation toxicity
   aggregates in ``nodes_csv``, see ``RULE_LABEL_METHODS``.
2. **External binary labels** (``labels_csv``), e.g. labels produced by an LLM, with values 0 (non-toxic) and 1 (toxic).

Optionally, a separately labelled hold-out set (``holdout_csv``) is excluded
from training/validation and used only for the final evaluation.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.utils import to_undirected

MIN_NODES = 10
MIN_LABELED = 5
TEST_FRAC = 0.20
VAL_FRAC = 0.20

# Conversation is toxic if the root toot is toxic, or if enough replies are
# toxic (both a minimum count and a minimum fraction of toxic replies).
RULE_LABEL_METHODS = ("toot_or_(num_replies_and_frac_replies)", "toot_or_num_replies")


def _as_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _build_row_lookup(feats_index_df, instance_col, node_col) -> dict:
    """(instance, conversation) -> row index in the feature matrix X."""
    df = feats_index_df.reset_index(drop=True).copy()
    df[instance_col] = _as_str(df[instance_col])
    df[node_col] = _as_str(df[node_col])
    return dict(zip(zip(df[instance_col], df[node_col]), df.index))


def _build_label_table(nodes_df, instance_col, node_col) -> pd.DataFrame:
    df = nodes_df.copy()
    df[instance_col] = _as_str(df[instance_col])
    df[node_col] = _as_str(df[node_col])
    return df


def _rule_labels(merged, label_col, bin_threshold, conv_threshold_num, conv_threshold_fraction, client_id):
    suffix = str(bin_threshold).replace(".", "_")
    root_col = "root_toxicity"
    num_col = f"num_toxic_replies_{suffix}"
    frac_col = f"frac_toxic_replies_{suffix}"

    for c in (root_col, num_col, frac_col):
        if c not in merged.columns:
            raise ValueError(f"[loader] client={client_id}: missing column '{c}' for label method '{label_col}'.")

    root = pd.to_numeric(merged[root_col], errors="coerce")
    num_toxic = pd.to_numeric(merged[num_col], errors="coerce")
    frac_toxic = pd.to_numeric(merged[frac_col], errors="coerce")

    toxic = (root >= float(bin_threshold)) | (
        (num_toxic >= float(conv_threshold_num)) & (frac_toxic >= float(conv_threshold_fraction))
    )
    has_data = root.notna() | num_toxic.notna() | frac_toxic.notna()

    meta = {
        "type": "binary",
        "scheme": "toot_or_num_replies_hybrid",
        "threshold": float(bin_threshold),
        "min_toxic_count": int(conv_threshold_num),
        "min_toxic_fraction": float(conv_threshold_fraction),
    }
    return toxic, has_data, meta


def _compute_labels(
    node_keys,
    label_table,
    label_col,
    *,
    index_cols,
    client_id,
    binary_labels,
    bin_threshold,
    conv_threshold_num,
    conv_threshold_fraction,
    verbose,
):
    """
    Labels for the nodes of one client graph.

    Returns ``(y, label_meta, class_num, labeled_idx)`` where ``y`` is int64
    with -1 for unlabeled nodes.
    """
    n = len(node_keys)
    y = np.full(n, -1, dtype=np.int64)

    merged = pd.DataFrame(node_keys, columns=index_cols).merge(label_table, on=index_cols, how="left")

    if label_col in RULE_LABEL_METHODS:
        toxic, has_data, meta = _rule_labels(
            merged, label_col, bin_threshold, conv_threshold_num, conv_threshold_fraction, client_id
        )
        y[has_data.to_numpy()] = toxic[has_data].astype(np.int64).to_numpy()

    elif label_col in merged.columns:
        vals = pd.to_numeric(merged[label_col], errors="coerce")
        has_data = vals.notna()
        if binary_labels:
            labels = vals[has_data].astype(np.int64)
            if not labels.isin([0, 1]).all():
                raise ValueError(f"[loader] client={client_id}: '{label_col}' contains values other than 0/1.")
            y[has_data.to_numpy()] = labels.to_numpy()
            meta = {"type": "binary", "scheme": "external", "label_col": label_col}
        else:
            y[has_data.to_numpy()] = (vals[has_data] >= float(bin_threshold)).astype(np.int64).to_numpy()
            meta = {"type": "binary", "scheme": "threshold", "threshold": float(bin_threshold)}

    else:
        if verbose:
            print(f"[loader] client={client_id}: label column '{label_col}' not found; all nodes unlabeled.")
        meta = None

    labeled_idx = np.where(y >= 0)[0]
    class_num = 2 if len(np.unique(y[labeled_idx])) >= 2 else 0
    return y, meta, class_num, labeled_idx


def _read_binary_label_csv(path, node_col, label_col, what):
    """Read a CSV of conversation-level 0/1 labels; one label per conversation."""
    df = pd.read_csv(path)

    missing = {node_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"{what} CSV is missing required columns: {sorted(missing)}")

    df = df[[node_col, label_col]].copy()
    df[node_col] = _as_str(df[node_col])
    df[label_col] = pd.to_numeric(df[label_col], errors="raise").astype(int)

    if not df[label_col].isin([0, 1]).all():
        raise ValueError(f"{what} labels must be binary 0/1.")

    conflicts = df.groupby(node_col)[label_col].nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        raise ValueError(
            f"Conflicting {what} labels for conversation IDs: {conflicts.index.tolist()[:20]}"
        )

    return df.drop_duplicates(subset=[node_col])


def load_holdout_labels(holdout_csv, node_col="conversation_id", label_col="llm_label"):
    """Hold-out conversation labels as ``{conversation_id: 0/1}``."""
    if holdout_csv is None:
        return {}
    df = _read_binary_label_csv(holdout_csv, node_col, label_col, "Hold-out")
    return dict(zip(df[node_col], df[label_col]))


def attach_external_labels(nodes_df, labels_csv, node_col, label_col):
    """Merge external 0/1 conversation labels into ``nodes_df`` by conversation id."""
    labels = _read_binary_label_csv(labels_csv, node_col, label_col, "External")
    nodes_df = nodes_df.copy()
    nodes_df[node_col] = _as_str(nodes_df[node_col])
    if label_col in nodes_df.columns:
        nodes_df = nodes_df.drop(columns=[label_col])
    return nodes_df.merge(labels, on=node_col, how="left")


def _split_masks(y, labeled_idx, n, random_seed):
    """Stratified train / validation / test split of the labelled nodes."""
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    y_l = y[labeled_idx]
    classes, counts = np.unique(y_l, return_counts=True)
    n_test = int(max(1, round(labeled_idx.size * TEST_FRAC)))
    can_stratify = len(classes) >= 2 and counts.min() >= 2 and n_test >= len(classes)

    trainval_idx, test_idx = train_test_split(
        labeled_idx,
        test_size=TEST_FRAC,
        random_state=random_seed,
        stratify=y_l if can_stratify else None,
    )

    y_tv = y[trainval_idx]
    classes_tv, counts_tv = np.unique(y_tv, return_counts=True)
    n_val = int(max(1, round(trainval_idx.size * VAL_FRAC)))
    can_stratify_tv = len(classes_tv) >= 2 and counts_tv.min() >= 2 and n_val >= len(classes_tv)

    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=VAL_FRAC,
        random_state=random_seed,
        stratify=y_tv if can_stratify_tv else None,
    )

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask, train_idx, val_idx, test_idx


def _subsample_train_labels(y, train_idx, max_labels_per_client, random_seed):
    """
    Keep at most ``max_labels_per_client`` labelled training nodes, balanced
    between classes where possible. Dropped nodes are set to -1.
    """
    y_train = y.copy()
    if max_labels_per_client is None or train_idx.size <= max_labels_per_client:
        return y_train

    rng = np.random.default_rng(random_seed)
    pos_idx = train_idx[y[train_idx] == 1]
    neg_idx = train_idx[y[train_idx] == 0]

    n_pos = min(max_labels_per_client // 2, len(pos_idx))
    n_neg = min(max_labels_per_client - n_pos, len(neg_idx))
    n_pos = min(max_labels_per_client - n_neg, len(pos_idx))

    keep_pos = rng.choice(pos_idx, size=n_pos, replace=False) if n_pos > 0 else np.array([], dtype=np.int64)
    keep_neg = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg > 0 else np.array([], dtype=np.int64)
    drop_idx = np.setdiff1d(train_idx, np.concatenate([keep_pos, keep_neg]))
    y_train[drop_idx] = -1
    return y_train


def build_fedml_dataset(graphs, batch_size=1):
    """Assemble the tuple FedML expects from a list of per-client graphs."""
    client_num = len(graphs)
    train_local = {c: GeoDataLoader([g], batch_size=1, shuffle=False) for c, g in enumerate(graphs)}
    test_local = {c: GeoDataLoader([g], batch_size=1, shuffle=False) for c, g in enumerate(graphs)}
    train_local_num = {c: int(g.x.size(0)) for c, g in enumerate(graphs)}
    train_global = GeoDataLoader(graphs, batch_size=batch_size or 1, shuffle=False)
    test_global = GeoDataLoader(graphs, batch_size=batch_size or 1, shuffle=False)

    train_total = int(sum(train_local_num.values()))
    class_num = int(max((getattr(g, "num_classes", 0) or 0) for g in graphs)) if graphs else 0

    return (
        client_num,
        train_total,
        train_total,
        train_global,
        test_global,
        train_local_num,
        train_local,
        test_local,
        class_num,
    )


def build_federated_loaders(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    feats_index_df: pd.DataFrame,
    X: np.ndarray,
    *,
    instance_col: str = "instance_toot_found_id",
    node_col: str = "conversation_id",
    edge_src_col: str = "source",
    edge_dst_col: str = "target",
    edge_weight_col: str = "score",
    label_col: str = "toot_or_(num_replies_and_frac_replies)",
    labels_csv: str = None,
    bin_threshold: float = 0.5,
    conv_threshold_fraction: float = 0.5,
    conv_threshold_num: int = 2,
    holdout_csv: str = None,
    holdout_label_col: str = "llm_label",
    max_labels_per_client: int = None,
    random_seed: int = 50,
    require_edges: bool = True,
    batch_size: int = 1,
    graph_nodes_csv: str = None,
    verbose: bool = True,
):
    """
    Build one PyG graph per instance from intra-instance edges.

    Returns ``(fedml_dataset, mappings)``; see ``build_fedml_dataset`` for the
    dataset tuple. ``mappings`` holds ``cid_to_instance``, ``instance_to_cid``,
    ``node_ids_by_cid``, ``idx_map_by_cid`` and ``label_meta_by_cid``.
    """
    nodes_df = nodes_df.copy()
    edges_df = edges_df.copy()
    feats_index_df = feats_index_df.copy()

    for df, cols in [
        (nodes_df, [instance_col, node_col]),
        (edges_df, [instance_col, edge_src_col, edge_dst_col]),
        (feats_index_df, [instance_col, node_col]),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = _as_str(df[c])

    binary_labels = labels_csv is not None
    if binary_labels:
        nodes_df = attach_external_labels(nodes_df, labels_csv, node_col, label_col)
        if verbose:
            print(f"[loader] {nodes_df[label_col].notna().sum():,} conversations with external labels.", flush=True)

    holdout_labels = load_holdout_labels(holdout_csv, node_col=node_col, label_col=holdout_label_col)
    holdout_ids = set(holdout_labels)
    if verbose and holdout_csv is not None:
        print(f"[loader] {len(holdout_ids):,} hold-out conversations.", flush=True)

    row_lookup = _build_row_lookup(feats_index_df, instance_col, node_col)
    label_table = _build_label_table(nodes_df, instance_col, node_col)
    index_cols = [instance_col, node_col]

    edges_by_inst = dict(tuple(edges_df.groupby(instance_col)))
    feats_by_inst = dict(tuple(feats_index_df.groupby(instance_col)))
    instances = sorted(set(nodes_df[instance_col]) | set(edges_by_inst) | set(feats_by_inst))
    if verbose:
        print(f"[loader] {len(instances)} instances.", flush=True)

    graphs, class_nums = [], []
    cid_to_instance, instance_to_cid = {}, {}
    node_ids_by_cid, idx_map_by_cid, label_meta_by_cid = {}, {}, {}
    graph_node_records = []

    for inst in instances:
        e_sub = edges_by_inst.get(inst)
        f_sub = feats_by_inst.get(inst)

        if f_sub is None or f_sub.empty:
            if verbose:
                print(f"[loader] {inst}: no features; skipped")
            continue

        if e_sub is None or e_sub.empty:
            if require_edges:
                if verbose:
                    print(f"[loader] {inst}: no edges; skipped")
                continue
            e_sub = pd.DataFrame(columns=[edge_src_col, edge_dst_col, edge_weight_col])

        feat_nodes = set(f_sub[node_col])
        e_sub = e_sub[e_sub[edge_src_col].isin(feat_nodes) & e_sub[edge_dst_col].isin(feat_nodes)]
        if e_sub.empty and require_edges:
            if verbose:
                print(f"[loader] {inst}: no edges between featurised conversations; skipped")
            continue

        node_ids = sorted(set(e_sub[edge_src_col]) | set(e_sub[edge_dst_col]))
        n = len(node_ids)
        if n < MIN_NODES:
            if verbose:
                print(f"[loader] {inst}: only {n} nodes; skipped")
            continue

        id2local = {nid: i for i, nid in enumerate(node_ids)}
        rows = np.array([row_lookup[(inst, nid)] for nid in node_ids], dtype=np.int64)
        x_inst = np.asarray(X[rows], dtype=np.float32, order="C")

        y_auto, label_meta, _, _ = _compute_labels(
            [(inst, nid) for nid in node_ids],
            label_table,
            label_col,
            index_cols=index_cols,
            client_id=inst,
            binary_labels=binary_labels,
            bin_threshold=bin_threshold,
            conv_threshold_num=conv_threshold_num,
            conv_threshold_fraction=conv_threshold_fraction,
            verbose=verbose,
        )

        # Hold-out conversations stay in the graph but are unlabeled for
        # training, validation and the automatic test split.
        holdout_mask = np.array([nid in holdout_ids for nid in node_ids], dtype=bool)
        y_holdout = np.full(n, -1, dtype=np.int64)
        for i in np.where(holdout_mask)[0]:
            y_holdout[i] = holdout_labels[node_ids[i]]
        y_auto[holdout_mask] = -1

        labeled_idx = np.where(y_auto >= 0)[0]
        class_num = 2 if len(np.unique(y_auto[labeled_idx])) >= 2 else 0

        if labeled_idx.size < MIN_LABELED or class_num <= 1:
            if verbose:
                print(f"[loader] {inst}: labeled={labeled_idx.size}, classes={class_num}; skipped")
            continue

        if max_labels_per_client is not None and labeled_idx.size < max_labels_per_client:
            if verbose:
                print(
                    f"[loader] {inst}: labeled={labeled_idx.size} "
                    f"< max_labels_per_client={max_labels_per_client}; skipped"
                )
            continue

        src = e_sub[edge_src_col].map(id2local).to_numpy(dtype=np.int64)
        dst = e_sub[edge_dst_col].map(id2local).to_numpy(dtype=np.int64)
        edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

        w = pd.to_numeric(e_sub[edge_weight_col], errors="coerce").to_numpy(dtype=np.float32)
        finite = np.isfinite(w)
        if not finite.all():
            edge_index = edge_index[:, finite]
            w = w[finite]

        edge_index, edge_weight = to_undirected(
            edge_index, edge_attr=torch.from_numpy(w), num_nodes=n, reduce="mean"
        )
        edge_weight = edge_weight.view(-1)

        train_mask, val_mask, auto_test_mask, train_idx, _, _ = _split_masks(
            y_auto, labeled_idx, n, random_seed
        )
        assert not np.any((train_mask | val_mask | auto_test_mask) & holdout_mask), (
            f"Hold-out conversations leaked into train/val/test for instance {inst}"
        )

        y_train = _subsample_train_labels(y_auto, train_idx, max_labels_per_client, random_seed)
        train_mask &= y_train >= 0

        if train_mask.sum() < MIN_LABELED:
            if verbose:
                print(f"[loader] {inst}: labeled_train={int(train_mask.sum())} after subsampling; skipped")
            continue

        data = Data(
            x=torch.from_numpy(x_inst),
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=torch.from_numpy(y_train),
        )
        data.y_auto = torch.from_numpy(y_auto)
        data.y_holdout = torch.from_numpy(y_holdout)
        data.train_mask = torch.from_numpy(train_mask)
        data.val_mask = torch.from_numpy(val_mask)
        data.auto_test_mask = torch.from_numpy(auto_test_mask)
        data.holdout_mask = torch.from_numpy(holdout_mask)
        data.num_classes = class_num
        data.instance_id = inst
        data.node_ids = node_ids
        data.label_meta = label_meta
        data.conversation_ids = torch.tensor([int(nid) for nid in node_ids], dtype=torch.long)

        cid = len(graphs)
        graphs.append(data)
        class_nums.append(class_num)
        cid_to_instance[cid] = inst
        instance_to_cid[inst] = cid
        node_ids_by_cid[cid] = node_ids
        idx_map_by_cid[cid] = id2local
        label_meta_by_cid[cid] = label_meta
        graph_node_records.extend(
            {"cid": cid, instance_col: inst, node_col: nid} for nid in node_ids
        )

        if verbose:
            print(
                f"[loader] {inst}: nodes={n} edges={edge_index.size(1)} labeled={labeled_idx.size} "
                f"holdout={int(holdout_mask.sum())} "
                f"w=[{float(edge_weight.min()):.3f}, {float(edge_weight.max()):.3f}]",
                flush=True,
            )

    if not graphs:
        raise RuntimeError("No usable client graphs. Check ID alignment and label thresholds.")

    if verbose:
        print(f"[loader] {len(graphs)} client graphs.", flush=True)

    if graph_nodes_csv is not None:
        pd.DataFrame(graph_node_records).to_csv(graph_nodes_csv, index=False)

    mappings = {
        "cid_to_instance": cid_to_instance,
        "instance_to_cid": instance_to_cid,
        "node_ids_by_cid": node_ids_by_cid,
        "idx_map_by_cid": idx_map_by_cid,
        "label_meta_by_cid": label_meta_by_cid,
    }
    return build_fedml_dataset(graphs, batch_size), mappings
