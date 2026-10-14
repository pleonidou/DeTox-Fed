"""Per-client feature selection and standardisation."""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# Layout of the conversation feature vector produced by the feature pipeline:
#   [ deepwalk | author | sentiment | conversation_length | num_users | reaction_time ]
FEATURE_BLOCKS = {
    "deepwalk": 384,
    "author": 9,
    "sentiment": 5,
    "conversation_length": 1,
    "num_users": 1,
    "reaction_time": 1,
}

# Order in which selected blocks are concatenated after scaling.
OUTPUT_ORDER = ("deepwalk", "author", "sentiment", "reaction_time", "num_users", "conversation_length")


def _block_slices(dims):
    slices, start = {}, 0
    for name, dim in dims.items():
        slices[name] = slice(start, start + dim)
        start += dim
    return slices, start


def scale_features_per_client(graphs, selected_blocks, dims=None, log1p_author=True):
    """
    Keep the selected feature blocks and standardise each block per client.

    Scalers are fitted on the client's training nodes only, so no statistics
    leave the client and no test information leaks into scaling. The
    conversation length, number of users and reaction time are already
    log1p-transformed by the feature pipeline; author counts are log1p-
    transformed here. Sentiment is not log-transformed since it can be negative.

    Returns ``(graphs, scalers_by_client)``; ``graphs`` are modified in place.
    """
    dims = dims or FEATURE_BLOCKS
    selected = [b for b in OUTPUT_ORDER if b in set(selected_blocks)]
    if not selected:
        raise ValueError("No feature blocks selected.")

    slices, expected_dim = _block_slices(dims)
    scalers_by_client = {}

    for cid, g in enumerate(graphs):
        x = g.x.detach().cpu().numpy()
        train_mask = g.train_mask.detach().cpu().numpy().astype(bool)

        if train_mask.sum() == 0:
            raise RuntimeError(f"Client {cid} has no training nodes to fit the scalers on.")
        if x.shape[1] != expected_dim:
            raise RuntimeError(f"Client {cid}: expected feature dim {expected_dim}, got {x.shape[1]}.")

        parts, scalers = [], {}
        for block in selected:
            block_all = x[:, slices[block]]
            block_train = block_all[train_mask]
            if block == "author" and log1p_author:
                block_all = np.log1p(np.maximum(block_all, 0))
                block_train = np.log1p(np.maximum(block_train, 0))

            scaler = StandardScaler().fit(block_train)
            scalers[block] = scaler
            parts.append(scaler.transform(block_all))

        g.x = torch.from_numpy(np.concatenate(parts, axis=1).astype(np.float32))
        scalers_by_client[cid] = scalers

    return graphs, scalers_by_client
