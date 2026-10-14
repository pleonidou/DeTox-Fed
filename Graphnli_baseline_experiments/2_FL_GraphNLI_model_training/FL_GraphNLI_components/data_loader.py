"""
Federated data loading for GraphNLI.

Input: the random-walk CSV produced by ``1_create_conversation_graph/02_random_walks.ipynb``,
one row per toot with columns ``instance_toot_found_id, conversation_id,
toot_id, sent1..sent4, toxicity``.

* One Fediverse instance is one FL client.
* Per instance, conversations with at least ``min_conversation_length`` toots
  are eligible; instances need at least ``min_conversations`` of them.
* Per instance, up to ``n_toxic_conversations`` conversations containing a
  toxic toot and ``n_non_toxic_conversations`` without one are sampled.
* The same conversation can be observed on several instances, so
  ``conversation_id`` is split into train / validation / test **once,
  globally**. Every observation of a conversation lands in the same split on
  every client, which prevents leakage between splits across clients.
* With a hold-out CSV, the listed ``(instance, conversation)`` pairs are the
  test set instead of a random split, and their conversations are excluded
  from training and validation on every client.
* The same toot ID can occur in more than one conversation, so toots of test
  conversations are also removed from training and validation, and toots of
  validation conversations from training, on every client.

During FL rounds, FedML's "test" loaders are the validation loaders; the
held-out test loaders are returned separately and used only after training.
"""

import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

INSTANCE_COL = "instance_toot_found_id"
CONVERSATION_COL = "conversation_id"
LOCAL_CONVERSATION_KEY = [INSTANCE_COL, CONVERSATION_COL]
REQUIRED_COLUMNS = [INSTANCE_COL, CONVERSATION_COL, "toot_id", "sent1", "sent2", "sent3", "sent4", "toxicity"]


class GraphNLIDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "sent1": str(row["sent1"]),
            "sent2": str(row["sent2"]),
            "sent3": str(row["sent3"]),
            "sent4": str(row["sent4"]),
            "toxicity": float(row["toxicity"]),
            "instance_id": row[INSTANCE_COL],
            "conversation_id": row[CONVERSATION_COL],
            "toot_id": row["toot_id"],
        }


def graphnli_collate(batch):
    out = {key: [x[key] for x in batch] for key in ("sent1", "sent2", "sent3", "sent4", "instance_id", "conversation_id", "toot_id")}
    out["toxicity"] = torch.tensor([x["toxicity"] for x in batch], dtype=torch.float32)
    return out


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(df, batch_size, shuffle):
    return DataLoader(
        GraphNLIDataset(df),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=graphnli_collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _split_conversations(table, test_size, random_state):
    """Split a one-row-per-conversation table, stratified on ``has_toxic_toot`` when possible."""
    if len(table) < 2:
        raise ValueError("Need at least 2 conversations to split.")
    if table[CONVERSATION_COL].duplicated().any():
        raise RuntimeError("Conversation table contains duplicate conversation_id values.")

    counts = table["has_toxic_toot"].value_counts()
    stratify = table["has_toxic_toot"] if len(counts) == 2 and counts.min() >= 2 else None

    first, second = train_test_split(
        table, test_size=test_size, random_state=random_state, shuffle=True, stratify=stratify
    )
    return first.reset_index(drop=True), second.reset_index(drop=True)


def _ids(df, column):
    return set(df[column].drop_duplicates().tolist())


def _count_cross_client_duplicates(client_sets):
    counts = Counter()
    for values in client_sets.values():
        counts.update(values)
    return sum(1 for c in counts.values() if c > 1)


def _assert_client_disjoint(client_sets, what):
    owners = {}
    for cid, values in client_sets.items():
        for value in values:
            previous = owners.setdefault(value, cid)
            if previous != cid:
                raise RuntimeError(f"{what}={value!r} appears in FL clients {previous} and {cid}.")


def _assert_disjoint(train, validation, test, what, where="globally"):
    for name, overlap in [
        ("train/validation", train & validation),
        ("train/test", train & test),
        ("validation/test", validation & test),
    ]:
        if overlap:
            raise RuntimeError(f"{what} overlap between {name} {where}: {len(overlap)} IDs.")


def _sample_instance(instance_df, instance_id, *, exclude_ids, min_conversations, n_toxic, n_non_toxic, random_seed, verbose):
    """Balanced sample of conversations for one instance, or ``None`` if the instance is not eligible."""
    info = (
        instance_df.groupby(LOCAL_CONVERSATION_KEY)
        .agg(
            num_toots=("toot_id", "nunique"),
            num_toxic_toots=("is_toxic", "sum"),
            has_toxic_toot=("is_toxic", "any"),
        )
        .reset_index()
    )
    if exclude_ids:
        info = info[~info[CONVERSATION_COL].isin(exclude_ids)].copy()

    if len(info) < int(min_conversations):
        if verbose:
            print(f"[loader] skip instance={instance_id}: {len(info)} conversations < {min_conversations}", flush=True)
        return None
    if instance_df["is_toxic"].nunique() < 2:
        if verbose:
            print(f"[loader] skip instance={instance_id}: only one toot class", flush=True)
        return None

    toxic = info[info["has_toxic_toot"]]
    non_toxic = info[~info["has_toxic_toot"]]
    if toxic.empty or non_toxic.empty:
        if verbose:
            print(f"[loader] skip instance={instance_id}: missing one conversation class", flush=True)
        return None

    # Instance-specific seed so that instances with identical candidates do
    # not select identical rows only because they share the random state.
    seed = int(random_seed) + sum(ord(ch) for ch in str(instance_id))

    selected = pd.concat(
        [
            toxic.sample(n=min(int(n_toxic), len(toxic)), random_state=seed),
            non_toxic.sample(n=min(int(n_non_toxic), len(non_toxic)), random_state=seed),
        ],
        ignore_index=True,
    )
    return (
        selected.drop_duplicates(subset=[CONVERSATION_COL])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def _read_holdout_pairs(holdout_csv, instance_col, conversation_col, df, verbose):
    """(instance, conversation) pairs of the hold-out set; all must exist in the filtered data."""
    holdout = pd.read_csv(holdout_csv, low_memory=False)
    missing = {instance_col, conversation_col} - set(holdout.columns)
    if missing:
        raise ValueError(f"Hold-out CSV is missing columns: {sorted(missing)}")

    holdout = holdout[[instance_col, conversation_col]].dropna().drop_duplicates().astype(str)
    pairs = set(zip(holdout[instance_col], holdout[conversation_col]))
    if not pairs:
        raise ValueError("The hold-out CSV contains no (instance, conversation) pairs.")

    unavailable = pairs - set(zip(df[INSTANCE_COL], df[CONVERSATION_COL]))
    if unavailable:
        raise RuntimeError(
            f"{len(unavailable)} hold-out (instance, conversation) pairs are not in the data after "
            f"the conversation-length filter. Examples: {sorted(unavailable)[:10]}"
        )

    if verbose:
        print(f"[loader] hold-out: {len(pairs):,} (instance, conversation) pairs, "
              f"{len({c for _, c in pairs}):,} conversations", flush=True)
    return pairs


def build_federated_graphnli_loaders(
    csv_path: str,
    *,
    batch_size: int = 16,
    toxicity_threshold: float = 0.5,
    min_conversation_length: int = 10,
    min_conversations: int = 500,
    n_toxic_conversations: int = 250,
    n_non_toxic_conversations: int = 250,
    test_fraction: float = 0.20,
    validation_fraction_of_remaining: float = 0.20,
    random_seed: int = 4000,
    cross_client_deduplicate: bool = False,
    holdout_csv: str = None,
    holdout_instance_col: str = "instance_toot_found_id",
    holdout_conversation_col: str = "conversation_id",
    shuffle_train_labels: bool = False,
    verbose: bool = True,
):
    """
    Build FedML loaders for GraphNLI.

    ``holdout_csv``: CSV of ``(instance, conversation)`` pairs, e.g. the
    manually annotated conversations, used as the test set. Only the pairs are
    read; the toot targets stay the toxicity scores of ``csv_path``. Every
    pair must reach the test loaders, otherwise an error is raised.

    ``cross_client_deduplicate``: keep each conversation on only one client
    (the instance with the smallest ID as string).

    ``shuffle_train_labels``: sanity check only. Permutes the toxicity scores
    of the training rows of every client, so the model should not learn;
    validation and test labels are untouched.

    Returns ``(fedml_dataset, metadata)``. The FedML tuple holds the
    validation loaders in its test slot; ``metadata`` holds the test loaders
    (``final_test_local``), the client mapping, a per-client split summary,
    the global split manifest and a leakage audit.
    """
    set_seed(random_seed)

    df = pd.read_csv(csv_path, low_memory=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for column in ("sent1", "sent2", "sent3", "sent4"):
        df[column] = df[column].fillna("").astype(str)
    df["toxicity"] = pd.to_numeric(df["toxicity"], errors="coerce")
    df = df.dropna(subset=[INSTANCE_COL, CONVERSATION_COL, "toot_id", "toxicity"]).copy()
    df[INSTANCE_COL] = df[INSTANCE_COL].astype(str)
    df[CONVERSATION_COL] = df[CONVERSATION_COL].astype(str)
    df["is_toxic"] = df["toxicity"] >= float(toxicity_threshold)

    # A conversation is eligible on an instance when that instance observes at
    # least min_conversation_length unique toots of it.
    lengths = df.groupby(LOCAL_CONVERSATION_KEY)["toot_id"].nunique().reset_index(name="num_toots")
    eligible = lengths.loc[lengths["num_toots"] >= int(min_conversation_length), LOCAL_CONVERSATION_KEY]
    df = df.merge(eligible, on=LOCAL_CONVERSATION_KEY, how="inner")

    if verbose:
        print(
            f"[loader] after length filter: rows={len(df):,} "
            f"conversations={df[CONVERSATION_COL].nunique():,} toots={df['toot_id'].nunique():,}",
            flush=True,
        )

    holdout_pairs, holdout_ids, holdout_toot_ids = set(), set(), set()
    if holdout_csv is not None:
        holdout_pairs = _read_holdout_pairs(holdout_csv, holdout_instance_col, holdout_conversation_col, df, verbose)
        holdout_ids = {c for _, c in holdout_pairs}
        in_holdout = pd.MultiIndex.from_frame(df[LOCAL_CONVERSATION_KEY]).isin(
            pd.MultiIndex.from_tuples(holdout_pairs, names=LOCAL_CONVERSATION_KEY)
        )
        holdout_toot_ids = set(df.loc[in_holdout, "toot_id"].dropna().drop_duplicates().tolist())

    # Per-instance balanced sampling (hold-out conversations excluded).
    selected_by_instance, instance_frames = {}, {}
    for instance_id in sorted(df[INSTANCE_COL].unique(), key=str):
        instance_df = df[df[INSTANCE_COL] == instance_id].copy()
        selected = _sample_instance(
            instance_df,
            instance_id,
            exclude_ids=holdout_ids,
            min_conversations=min_conversations,
            n_toxic=n_toxic_conversations,
            n_non_toxic=n_non_toxic_conversations,
            random_seed=random_seed,
            verbose=verbose,
        )
        if selected is not None:
            selected_by_instance[instance_id] = selected
            instance_frames[instance_id] = instance_df

    if not selected_by_instance:
        raise RuntimeError("No eligible instances.")

    observations = pd.concat(
        [
            s[[INSTANCE_COL, CONVERSATION_COL, "has_toxic_toot"]].assign(instance_id=iid)
            for iid, s in selected_by_instance.items()
        ],
        ignore_index=True,
    )

    if verbose:
        shared = int((observations.groupby(CONVERSATION_COL)["instance_id"].nunique() > 1).sum())
        print(f"[loader] selected conversations on more than one instance: {shared:,}", flush=True)

    if cross_client_deduplicate:
        observations = (
            observations.assign(_key=observations["instance_id"].astype(str))
            .sort_values([CONVERSATION_COL, "_key"], kind="mergesort")
            .drop_duplicates(subset=[CONVERSATION_COL], keep="first")
            .drop(columns=["_key"])
            .reset_index(drop=True)
        )

    owned_by_instance = {}
    for iid, selected in selected_by_instance.items():
        owned = set(observations.loc[observations["instance_id"] == iid, CONVERSATION_COL].tolist())
        owned_by_instance[iid] = selected[selected[CONVERSATION_COL].isin(owned)].copy()

    # One global split of conversation IDs.
    selected_ids = set(observations[CONVERSATION_COL].tolist())
    global_info = (
        df[df[CONVERSATION_COL].isin(selected_ids)]
        .groupby(CONVERSATION_COL)
        .agg(has_toxic_toot=("is_toxic", "any"), global_unique_toots=("toot_id", "nunique"))
        .reset_index()
    )
    if len(global_info) < 3:
        raise RuntimeError("Too few selected conversations for a train/validation/test split.")

    if holdout_csv is not None:
        train_global, val_global = _split_conversations(global_info, float(validation_fraction_of_remaining), int(random_seed))
        test_global = pd.DataFrame({CONVERSATION_COL: sorted(holdout_ids)})
        test_ids = set(holdout_ids)
    else:
        train_val, test_global = _split_conversations(global_info, float(test_fraction), int(random_seed))
        train_global, val_global = _split_conversations(train_val, float(validation_fraction_of_remaining), int(random_seed))
        test_ids = _ids(test_global, CONVERSATION_COL)

    train_ids = _ids(train_global, CONVERSATION_COL)
    val_ids = _ids(val_global, CONVERSATION_COL)
    _assert_disjoint(train_ids, val_ids, test_ids, "conversation_id")

    # Toots of validation conversations are removed from training everywhere.
    val_toot_ids = set(df.loc[df[CONVERSATION_COL].isin(val_ids), "toot_id"].dropna().drop_duplicates().tolist())

    split_manifest = pd.concat(
        [
            train_global.assign(split="train"),
            val_global.assign(split="validation"),
            test_global.assign(split="holdout" if holdout_csv is not None else "test"),
        ],
        ignore_index=True,
        sort=False,
    )
    if verbose:
        print(
            f"[loader] global split: train={len(train_ids):,} validation={len(val_ids):,} test={len(test_ids):,}",
            flush=True,
        )

    # Local loaders following the global split.
    train_local, validation_local, final_test_local, train_local_num = {}, {}, {}, {}
    train_dfs, validation_dfs, test_dfs = [], [], []
    cid_to_instance, instance_to_cid = {}, {}
    client_summary = []
    client_conversations, client_toots = {}, {}

    cid = 0
    for iid in sorted(owned_by_instance, key=str):
        selected = owned_by_instance[iid]
        if selected.empty:
            if verbose:
                print(f"[loader] skip instance={iid}: no conversations left after de-duplication", flush=True)
            continue

        ids = set(selected[CONVERSATION_COL].tolist())
        if holdout_csv is not None:
            client_test_ids = {c for i, c in holdout_pairs if i == str(iid)}
        else:
            client_test_ids = ids & test_ids

        instance_df = instance_frames[iid]
        train_df = instance_df[instance_df[CONVERSATION_COL].isin(ids & train_ids)].copy()
        val_df = instance_df[instance_df[CONVERSATION_COL].isin(ids & val_ids)].copy()
        test_df = instance_df[instance_df[CONVERSATION_COL].isin(client_test_ids)].copy()

        if shuffle_train_labels:
            train_df["toxicity"] = train_df["toxicity"].sample(frac=1.0, random_state=random_seed + cid).to_numpy()
            train_df["is_toxic"] = train_df["toxicity"] >= float(toxicity_threshold)

        # Toot-level cleanup, priority test > validation > train.
        n_train, n_val = len(train_df), len(val_df)
        if holdout_csv is not None:
            train_df = train_df[~train_df["toot_id"].isin(holdout_toot_ids)].copy()
            val_df = val_df[~val_df["toot_id"].isin(holdout_toot_ids)].copy()
        train_df = train_df[~train_df["toot_id"].isin(val_toot_ids)].copy()
        if verbose and (n_train > len(train_df) or n_val > len(val_df)):
            print(
                f"[loader] instance={iid}: removed {n_train - len(train_df):,} train and "
                f"{n_val - len(val_df):,} validation rows sharing toots with other splits",
                flush=True,
            )

        if train_df.empty or val_df.empty:
            if verbose:
                print(
                    f"[loader] skip instance={iid}: empty split (train={len(train_df):,}, validation={len(val_df):,})",
                    flush=True,
                )
            continue

        conv_sets = [_ids(d, CONVERSATION_COL) for d in (train_df, val_df, test_df)]
        toot_sets = [_ids(d, "toot_id") for d in (train_df, val_df, test_df)]
        _assert_disjoint(*conv_sets, "conversation_id", where=f"on instance {iid}")
        _assert_disjoint(*toot_sets, "toot_id", where=f"on instance {iid}")

        train_local[cid] = _make_loader(train_df, batch_size, shuffle=True)
        validation_local[cid] = _make_loader(val_df, batch_size, shuffle=False)
        if len(test_df) > 0:
            final_test_local[cid] = _make_loader(test_df, batch_size, shuffle=False)
            test_dfs.append(test_df)
        train_local_num[cid] = int(len(train_df))

        train_dfs.append(train_df)
        validation_dfs.append(val_df)
        cid_to_instance[cid] = iid
        instance_to_cid[iid] = cid
        client_conversations[cid] = set().union(*conv_sets)
        client_toots[cid] = set().union(*toot_sets)

        client_summary.append({
            "cid": cid,
            "instance_id": iid,
            "selected_conversations_before_cross_client_dedup": int(len(selected_by_instance[iid])),
            "selected_conversations_after_cross_client_dedup": int(len(selected)),
            "train_conversations": len(conv_sets[0]),
            "validation_conversations": len(conv_sets[1]),
            "test_conversations": len(conv_sets[2]),
            "train_toots": len(train_df),
            "validation_toots": len(val_df),
            "test_toots": len(test_df),
            "train_unique_toots": len(toot_sets[0]),
            "validation_unique_toots": len(toot_sets[1]),
            "test_unique_toots": len(toot_sets[2]),
            "train_toxic_toots": int(train_df["is_toxic"].sum()),
            "validation_toxic_toots": int(val_df["is_toxic"].sum()),
            "test_toxic_toots": int(test_df["is_toxic"].sum()),
        })

        if verbose:
            print(
                f"[loader] client={cid} instance={iid} | selected={len(selected)} "
                f"train={len(train_df):,} validation={len(val_df):,} test={len(test_df):,}",
                flush=True,
            )
        cid += 1

    if not train_local:
        raise RuntimeError("No eligible clients remained after the global split.")

    if holdout_csv is not None:
        reached = {
            (str(cid_to_instance[c]), conv)
            for c, loader in final_test_local.items()
            for conv in loader.dataset.df[CONVERSATION_COL].astype(str).drop_duplicates()
        }
        missing_pairs, extra_pairs = holdout_pairs - reached, reached - holdout_pairs
        if verbose:
            print(
                f"[loader] hold-out pairs: expected={len(holdout_pairs)} in test loaders={len(reached)} "
                f"missing={len(missing_pairs)} extra={len(extra_pairs)}",
                flush=True,
            )
        if missing_pairs:
            raise RuntimeError(
                f"{len(missing_pairs)} hold-out pairs did not reach the test loaders "
                f"(their instance is not an FL client). Examples: {sorted(missing_pairs)[:10]}"
            )
        if extra_pairs:
            raise RuntimeError(f"Test loaders contain {len(extra_pairs)} unexpected pairs. Examples: {sorted(extra_pairs)[:10]}")

    if not test_dfs:
        raise RuntimeError("No test rows were created.")

    # Global leakage audit across all clients.
    all_train = pd.concat(train_dfs, ignore_index=True)
    all_val = pd.concat(validation_dfs, ignore_index=True)
    all_test = pd.concat(test_dfs, ignore_index=True)
    conv_all = [_ids(d, CONVERSATION_COL) for d in (all_train, all_val, all_test)]
    toot_all = [_ids(d, "toot_id") for d in (all_train, all_val, all_test)]
    _assert_disjoint(*conv_all, "conversation_id")
    _assert_disjoint(*toot_all, "toot_id")

    duplicate_conversations = _count_cross_client_duplicates(client_conversations)
    duplicate_toots = _count_cross_client_duplicates(client_toots)
    if cross_client_deduplicate:
        _assert_client_disjoint(client_conversations, "conversation_id")
        _assert_client_disjoint(client_toots, "toot_id")

    if verbose:
        print(
            "[loader] leakage audit passed: train, validation and test are disjoint by conversation_id and toot_id. "
            f"Conversations on more than one client (same split): {duplicate_conversations}, "
            f"toots: {duplicate_toots}.",
            flush=True,
        )

    leakage_audit = pd.DataFrame([{
        "conversation_train_validation_overlap": len(conv_all[0] & conv_all[1]),
        "conversation_train_test_overlap": len(conv_all[0] & conv_all[2]),
        "conversation_validation_test_overlap": len(conv_all[1] & conv_all[2]),
        "toot_train_validation_overlap": len(toot_all[0] & toot_all[1]),
        "toot_train_test_overlap": len(toot_all[0] & toot_all[2]),
        "toot_validation_test_overlap": len(toot_all[1] & toot_all[2]),
        "cross_client_duplicate_conversations": int(duplicate_conversations),
        "cross_client_duplicate_toots": int(duplicate_toots),
        "cross_client_deduplicate": bool(cross_client_deduplicate),
        "holdout": holdout_csv is not None,
        "shuffle_train_labels": bool(shuffle_train_labels),
    }])

    train_total = int(sum(train_local_num.values()))
    validation_total = int(sum(len(loader.dataset) for loader in validation_local.values()))

    # Standard FedML tuple; validation sits in FedML's "test" slot.
    fedml_dataset = (
        train_total,
        validation_total,
        _make_loader(pd.concat(train_dfs, ignore_index=True), batch_size, shuffle=False),
        _make_loader(pd.concat(validation_dfs, ignore_index=True), batch_size, shuffle=False),
        train_local_num,
        train_local,
        validation_local,
        2,
    )

    metadata = {
        "client_num": len(train_local),
        "cid_to_instance": cid_to_instance,
        "instance_to_cid": instance_to_cid,
        "client_split_summary": pd.DataFrame(client_summary),
        "global_split_manifest": split_manifest,
        "leakage_audit": leakage_audit,
        "final_test_local": final_test_local,
        "final_test_global": _make_loader(pd.concat(test_dfs, ignore_index=True), batch_size, shuffle=False),
    }
    return fedml_dataset, metadata
