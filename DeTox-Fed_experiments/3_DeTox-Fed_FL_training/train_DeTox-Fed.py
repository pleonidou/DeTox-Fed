"""
Train and evaluate DeTox-Fed: federated GNN toxicity detection where every
Fediverse instance is one FL client.

Example
-------
    python DeTox-Fed_experiments/3_DeTox-Fed_FL_training/train_DeTox-Fed.py \
        @DeTox-Fed_experiments/3_DeTox-Fed_FL_training/args_for_training/pleroma.args \
        --use_deepwalk --use_author \
        --use_sentiment --use_conv_length --use_num_user --use_reaction_time
"""

import argparse
import csv
import importlib
import os
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from DeTox_Fed_FL_components.aggregator import DeToxFedAggregator
from DeTox_Fed_FL_components.data_loader import RULE_LABEL_METHODS, build_fedml_dataset, build_federated_loaders
from DeTox_Fed_FL_components.features import scale_features_per_client
from DeTox_Fed_FL_components.metrics import METRIC_COLUMNS, pool_client_metrics
from DeTox_Fed_FL_components.models import MODELS, build_model
from DeTox_Fed_FL_components.trainer import Trainer

FEATURE_FLAGS = {
    "use_deepwalk": "deepwalk",
    "use_author": "author",
    "use_sentiment": "sentiment",
    "use_conv_length": "conversation_length",
    "use_num_user": "num_users",
    "use_reaction_time": "reaction_time",
}

REQUIRED_EDGE_COLUMNS = {"instance_toot_found_id", "source", "target", "score"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars="@",
    )

    data = p.add_argument_group("data")
    data.add_argument("--nodes_csv", required=True, help="Per-conversation toxicity aggregates (one row per conversation).")
    data.add_argument("--edges", required=True, help="Backbone intra-instance edges (.pkl DataFrame).")
    data.add_argument("--features_npy", required=True, help="Conversation feature matrix (.npy).")
    data.add_argument("--features_index", required=True, help="Row index of the feature matrix (.parquet).")

    labels = p.add_argument_group("labels")
    labels.add_argument(
        "--conversation_toxicity_method",
        default="toot_or_(num_replies_and_frac_replies)",
        help="Rule-based method (%s), or a label column name when using --labels_csv." % ", ".join(RULE_LABEL_METHODS),
    )
    labels.add_argument("--toxicity_binary_threshold", type=float, default=0.5)
    labels.add_argument("--conversation_level_toxic_threshold_fraction", type=float, default=0.5)
    labels.add_argument("--conversation_level_toxic_threshold_num", type=int, default=2)
    labels.add_argument(
        "--labels_csv",
        default=None,
        help="Optional CSV with external 0/1 conversation labels in column --conversation_toxicity_method.",
    )
    labels.add_argument(
        "--holdout_csv",
        default=None,
        help="Optional CSV of separately labelled conversations, excluded from training and used for final evaluation.",
    )
    labels.add_argument("--holdout_label_col", default="llm_label")
    labels.add_argument(
        "--max_labels_per_client",
        type=int,
        default=None,
        help="Max labelled training nodes per client, balanced between classes. Default: use all.",
    )

    feats = p.add_argument_group("feature blocks (select at least one)")
    for flag in FEATURE_FLAGS:
        feats.add_argument(f"--{flag}", action="store_true")

    fl = p.add_argument_group("model and federated training")
    fl.add_argument("--model", default="sage", choices=MODELS)
    fl.add_argument("--hidden_dim", type=int, default=128)
    fl.add_argument("--dropout", type=float, default=0.2)
    fl.add_argument("--heads", type=int, default=4, help="Attention heads (GAT only).")
    fl.add_argument("--rounds", type=int, default=20)
    fl.add_argument("--client_num_per_round", type=int, default=5)
    fl.add_argument("--local_epochs", type=int, default=1)
    fl.add_argument("--batch_size", type=int, default=1)
    fl.add_argument("--learning_rate", type=float, default=1e-3)
    fl.add_argument("--frequency_of_the_test", type=int, default=10)
    fl.add_argument("--random_seed", type=int, default=50)

    out = p.add_argument_group("output")
    out.add_argument("--log_folder", required=True)
    out.add_argument("--results_csv", default=None, help="Summary CSV to append to. Default: <log_folder>/results_summary.csv")
    out.add_argument("--tag", default="", help="Free-text label stored with each results row.")

    args = p.parse_args(argv)

    if not any(getattr(args, f) for f in FEATURE_FLAGS):
        p.error("select at least one feature block: " + ", ".join(f"--{f}" for f in FEATURE_FLAGS))

    return args


def _disable_fedml_mlops():
    try:
        import fedml.mlops as mlops

        for name in (
            "log_aggregation_status",
            "log_training_status",
            "log_client_model_info",
            "log_server_model_info",
            "log_round_info",
            "log_training_finished_status",
            "log_aggregation_finished_status",
            "event",
        ):
            setattr(mlops, name, lambda *a, **k: None)
    except Exception as e:
        print(f"[fedml] Could not disable MLOps logging: {e}", flush=True)


def _import_runner():
    try:
        from fedml import FedMLRunner
    except ImportError:
        from fedml.api import FedMLRunner
    return FedMLRunner


def patch_fedml():
    """Make FedML's single-process FedAvg use the GNN trainer and aggregator."""
    _disable_fedml_mlops()

    def create_trainer(model, args):
        return Trainer(model, args)

    def create_aggregator(model, args):
        return DeToxFedAggregator(model, args)

    importlib.import_module("fedml.ml.trainer.trainer_creator").create_model_trainer = create_trainer
    fedavg = importlib.import_module("fedml.simulation.sp.fedavg.fedavg_api")
    fedavg.create_model_trainer = create_trainer

    try:
        importlib.import_module("fedml.ml.aggregator.aggregator_creator").create_model_aggregator = create_aggregator
        if hasattr(fedavg, "create_model_aggregator"):
            fedavg.create_model_aggregator = create_aggregator
    except ImportError:
        pass

    def local_test_on_all_clients(self, round_idx):
        model = self.model_trainer.model
        model.load_state_dict(self.model_trainer.get_model_params())
        DeToxFedAggregator(model, self.args).test_all(
            self.train_data_local_dict,
            self.test_data_local_dict,
            self.device,
            self.args,
            round_idx=round_idx,
        )

    fedavg.FedAvgAPI._local_test_on_all_clients = local_test_on_all_clients


def load_edges(path):
    edges = pd.read_pickle(path)
    if not isinstance(edges, pd.DataFrame):
        raise SystemExit(f"{path} must contain a pandas DataFrame.")
    missing = REQUIRED_EDGE_COLUMNS - set(edges.columns)
    if missing:
        raise SystemExit(f"Edges file is missing columns: {sorted(missing)}")
    return edges


def append_csv(path, header, row):
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


def run_config_columns(args, fed_args):
    cols = {flag: int(getattr(args, flag)) for flag in FEATURE_FLAGS}
    cols.update({
        "tag": args.tag,
        "model": args.model,
        "rounds": fed_args.comm_round,
        "client_num_per_round": fed_args.client_num_per_round,
        "client_num_in_total": fed_args.client_num_in_total,
        "local_epochs": fed_args.epochs,
        "batch_size": fed_args.batch_size,
        "lr": fed_args.learning_rate,
        "random_seed": args.random_seed,
        "max_labels_per_client": args.max_labels_per_client,
        "toxicity_binary_threshold": args.toxicity_binary_threshold,
        "conv_level_toxicity_fraction": args.conversation_level_toxic_threshold_fraction,
        "conv_level_toxicity_num": args.conversation_level_toxic_threshold_num,
        "conversation_toxicity_method": args.conversation_toxicity_method,
    })
    return cols


def write_summary(path, config, timings, pooled):
    header = [*config, *timings, "test_total", "acc", "loss", *METRIC_COLUMNS, "tn", "fp", "fn", "tp"]
    values = {**config, **timings, **pooled}
    append_csv(path, header, [values[c] for c in header])


def main(argv=None):
    args = parse_args(argv)
    print(args, flush=True)

    log_folder = Path(args.log_folder).expanduser().resolve()
    log_folder.mkdir(parents=True, exist_ok=True)

    fed_args = types.SimpleNamespace(
        training_type="simulation",
        backend="sp",
        using_mlops=False,
        client_optimizer="sgd",
        data_cache_dir=str(Path(args.nodes_csv).expanduser().resolve().parent),
        client_num_in_total=0,
        client_num_per_round=args.client_num_per_round,
        comm_round=args.rounds,
        epochs=args.local_epochs,
        batch_size=args.batch_size,
        federated_optimizer="FedAvg",
        dataset="pleroma",
        learning_rate=args.learning_rate,
        frequency_of_the_test=args.frequency_of_the_test,
        is_mobile=False,
        enable_tensorboard=False,
        save_model=False,
        enable_wandb=False,
        using_wandb=False,
        log_round_metrics=False,
        log_folder=str(log_folder),
        using_gpu=torch.cuda.is_available(),
        rank=0,
    )

    nodes_df = pd.read_csv(Path(args.nodes_csv).expanduser())
    feats_index_df = pd.read_parquet(Path(args.features_index).expanduser())
    X = np.load(Path(args.features_npy).expanduser(), mmap_mode="r")
    edges_df = load_edges(Path(args.edges).expanduser())
    print(
        f"[data] conversations={len(nodes_df):,} features={X.shape} edges={len(edges_df):,}",
        flush=True,
    )

    dataset, maps = build_federated_loaders(
        nodes_df=nodes_df,
        edges_df=edges_df,
        feats_index_df=feats_index_df,
        X=X,
        label_col=args.conversation_toxicity_method,
        labels_csv=args.labels_csv,
        bin_threshold=args.toxicity_binary_threshold,
        conv_threshold_fraction=args.conversation_level_toxic_threshold_fraction,
        conv_threshold_num=args.conversation_level_toxic_threshold_num,
        holdout_csv=args.holdout_csv,
        holdout_label_col=args.holdout_label_col,
        max_labels_per_client=args.max_labels_per_client,
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        graph_nodes_csv=log_folder / "graph_nodes.csv",
    )

    train_global = dataset[3]
    graphs = list(train_global.dataset)
    selected = [block for flag, block in FEATURE_FLAGS.items() if getattr(args, flag)]
    graphs, _ = scale_features_per_client(graphs, selected)

    (
        client_num,
        train_total,
        test_total,
        train_global,
        test_global,
        train_local_num,
        train_local,
        test_local,
        class_num,
    ) = build_fedml_dataset(graphs, args.batch_size)

    fed_args.client_num_in_total = client_num
    fed_args.client_num_per_round = min(fed_args.client_num_per_round, max(1, client_num))
    fed_args.worker_num = fed_args.client_num_per_round

    patch_fedml()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, in_dim=int(graphs[0].x.shape[1]), out_dim=class_num or 2, args=args)

    fed_args.model_trainer = Trainer
    fed_args.model_aggregator = DeToxFedAggregator

    runner = _import_runner()(
        fed_args,
        device,
        (train_total, test_total, train_global, test_global, train_local_num, train_local, test_local, class_num),
        model,
    )

    start = time.time()
    runner.run()
    train_time = time.time() - start

    final_trainer = Trainer(model, fed_args)
    final_trainer.model.to(device)
    config = run_config_columns(args, fed_args)

    # Final evaluation on the automatically labelled test split.
    start = time.time()
    pooled = pool_client_metrics(final_trainer.test(loader, device, fed_args) for loader in test_local.values())
    test_time = time.time() - start

    if pooled is None:
        print("[test] No labelled test nodes.", flush=True)
    else:
        results_csv = (
            Path(args.results_csv).expanduser().resolve() if args.results_csv else log_folder / "results_summary.csv"
        )
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        write_summary(results_csv, config, {"train_time_sec": train_time, "test_time_sec": test_time}, pooled)
        print(
            f"[test] n={pooled['test_total']} acc={pooled['acc']:.4f} macro_f1={pooled['macro_f1']:.4f} "
            f"toxic_precision={pooled['positive_precision']:.4f} toxic_recall={pooled['positive_recall']:.4f}",
            flush=True,
        )
        print(f"[test] results appended to {results_csv}", flush=True)

    if args.holdout_csv is not None:
        evaluate_holdout(final_trainer, test_local, maps, device, fed_args, config, train_time, log_folder)


def evaluate_holdout(trainer, test_local, maps, device, fed_args, config, train_time, log_folder):
    """Evaluate on the hold-out conversations and save per-conversation predictions."""
    start = time.time()
    per_client, predictions = [], []

    for cid, loader in test_local.items():
        m = trainer.test_holdout(loader, device, fed_args)
        instance_id = maps["cid_to_instance"].get(cid, cid)
        for row in m.pop("predictions", []):
            predictions.append({"instance_id": instance_id, **row})
        per_client.append(m)
        if m["test_total"]:
            print(
                f"[holdout] instance={instance_id} n={m['test_total']} acc={m['test_acc']:.4f} "
                f"macro_f1={m['f1_macro']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f}",
                flush=True,
            )

    pooled = pool_client_metrics(per_client)
    holdout_time = time.time() - start

    predictions_csv = log_folder / "holdout_predictions.csv"
    pd.DataFrame(
        predictions,
        columns=["instance_id", "conversation_id", "true_label", "predicted_label", "predicted_prob_toxic"],
    ).to_csv(predictions_csv, index=False)
    print(f"[holdout] predictions saved to {predictions_csv}", flush=True)

    if pooled is None:
        print("[holdout] No hold-out conversations found in the client graphs.", flush=True)
        return

    print(
        f"[holdout] pooled n={pooled['test_total']} acc={pooled['acc']:.4f} "
        f"macro_f1={pooled['macro_f1']:.4f} weighted_f1={pooled['weighted_f1']:.4f} "
        f"toxic_precision={pooled['positive_precision']:.4f} toxic_recall={pooled['positive_recall']:.4f} "
        f"| tn={pooled['tn']} fp={pooled['fp']} fn={pooled['fn']} tp={pooled['tp']}",
        flush=True,
    )
    summary_csv = log_folder / "holdout_results_summary.csv"
    write_summary(summary_csv, config, {"train_time_sec": train_time, "test_time_sec": holdout_time}, pooled)
    print(f"[holdout] results appended to {summary_csv}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("FEDML_MLOPS_ENABLE", "0")
    os.environ.setdefault("FEDML_DISABLE_MLOPS", "1")
    main()
