"""
Federated GraphNLI baseline: GraphNLI trained with FedAvg, one Fediverse
instance per FL client.

Example (from the repository root)
----------------------------------
    python Graphnli_baseline_experiments/2_FL_GraphNLI_model_training/train.py \
        @Graphnli_baseline_experiments/2_FL_GraphNLI_model_training/args_for_training/mastodon.args
"""

import argparse
import csv
import importlib
import os
import time
import types
from pathlib import Path

import numpy as np
import torch

# The transformer is loaded from a local directory (--model_path).
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from FL_GraphNLI_components.aggregator import GraphNLIAggregator
from FL_GraphNLI_components.data_loader import build_federated_graphnli_loaders, set_seed
from FL_GraphNLI_components.metrics import metrics_from_confusion
from FL_GraphNLI_components.model import build_model
from FL_GraphNLI_components.trainer import GraphNLITrainer


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars="@",
    )

    p.add_argument("--data_csv", required=True, help="Random-walk CSV from 1_create_conversation_graph/02_random_walks.ipynb.")
    p.add_argument("--model_path", required=True, help="Local directory of the pretrained transformer (e.g. distilroberta-base).")
    p.add_argument("--log_folder", required=True)

    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--client_num_per_round", type=int, default=5)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--frequency_of_the_test", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.2, help="Decay of the context-toot weights.")
    p.add_argument("--toxicity_threshold", type=float, default=0.5)

    p.add_argument("--min_conversation_length", type=int, default=10)
    p.add_argument("--min_conversations", type=int, default=500)
    p.add_argument("--n_toxic_conversations", type=int, default=250)
    p.add_argument("--n_non_toxic_conversations", type=int, default=250)
    p.add_argument("--test_fraction", type=float, default=0.20)
    p.add_argument("--validation_fraction_of_remaining", type=float, default=0.20)
    p.add_argument(
        "--holdout_csv",
        default=None,
        help="CSV of (instance_toot_found_id, conversation_id) pairs used as the test set instead of a random split.",
    )
    p.add_argument("--cross_client_deduplicate", action="store_true", help="Keep each conversation on one client only.")
    p.add_argument(
        "--shuffle_train_labels",
        action="store_true",
        help="Sanity check: permute training labels on every client (the model should not learn).",
    )
    p.add_argument("--random_seed", type=int, default=4000)

    return p.parse_args(argv)


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
    """Make FedML's single-process FedAvg use the GraphNLI trainer and aggregator."""
    _disable_fedml_mlops()

    def create_trainer(model, args):
        return GraphNLITrainer(model, args)

    def create_aggregator(model, args):
        return GraphNLIAggregator(model, args)

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
        GraphNLIAggregator(model, self.args).test_all(
            self.train_data_local_dict,
            self.test_data_local_dict,
            self.device,
            self.args,
            round_idx=round_idx,
        )

    fedavg.FedAvgAPI._local_test_on_all_clients = local_test_on_all_clients


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    fieldnames = fieldnames or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_final_test(model, final_test_local, cid_to_instance, device, fed_args, output_folder):
    """
    Evaluate the final global model on every client's held-out test set.

    Writes per-instance metrics and per-toot predictions, and returns pooled
    metrics and the unweighted mean over instances.
    """
    trainer = GraphNLITrainer(model, fed_args)
    per_instance, predictions = [], []
    tn = fp = fn = tp = 0
    total = correct = 0
    loss_sum = 0.0

    for cid, loader in final_test_local.items():
        m, rows = trainer.test(loader, device, fed_args, return_predictions=True)
        predictions.extend(rows)

        n = int(m.get("test_total") or 0)
        total += n
        correct += int(m.get("test_correct") or 0)
        if n:
            loss_sum += float(m.get("test_loss", 0.0)) * n
        tn += int(m["cm_00"])
        fp += int(m["cm_01"])
        fn += int(m["cm_10"])
        tp += int(m["cm_11"])

        per_instance.append({
            "cid": cid,
            "instance_id": cid_to_instance[cid],
            "test_total": n,
            "test_acc": m.get("test_acc"),
            "test_loss": m.get("test_loss"),
            "weighted_f1": m.get("f1_weighted"),
            "macro_f1": m.get("f1_macro"),
            "toxic_precision": m.get("precision"),
            "toxic_recall": m.get("recall"),
            "balanced_acc": m.get("balanced_acc"),
            "roc_auc": m.get("roc_auc"),
            "pr_auc": m.get("pr_auc"),
            "tn": m["cm_00"],
            "fp": m["cm_01"],
            "fn": m["cm_10"],
            "tp": m["cm_11"],
        })
        print(
            f"[test] instance={cid_to_instance[cid]} n={n} macro_f1={m.get('f1_macro')} "
            f"precision={m.get('precision')} recall={m.get('recall')}",
            flush=True,
        )

    output_folder.mkdir(parents=True, exist_ok=True)
    if per_instance:
        write_csv(output_folder / "final_per_instance_metrics.csv", per_instance)
    write_csv(
        output_folder / "final_test_predictions.csv",
        predictions,
        ["instance_id", "conversation_id", "toot_id", "true_score", "true_label", "predicted_score", "predicted_label"],
    )
    print(f"[test] saved {len(predictions):,} toot predictions", flush=True)

    pooled = metrics_from_confusion(tn, fp, fn, tp)

    def mean_of(key):
        values = [r[key] for r in per_instance if r[key] is not None]
        return float(np.mean(values)) if values else None

    return {
        "client_num": len(per_instance),
        "pooled_test_total": total,
        "pooled_acc": correct / total if total else 0.0,
        "pooled_loss": loss_sum / total if total else 0.0,
        "pooled_weighted_f1": pooled["weighted_f1"],
        "pooled_macro_f1": pooled["macro_f1"],
        "pooled_toxic_precision": pooled["positive_precision"],
        "pooled_toxic_recall": pooled["positive_recall"],
        "mean_instance_macro_f1": mean_of("macro_f1"),
        "mean_instance_toxic_precision": mean_of("toxic_precision"),
        "mean_instance_toxic_recall": mean_of("toxic_recall"),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main(argv=None):
    args = parse_args(argv)
    print(args, flush=True)
    set_seed(args.random_seed)

    data_csv = Path(args.data_csv).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    log_folder = Path(args.log_folder).expanduser().resolve()
    log_folder.mkdir(parents=True, exist_ok=True)

    if not data_csv.exists():
        raise FileNotFoundError(f"Dataset not found: {data_csv}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    fed_args = types.SimpleNamespace(
        training_type="simulation",
        backend="sp",
        using_mlops=False,
        client_optimizer="adamw",
        data_cache_dir=str(data_csv.parent),
        client_num_in_total=0,
        client_num_per_round=args.client_num_per_round,
        comm_round=args.rounds,
        epochs=args.local_epochs,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        federated_optimizer="FedAvg",
        dataset="graphnli",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        toxicity_threshold=args.toxicity_threshold,
        gamma=args.gamma,
        model_path=str(model_path),
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

    dataset, metadata = build_federated_graphnli_loaders(
        str(data_csv),
        batch_size=args.batch_size,
        toxicity_threshold=args.toxicity_threshold,
        min_conversation_length=args.min_conversation_length,
        min_conversations=args.min_conversations,
        n_toxic_conversations=args.n_toxic_conversations,
        n_non_toxic_conversations=args.n_non_toxic_conversations,
        test_fraction=args.test_fraction,
        validation_fraction_of_remaining=args.validation_fraction_of_remaining,
        random_seed=args.random_seed,
        cross_client_deduplicate=args.cross_client_deduplicate,
        holdout_csv=args.holdout_csv,
        shuffle_train_labels=args.shuffle_train_labels,
        verbose=True,
    )

    metadata["client_split_summary"].to_csv(log_folder / "client_splits.csv", index=False)
    metadata["leakage_audit"].to_csv(log_folder / "leakage_audit.csv", index=False)

    client_num = metadata["client_num"]
    fed_args.client_num_in_total = client_num
    fed_args.client_num_per_round = min(fed_args.client_num_per_round, max(1, client_num))
    fed_args.worker_num = fed_args.client_num_per_round
    print(f"[data] {client_num} clients: {metadata['cid_to_instance']}", flush=True)

    patch_fedml()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)

    model = build_model(fed_args)
    fed_args.model_trainer = GraphNLITrainer
    fed_args.model_aggregator = GraphNLIAggregator
    runner = _import_runner()(fed_args, device, dataset, model)

    _sync()
    start = time.time()
    runner.run()
    _sync()
    train_time = time.time() - start
    print(f"[train] {train_time:.2f} s", flush=True)

    start = time.time()
    summary = evaluate_final_test(
        model=model,
        final_test_local=metadata["final_test_local"],
        cid_to_instance=metadata["cid_to_instance"],
        device=device,
        fed_args=fed_args,
        output_folder=log_folder,
    )
    _sync()
    summary["train_time_seconds"] = train_time
    summary["test_time_seconds"] = time.time() - start

    write_csv(log_folder / "final_summary.csv", [summary])
    print(f"[test] summary: {summary}", flush=True)


if __name__ == "__main__":
    main()
