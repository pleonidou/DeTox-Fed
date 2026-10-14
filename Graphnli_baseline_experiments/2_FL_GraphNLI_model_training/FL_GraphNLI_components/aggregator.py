"""FedML server aggregator: validates the global model on every client."""

import csv
from collections import OrderedDict
from pathlib import Path

import torch
from fedml.core import ServerAggregator

from FL_GraphNLI_components.metrics import METRIC_COLUMNS, pool_client_metrics
from FL_GraphNLI_components.trainer import GraphNLITrainer


class GraphNLIAggregator(ServerAggregator):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.model = model
        self.args = args

        if getattr(args, "using_gpu", False) and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.round_idx = 0

        log_folder = Path(getattr(args, "log_folder", "./logs")).expanduser().resolve()
        log_folder.mkdir(parents=True, exist_ok=True)
        self.round_log_path = log_folder / "server_round_metrics.csv"

    def get_model_params(self):
        sd = self.model.state_dict()
        return OrderedDict((k, v.detach().cpu().clone()) for k, v in sd.items())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=True)

    def test(self, test_data, device, args):
        self.device = device
        eval_args = self.args if args is None else args
        return GraphNLITrainer(self.model, eval_args).test(test_data, self.device, eval_args)

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args=None, round_idx=None) -> bool:
        """Evaluate on every client's validation loader and log the pooled metrics."""
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        eval_args = self.args if args is None else args
        trainer = GraphNLITrainer(self.model, eval_args)

        pooled = pool_client_metrics(
            trainer.test(loader, self.device, eval_args) for loader in test_data_local_dict.values()
        )
        if pooled is None:
            print("[server] No validation data.", flush=True)
            return True

        if round_idx is None:
            self.round_idx += 1
            round_idx = self.round_idx

        print(
            f"[server] round={round_idx} n={pooled['total']} acc={pooled['acc']:.4f} loss={pooled['loss']:.4f} "
            f"precision={pooled['positive_precision']:.4f} recall={pooled['positive_recall']:.4f} "
            f"macro_f1={pooled['macro_f1']:.4f} weighted_f1={pooled['weighted_f1']:.4f} "
            f"| tn={pooled['tn']} fp={pooled['fp']} fn={pooled['fn']} tp={pooled['tp']}",
            flush=True,
        )

        header = ["round", "validation_total", "acc", "loss", *METRIC_COLUMNS, "tn", "fp", "fn", "tp"]
        row = [round_idx, pooled["total"], pooled["acc"], pooled["loss"]]
        row += [pooled[c] for c in METRIC_COLUMNS]
        row += [pooled["tn"], pooled["fp"], pooled["fn"], pooled["tp"]]

        write_header = not self.round_log_path.exists()
        with open(self.round_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)

        return True
