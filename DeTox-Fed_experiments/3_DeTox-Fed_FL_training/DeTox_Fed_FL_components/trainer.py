"""FedML client trainer for PyTorch Geometric node classification."""

import csv
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from fedml.core import ClientTrainer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

EMPTY_TEST_METRICS = {
    "test_total": 0,
    "test_correct": 0,
    "test_loss": 0.0,
    "test_acc": 0.0,
    "f1_weighted": None,
    "f1_macro": None,
    "roc_auc": None,
    "pr_auc": None,
    "balanced_acc": None,
    "precision": None,
    "recall": None,
    "cm_00": 0,
    "cm_01": 0,
    "cm_10": 0,
    "cm_11": 0,
}


def _prob_positive(logits):
    """Probability of class 1 from ``[N, 2]`` logits."""
    return F.softmax(logits, dim=-1)[:, 1]


def _binary_metrics(y_true, y_pred, y_prob):
    """
    Binary metrics with labels fixed to {0, 1} so shapes stay stable when a
    client's test set contains a single class. AUCs are ``None`` in that case.
    """
    labels = [0, 1]
    out = {
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "balanced_acc": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }

    if len(set(int(v) for v in y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out["cm_00"], out["cm_01"], out["cm_10"], out["cm_11"] = map(int, cm.ravel())
    return out


class Trainer(ClientTrainer):
    """
    Local trainer used by every FL client.

    Each client holds one graph (one Fediverse instance). Training uses the
    ``train_mask`` nodes; evaluation uses either the automatically labelled
    test split (``eval_kind="auto"``) or the separately labelled hold-out set
    (``eval_kind="holdout"``).
    """

    def __init__(self, model, args):
        super().__init__(model, args)
        self.model = model
        self.args = args
        self.device = torch.device("cpu")

        lr = float(getattr(args, "learning_rate", 1e-3))
        wd = float(getattr(args, "weight_decay", 1e-4))
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

        self._base_loss = torch.nn.CrossEntropyLoss(reduction="mean")
        self.use_weighted_loss = bool(int(getattr(args, "weighted_loss", 1)))

        self.log_round_metrics = bool(getattr(args, "log_round_metrics", False))
        self._eval_round = 0

        log_folder = getattr(args, "log_folder", None) or "./logs"
        self.log_folder = Path(log_folder).expanduser().resolve()
        self.log_folder.mkdir(parents=True, exist_ok=True)
        self.round_log_path = self.log_folder / "client_round_metrics.csv"

    def get_model_params(self):
        sd = self.model.state_dict()
        return OrderedDict((k, v.detach().cpu().clone()) for k, v in sd.items())

    def set_model_params(self, params):
        self.model.load_state_dict(params, strict=True)

    @staticmethod
    def _select_mask(g, kind):
        if kind == "train":
            mask = getattr(g, "train_mask", None)
        elif kind == "auto":
            mask = getattr(g, "auto_test_mask", None)
        elif kind == "holdout":
            mask = getattr(g, "holdout_mask", None)
        else:
            raise ValueError(f"Unknown mask kind '{kind}'.")

        if mask is not None:
            return mask
        if getattr(g, "y", None) is not None:
            return g.y >= 0
        return None

    @staticmethod
    def _select_target(g, kind):
        if kind == "holdout":
            return getattr(g, "y_holdout", None)
        target = getattr(g, "y_auto", None)
        return target if target is not None else g.y

    def _forward(self, g):
        if getattr(self.model, "supports_edge_weight", False):
            return self.model(g.x, g.edge_index, getattr(g, "edge_weight", None))
        return self.model(g.x, g.edge_index)

    def _weighted_ce(self, y):
        """Cross-entropy with inverse-frequency class weights over {0, 1}."""
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return self._base_loss
        total = n_pos + n_neg
        weights = torch.tensor(
            [total / (2.0 * n_neg), total / (2.0 * n_pos)],
            dtype=torch.float32,
            device=self.device,
        )
        return torch.nn.CrossEntropyLoss(weight=weights, reduction="mean")

    def train(self, train_data, device, args):
        self.device = device
        self.model.to(self.device)
        self.model.train()

        local_epochs = int(getattr(args, "epochs", getattr(args, "local_epochs", 1)))
        last_loss = None

        for _ in range(local_epochs):
            for batch in train_data:
                g = batch.to(self.device)
                if getattr(g, "y", None) is None:
                    continue
                mask = self._select_mask(g, "train")
                if mask is None or int(mask.sum()) == 0:
                    continue

                logits = self._forward(g)
                y = g.y[mask]
                loss_fn = self._weighted_ce(y) if self.use_weighted_loss else self._base_loss
                loss = loss_fn(logits[mask], y)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                last_loss = float(loss.detach().cpu())

        return {"train_loss": last_loss} if last_loss is not None else {}

    @torch.no_grad()
    def test(self, test_data, device, args, eval_kind="auto"):
        """
        Evaluate on one client's graph(s).

        Returns the keys FedML expects (``test_total``, ``test_correct``,
        ``test_loss``) plus imbalance-aware metrics and the confusion matrix.
        For ``eval_kind="holdout"`` the per-conversation predictions are also
        returned under ``predictions``.
        """
        if eval_kind not in ("auto", "holdout"):
            raise ValueError(f"Unknown eval_kind '{eval_kind}'. Expected 'auto' or 'holdout'.")

        self.device = device
        self.model.to(self.device)
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        all_true, all_pred, all_prob = [], [], []
        predictions = []

        for batch in test_data:
            g = batch.to(self.device)

            target = self._select_target(g, eval_kind)
            mask = self._select_mask(g, eval_kind)
            if target is None or mask is None:
                continue

            mask = mask & (target >= 0)
            m = int(mask.sum())
            if m == 0:
                continue

            logits = self._forward(g)
            y = target[mask]
            loss = self._base_loss(logits[mask], y)
            pred = logits[mask].argmax(dim=-1)
            prob = _prob_positive(logits[mask])

            if eval_kind == "holdout":
                node_idx = torch.where(mask)[0].cpu().tolist()
                conv_ids = g.conversation_ids
                for n, yt, yp, pp in zip(
                    node_idx, y.cpu().tolist(), pred.cpu().tolist(), prob.cpu().tolist()
                ):
                    predictions.append({
                        "conversation_id": int(conv_ids[n].item()),
                        "true_label": int(yt),
                        "predicted_label": int(yp),
                        "predicted_prob_toxic": float(pp),
                    })

            total_loss += float(loss) * m
            total_correct += int((pred == y).sum())
            total_examples += m
            all_true.append(y.cpu())
            all_pred.append(pred.cpu())
            all_prob.append(prob.cpu())

        if total_examples == 0:
            metrics = dict(EMPTY_TEST_METRICS)
            if eval_kind == "holdout":
                metrics["predictions"] = []
            return metrics

        y_true = torch.cat(all_true).numpy().tolist()
        y_pred = torch.cat(all_pred).numpy().tolist()
        y_prob = torch.cat(all_prob).numpy().tolist()

        metrics = {
            "test_total": total_examples,
            "test_correct": total_correct,
            "test_loss": total_loss / total_examples,
            "test_acc": total_correct / total_examples,
            **_binary_metrics(y_true, y_pred, y_prob),
        }
        if eval_kind == "holdout":
            metrics["predictions"] = predictions

        if self.log_round_metrics:
            self._log_round(metrics)

        return metrics

    @torch.no_grad()
    def test_holdout(self, test_data, device, args):
        return self.test(test_data, device, args, eval_kind="holdout")

    def _log_round(self, metrics):
        self._eval_round += 1
        header = [
            "round", "test_total", "test_acc", "test_loss",
            "f1_weighted", "f1_macro", "precision", "recall", "balanced_acc",
            "roc_auc", "pr_auc", "tn", "fp", "fn", "tp",
        ]
        row = [
            self._eval_round,
            metrics["test_total"],
            metrics["test_acc"],
            metrics["test_loss"],
            metrics.get("f1_weighted"),
            metrics.get("f1_macro"),
            metrics.get("precision"),
            metrics.get("recall"),
            metrics.get("balanced_acc"),
            metrics.get("roc_auc"),
            metrics.get("pr_auc"),
            metrics["cm_00"], metrics["cm_01"], metrics["cm_10"], metrics["cm_11"],
        ]
        write_header = not self.round_log_path.exists()
        with open(self.round_log_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(row)
