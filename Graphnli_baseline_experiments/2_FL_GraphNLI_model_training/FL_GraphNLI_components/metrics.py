"""Binary classification metrics."""

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

EMPTY_METRICS = {
    "test_total": 0,
    "test_correct": 0,
    "test_loss": 0.0,
    "test_acc": 0.0,
    "f1_weighted": None,
    "f1_macro": None,
    "precision": None,
    "recall": None,
    "balanced_acc": None,
    "roc_auc": None,
    "pr_auc": None,
    "cm_00": 0,
    "cm_01": 0,
    "cm_10": 0,
    "cm_11": 0,
}


def binary_metrics(y_true, y_pred, y_score):
    """Metrics with labels fixed to {0, 1}; AUCs only when both classes are present."""
    labels = [0, 1]
    tn, fp, fn, tp = map(int, confusion_matrix(y_true, y_pred, labels=labels).ravel())

    result = {
        "test_acc": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "balanced_acc": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "roc_auc": None,
        "pr_auc": None,
        "cm_00": tn,
        "cm_01": fp,
        "cm_10": fn,
        "cm_11": tp,
    }

    if len(set(y_true)) == 2:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            pass
        try:
            result["pr_auc"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            pass

    return result


def _safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def metrics_from_confusion(tn, fp, fn, tp):
    """Per-class, macro and support-weighted metrics; the positive class is toxic."""
    pos_precision = _safe_div(tp, tp + fp)
    pos_recall = _safe_div(tp, tp + fn)
    pos_f1 = _safe_div(2 * pos_precision * pos_recall, pos_precision + pos_recall)

    neg_precision = _safe_div(tn, tn + fn)
    neg_recall = _safe_div(tn, tn + fp)
    neg_f1 = _safe_div(2 * neg_precision * neg_recall, neg_precision + neg_recall)

    support_pos = tp + fn
    support_neg = tn + fp

    return {
        "positive_precision": pos_precision,
        "positive_recall": pos_recall,
        "positive_f1": pos_f1,
        "negative_precision": neg_precision,
        "negative_recall": neg_recall,
        "negative_f1": neg_f1,
        "macro_precision": 0.5 * (pos_precision + neg_precision),
        "macro_recall": 0.5 * (pos_recall + neg_recall),
        "macro_f1": 0.5 * (pos_f1 + neg_f1),
        "weighted_f1": _safe_div(pos_f1 * support_pos + neg_f1 * support_neg, support_pos + support_neg),
    }


METRIC_COLUMNS = [
    "weighted_f1",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "positive_precision",
    "positive_recall",
    "positive_f1",
    "negative_precision",
    "negative_recall",
    "negative_f1",
]


def pool_client_metrics(per_client_metrics):
    """Pool per-client results (dicts from ``GraphNLITrainer.test``). ``None`` if nothing was evaluated."""
    total = correct = 0
    loss_sum = 0.0
    tn = fp = fn = tp = 0

    for m in per_client_metrics:
        t = int(m.get("test_total") or 0)
        total += t
        correct += int(m.get("test_correct") or 0)
        if t:
            loss_sum += float(m.get("test_loss", 0.0)) * t
        tn += int(m.get("cm_00") or 0)
        fp += int(m.get("cm_01") or 0)
        fn += int(m.get("cm_10") or 0)
        tp += int(m.get("cm_11") or 0)

    if total == 0:
        return None

    return {
        "total": total,
        "acc": correct / total,
        "loss": loss_sum / total,
        **metrics_from_confusion(tn, fp, fn, tp),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
