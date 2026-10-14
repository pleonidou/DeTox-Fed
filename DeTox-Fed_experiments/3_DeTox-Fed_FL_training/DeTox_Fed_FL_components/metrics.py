"""Binary classification metrics computed from a pooled confusion matrix."""


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def metrics_from_confusion(tn: int, fp: int, fn: int, tp: int) -> dict:
    """
    Per-class, macro and support-weighted metrics for a binary task.

    The positive class (label 1) is "toxic"; the negative class (label 0) is
    "non-toxic". Negative-class recall is the true-negative rate.
    """
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
        "weighted_f1": _safe_div(
            pos_f1 * support_pos + neg_f1 * support_neg,
            support_pos + support_neg,
        ),
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
    """
    Pool per-client test results into a single set of metrics.

    ``per_client_metrics`` is an iterable of dicts returned by
    ``Trainer.test``. Loss is averaged weighted by the number of test nodes.
    Returns ``None`` when no client has labelled test nodes.
    """
    total = correct = 0
    loss_sum = 0.0
    tn = fp = fn = tp = 0

    for m in per_client_metrics:
        t = int(m.get("test_total") or 0)
        if t == 0:
            continue
        total += t
        correct += int(m.get("test_correct") or 0)
        loss_sum += float(m.get("test_loss", 0.0)) * t
        tn += int(m.get("cm_00") or 0)
        fp += int(m.get("cm_01") or 0)
        fn += int(m.get("cm_10") or 0)
        tp += int(m.get("cm_11") or 0)

    if total == 0:
        return None

    return {
        "test_total": total,
        "acc": correct / total,
        "loss": loss_sum / total,
        **metrics_from_confusion(tn, fp, fn, tp),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
