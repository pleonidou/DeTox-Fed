"""FedML client trainer for GraphNLI."""

from collections import OrderedDict

import torch
from fedml.core import ClientTrainer

from FL_GraphNLI_components.metrics import EMPTY_METRICS, binary_metrics

SENTENCE_COLUMNS = ("sent1", "sent2", "sent3", "sent4")


def _sentence_features(model, batch, device):
    features = []
    for column in SENTENCE_COLUMNS:
        tokens = model.tokenize(batch[column])
        features.append({k: v.to(device) if torch.is_tensor(v) else v for k, v in tokens.items()})
    return features


class GraphNLITrainer(ClientTrainer):
    """
    Local trainer used by every FL client.

    The model regresses the continuous toot toxicity; for evaluation, both the
    target and the prediction are binarised at ``args.toxicity_threshold``.
    """

    def __init__(self, model, args):
        super().__init__(model, args)
        self.model = model
        self.args = args
        self.device = torch.device("cpu")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(getattr(args, "learning_rate", 2e-5)),
            weight_decay=float(getattr(args, "weight_decay", 0.0)),
        )

    def get_model_params(self):
        sd = self.model.state_dict()
        return OrderedDict((k, v.detach().cpu().clone()) for k, v in sd.items())

    def set_model_params(self, params):
        self.model.load_state_dict(params, strict=True)

    def train(self, train_data, device, args):
        self.device = device
        self.model.to(self.device)
        self.model.train()

        local_epochs = int(getattr(args, "epochs", getattr(args, "local_epochs", 1)))
        max_grad_norm = float(getattr(args, "max_grad_norm", 1.0))
        total_loss = 0.0
        total_examples = 0
        last_loss = None

        for epoch in range(local_epochs):
            for batch_idx, batch in enumerate(train_data):
                labels = batch["toxicity"].to(self.device, dtype=torch.float32)
                _, loss = self.model(_sentence_features(self.model, batch, self.device), labels)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()

                n = int(labels.shape[0])
                last_loss = float(loss.detach().cpu())
                total_loss += last_loss * n
                total_examples += n

                if batch_idx == 0 or (batch_idx + 1) % 10 == 0 or batch_idx + 1 == len(train_data):
                    print(
                        f"[client] epoch={epoch + 1}/{local_epochs} "
                        f"batch={batch_idx + 1}/{len(train_data)} loss={last_loss:.6f}",
                        flush=True,
                    )

        return {"train_loss": total_loss / total_examples if total_examples else last_loss}

    @torch.no_grad()
    def test(self, test_data, device, args, return_predictions=False):
        """
        Evaluate on one client's loader.

        Returns the metrics dict (the format FedML expects), or
        ``(metrics, prediction_rows)`` when ``return_predictions=True``.
        """
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        threshold = float(getattr(args, "toxicity_threshold", 0.5))
        total_loss = 0.0
        total_examples = 0
        all_true, all_pred, all_score = [], [], []
        predictions = []

        for batch in test_data:
            labels = batch["toxicity"].to(self.device, dtype=torch.float32)
            output, loss = self.model(_sentence_features(self.model, batch, self.device), labels)

            score = output[:, 1]
            true_cpu = (labels >= threshold).long().cpu().numpy()
            pred_cpu = (score >= threshold).long().cpu().numpy()
            score_cpu = score.cpu().numpy()
            toxicity_cpu = labels.cpu().numpy()

            n = int(labels.shape[0])
            total_loss += float(loss.detach().cpu()) * n
            total_examples += n
            all_true.extend(true_cpu.tolist())
            all_pred.extend(pred_cpu.tolist())
            all_score.extend(score_cpu.tolist())

            if return_predictions:
                for i in range(n):
                    predictions.append({
                        "instance_id": batch["instance_id"][i],
                        "conversation_id": batch["conversation_id"][i],
                        "toot_id": batch["toot_id"][i],
                        "true_score": float(toxicity_cpu[i]),
                        "true_label": int(true_cpu[i]),
                        "predicted_score": float(score_cpu[i]),
                        "predicted_label": int(pred_cpu[i]),
                    })

        if total_examples == 0:
            metrics = dict(EMPTY_METRICS)
        else:
            metrics = binary_metrics(all_true, all_pred, all_score)
            metrics.update({
                "test_total": total_examples,
                "test_correct": int(sum(int(a == b) for a, b in zip(all_true, all_pred))),
                "test_loss": total_loss / total_examples,
            })

        return (metrics, predictions) if return_predictions else metrics
