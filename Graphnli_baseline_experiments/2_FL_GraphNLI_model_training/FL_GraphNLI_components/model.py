"""GraphNLI model for federated training."""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, models

from FL_GraphNLI_components.dense import Dense


class FederatedGraphNLI(nn.Module):
    """
    GraphNLI (Agarwal et al.) as a single module that FedML can average.

    Each sample is a random walk of four toots in a conversation tree: the
    target toot followed by three context toots. All four are encoded with a
    transformer and mean pooling. The context embeddings are combined with
    geometrically decreasing weights (``gamma``) into ``u``; the target
    embedding is ``v``. The classifier sees ``[u, v, |u - v|]``, and output
    column 1 is regressed (MSE) on the continuous toxicity score.
    """

    def __init__(self, model_path: str, gamma: float = 0.2):
        super().__init__()

        transformer = models.Transformer(model_path)
        embedding_dim = transformer.get_word_embedding_dimension()
        pooling = models.Pooling(
            embedding_dim,
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        dense = Dense(in_features=3 * embedding_dim, out_features=2)

        self.graphnli = SentenceTransformer(modules=[transformer, pooling, dense])
        self.gamma = float(gamma)
        self.loss_fct = nn.MSELoss()

    def tokenize(self, texts):
        return self.graphnli.tokenize(texts)

    def _encode(self, sentence_features):
        x = self.graphnli[0](sentence_features)
        x = self.graphnli[1](x)
        return x["sentence_embedding"]

    def weighted_aggregate(self, reps):
        """``v`` is the target toot; ``u`` is the weighted sum of the three context toots."""
        v = reps[0]
        weighted = []
        remaining = 1.0
        for i in range(1, 4):
            weight = self.gamma * remaining
            weighted.append(reps[i] * weight)
            remaining -= weight
        u = torch.sum(torch.stack(weighted), dim=0)
        return u, v

    def forward(self, sentence_features, labels=None):
        u, v = self.weighted_aggregate([self._encode(f) for f in sentence_features])
        output = self.graphnli[2](torch.cat([u, v, torch.abs(u - v)], dim=1))

        if labels is not None:
            return output, self.loss_fct(output[:, 1], labels.view(-1))
        return output


def build_model(args):
    return FederatedGraphNLI(model_path=args.model_path, gamma=float(getattr(args, "gamma", 0.2)))
