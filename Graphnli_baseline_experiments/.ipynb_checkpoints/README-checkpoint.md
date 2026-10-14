# Federated GraphNLI baseline

The baseline DeTox-Fed is compared with: GraphNLI trained with federated averaging,
with one Fediverse instance per FL client, on the same Pleroma and Mastodon data.

GraphNLI encodes each toot together with three toots reached by a weighted random walk
over its conversation tree, and predicts the toot's Detoxify toxicity score. The
toot-level predictions are then turned into conversation labels with the same rule
DeTox-Fed uses. The setup follows:

> V. Agarwal, A. Raman, N. Sastry, A. M. Abdelmoniem, G. Tyson, I. Castro.
> *Decentralised Moderation for Interoperable Social Networks: A Conversation-based
> Approach for Pleroma and the Fediverse.* ICWSM 2024.

> V. Agarwal, S. Joglekar, A. P. Young, N. Sastry.
> *GraphNLI: A Graph-based Natural Language Inference Model for Polarity Prediction in
> Online Debates.* The Web Conference (WWW) 2022.

The random-walk procedure, the model (DistilRoBERTa, mean pooling, γ-weighted context
aggregation, `[u, v, |u − v|]` head) and `FL_GraphNLI_components/dense.py` follow the
original GraphNLI implementation.

## Folder structure

```
Graphnli_baseline_experiments/
├── 1_create_conversation_graph/
│   ├── 01_conversation_trees.ipynb          # one reply tree per conversation and instance
│   └── 02_random_walks.ipynb                # one random walk per toot (training samples)
├── 2_FL_GraphNLI_model_training/
│   ├── train.py                             # federated training and test evaluation
│   ├── FL_GraphNLI_components/              # data loader, model, FedML trainer and aggregator
│   ├── args_for_training/                   # pleroma.args, mastodon.args
│   └── GraphNLI_experiments_scripts/        # run_seeds.sh, slurm_job.sh
├── 3_conversation-level-labels-evaluation/
│   └── conversation_level_evaluation.ipynb  # toot predictions → conversation labels and metrics
└── requirements.txt
```

## Setup

Use the main DeTox-Fed environment (`FedGNN`, see `DeTox-Fed_experiments/README.md`)
and add `sentence-transformers`:

```bash
pip install -r Graphnli_baseline_experiments/requirements.txt
```

Training runs offline, so download the transformer to `models/` at the repository root:

```bash
huggingface-cli download distilroberta-base --local-dir models/distilroberta-base
```

## Pipeline

The inputs come from the DeTox-Fed notebooks in
`DeTox-Fed_experiments/1_creating_conversation_graph` (steps 1 and 2):
`<dataset>_data_with_toxicity.csv`, `<dataset>_data.csv`, `replies.csv` and
`final_<dataset>_nodes_conv_tox.csv`, all in `data/<dataset>/`.

1. `01_conversation_trees.ipynb` → `data/<dataset>/graphnli/<dataset>_graphnli_conversations.pkl`
2. `02_random_walks.ipynb` → `data/<dataset>/graphnli/<dataset>_graphnli_random_walks.csv`
3. Training, from the repository root:

   ```bash
   python Graphnli_baseline_experiments/2_FL_GraphNLI_model_training/train.py \
       @Graphnli_baseline_experiments/2_FL_GraphNLI_model_training/args_for_training/mastodon.args

   # the three runs of the paper (one per seed):
   Graphnli_baseline_experiments/2_FL_GraphNLI_model_training/GraphNLI_experiments_scripts/run_seeds.sh mastodon
   ```

   The `.args` files expect the test conversations in
   `data/<dataset>/annotations/holdout_test_set.csv`. Results go to
   `results/graphnli/<dataset>/run_<seed>/`.

4. `conversation_level_evaluation.ipynb`: turns the toot-level test predictions into
   conversation labels and evaluates them against Detoxify-derived labels and against
   external (e.g. LLM) labels, including a rule calibrated on a separate split.

Run the notebooks from their own folder.

## Data split

- Per instance, conversations with at least `--min_conversation_length` toots are
  eligible, and an instance needs at least `--min_conversations` of them.
- Per instance, up to `--n_toxic_conversations` conversations containing a toxic toot
  and `--n_non_toxic_conversations` without one are sampled.
- The same conversation can appear on several instances, so conversations are split
  once, globally: a conversation is in the same split on every client.
- With `--holdout_csv` (used in the paper), the test set is exactly the
  `(instance_toot_found_id, conversation_id)` pairs in that file. These conversations
  are not sampled for training or validation on any instance, and each pair must
  belong to an FL client. Only the pairs are read; the toot targets stay the Detoxify
  scores. The other conversations are split into training and validation
  (`--validation_fraction_of_remaining`). Without `--holdout_csv`, conversations are
  split at random into training, validation and test (64 / 16 / 20 %).
- A toot can belong to more than one conversation, so toots of test conversations are
  removed from training and validation, and toots of validation conversations from
  training, on every client. The loader checks that no conversation or toot is
  shared between splits.
- During training the server evaluates on the validation sets. The test sets are used
  once, after training. FedAvg weights clients by their number of training toots.

`--shuffle_train_labels` permutes the training labels on every client. It is a sanity
check (the model should then not learn) and is off by default.

## Outputs (`--log_folder`)

| File | Content |
|---|---|
| `client_splits.csv` | conversations and toots per client and split |
| `leakage_audit.csv` | overlap counts between splits (all zero) |
| `server_round_metrics.csv` | validation metrics every `--frequency_of_the_test` rounds |
| `final_per_instance_metrics.csv` | test metrics per instance |
| `final_test_predictions.csv` | test prediction per toot |
| `final_summary.csv` | pooled and per-instance test metrics, training and test time |
