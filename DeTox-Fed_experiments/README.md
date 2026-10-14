# DeTox-Fed experiments

Code for DeTox-Fed: a federated graph neural network that detects toxic conversations
in the Fediverse. Each Fediverse instance is one federated-learning (FL) client. A
client builds a graph of the conversations observed on its instance, where two
conversations are connected when they share users, and trains a GNN node classifier
locally. Only model parameters are sent to the server, which averages them (FedAvg).
The code covers the Pleroma and Mastodon datasets.

## Folder structure

```
DeTox-Fed_experiments/
├── 1_creating_conversation_graph/
│   ├── 01a_preprocess_dataset_pleroma.ipynb     # ID mappings, toot observations, reply edges
│   ├── 01b_preprocess_dataset_mastodon.ipynb
│   ├── 02a_feature_pipeline_pleroma.ipynb       # toxicity, sentiment, features per conversation
│   ├── 02b_feature_pipeline_mastodon.ipynb
│   ├── 02c_descriptive_analysis.ipynb           # dataset statistics and figures
│   └── 03_conversation_graph.ipynb              # conversation graph per instance
├── 2_conversation_graph_backboning/
│   ├── noise_corrected_backboning.ipynb         # noise-corrected backbone of each graph
│   └── backboning_third_party_code/             # Coscia & Neffke (2017), unmodified
├── 3_DeTox-Fed_FL_training/
│   ├── train_DeTox-Fed.py                       # federated training and evaluation
│   ├── DeTox_Fed_FL_components/                 # data loader, features, models, FedML trainer and aggregator
│   ├── args_for_training/                       # pleroma.args, mastodon.args
│   └── DeTox-Fed_paper_experiments_scripts/     # experiment scripts of the paper
├── requirements.txt
└── requirements-deepwalk.txt
```

## Environments

Python 3.10. Two environments are needed, because Karate Club (used for DeepWalk)
conflicts with the package versions FedML needs.

```bash
# Main environment: everything except the DeepWalk step
conda create -n FedGNN python=3.10
conda activate FedGNN
pip install -r DeTox-Fed_experiments/requirements.txt

# DeepWalk environment: only Stage 6 of notebooks 02a/02b
conda create -n DeepWalk python=3.10
conda activate DeepWalk
pip install -r DeTox-Fed_experiments/requirements-deepwalk.txt
```

The language-identification step uses the FastText model `lid.176.bin`
([download](https://fasttext.cc/docs/en/language-identification.html)). Place it in
`models/` at the repository root.

## Data

The datasets are not included. All steps read from and write to `data/<dataset>/` at
the repository root. The raw data is expected in this layout:

```
data/
├── pleroma/raw/pleroma_conversations.zip
└── mastodon/raw/<topic>/<instance>/{posts.json, replies.json}
```

## Pipeline

Run the notebooks from their own folder, in this order. Set `DATASET` at the top of
the notebooks that are shared by both datasets.

| Step | Notebook | Main output in `data/<dataset>/` |
|---|---|---|
| 1 | `01a_preprocess_dataset_pleroma` / `01b_preprocess_dataset_mastodon` | ID mapping files, `<dataset>_toots_object.csv`, `replies.csv` |
| 2 | `02a_feature_pipeline_pleroma` / `02b_feature_pipeline_mastodon` | `final_<dataset>_nodes_conv_tox.csv` (toxicity per conversation) and the conversation feature matrix |
| – | `02c_descriptive_analysis` | figures and tables in `figures/<dataset>/` |
| 3 | `03_conversation_graph` | `<dataset>_intra_edges_1_common_user.pkl` |
| 4 | `noise_corrected_backboning` | `<dataset>_intra_edges_1_common_user_nc_backbone.pkl` |
| 5 | `train_DeTox-Fed.py` | results in `results/detoxfed/<dataset>/` |

Notebook 02 writes the feature matrix for all conversations (`_all`) and for
conversations with at least 5 and 10 toots (`_length_5`, `_length_10`).

### Conversation features

| Block | Dim | Flag |
|---|---|---|
| DeepWalk embedding of the reply tree (mean, sum and max pooled) | 384 | `--use_deepwalk` |
| Author features (mean, sum and max pooled) | 9 | `--use_author` |
| Sentiment: mean, final, minimum, negative drift, volatility | 5 | `--use_sentiment` |
| log(1 + conversation length) | 1 | `--use_conv_length` |
| log(1 + number of unique authors) | 1 | `--use_num_user` |
| log(1 + median reaction time) | 1 | `--use_reaction_time` |

Each block is standardised per client, using only that client's training nodes.

### Labels

By default, a conversation is toxic when its root toot is toxic, or when at least
`--conversation_level_toxic_threshold_num` replies **and** at least a fraction
`--conversation_level_toxic_threshold_fraction` of the replies are toxic. A toot is
toxic when its Detoxify score is at least `--toxicity_binary_threshold`.

Instead, `--labels_csv` can give ready-made 0/1 conversation labels (e.g. from an
LLM), in the column named by `--conversation_toxicity_method`.

## Training

From the repository root:

```bash
python DeTox-Fed_experiments/3_DeTox-Fed_FL_training/train_DeTox-Fed.py \
    @DeTox-Fed_experiments/3_DeTox-Fed_FL_training/args_for_training/pleroma.args \
    --use_deepwalk --use_author --use_sentiment \
    --use_conv_length --use_num_user --use_reaction_time
```

The `.args` files hold the settings of the paper. Any option can be overridden on the
command line; `--help` lists them all.

Outputs in `--log_folder`:

| File | Content |
|---|---|
| `server_round_metrics.csv` | global-model metrics every `--frequency_of_the_test` rounds |
| `results_summary.csv` (or `--results_csv`) | final test metrics, one row per run |
| `graph_nodes.csv` | conversations in each client graph |

### Hold-out test set

`--holdout_csv` takes a CSV with `conversation_id` and a 0/1 label column
(`--holdout_label_col`, default `llm_label`). These conversations stay in the graphs
but are excluded from training and validation, and are evaluated once after training.
This writes `holdout_results_summary.csv` and `holdout_predictions.csv` as well. It
can be combined with `--labels_csv`.

```bash
python DeTox-Fed_experiments/3_DeTox-Fed_FL_training/train_DeTox-Fed.py \
    @DeTox-Fed_experiments/3_DeTox-Fed_FL_training/args_for_training/mastodon.args \
    --use_deepwalk --use_author --use_sentiment \
    --use_conv_length --use_num_user --use_reaction_time \
    --holdout_csv data/mastodon/annotations/holdout_test_set.csv \
    --log_folder results/detoxfed/mastodon/holdout
```

## Paper experiments

Each script runs six seeds per setting and appends one row per run to a CSV in
`results/detoxfed/<dataset>/`. Run them from the repository root:

```bash
S=DeTox-Fed_experiments/3_DeTox-Fed_FL_training/DeTox-Fed_paper_experiments_scripts
$S/run_sensitivity.sh pleroma       # label budget, toxicity threshold, clients per round, conversation length
$S/run_feature_ablation.sh pleroma  # feature blocks
$S/run_hyperparameters.sh pleroma   # learning rate, local epochs
```

The `tag` column of the CSV names the setting of each row. Average over the seeds
per `tag` to get the reported values.
