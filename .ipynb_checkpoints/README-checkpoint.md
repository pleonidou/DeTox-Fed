# DeTox-Fed: Detecting Toxic Conversations in the Fediverse with Federated Graph Neural Networks

This repository contains the code and resources for the DeTox-Fed paper:
[DeTox-Fed: Detecting Toxic Conversations in the Fediverse with Federated Graph Neural Networks](https://arxiv.org/abs/2605.21054).

## Overview

DeTox-Fed detects toxic conversations with federated learning: each instance is one
client and keeps its data local. A client represents the conversations observed on
its instance as a graph, in which conversations are linked when they share users. The
graph is filtered with noise-corrected backboning. Each conversation is described by
the structure of its reply tree, the activity of its authors, its sentiment and
simple conversation statistics. Clients train a graph neural network locally, and the
server combines their models with federated averaging.


We evaluate DeTox-Fed on Pleroma and Mastodon data, using toxicity labels derived from
Detoxify scores and labels produced by an LLM. We compare it with a federated version
of GraphNLI and with few-shot LLM classifiers.

## Repository structure

| Folder | Content |
|---|---|
| [`DeTox-Fed_experiments/`](DeTox-Fed_experiments/README.md) | Data preparation, conversation graphs, backboning, and federated training of DeTox-Fed |
| [`Graphnli_baseline_experiments/`](Graphnli_baseline_experiments/README.md) | Federated GraphNLI baseline, built on the data prepared by `DeTox-Fed_experiments` |
| [`LLM_experiments/`](LLM_experiments/README.md) | Few-shot LLM baseline, with the per-seed scores used for the results |

Each folder has its own README with setup and usage instructions.

The data, models and results are not part of the repository. All code reads from and
writes to these folders at the repository root:

```
data/<dataset>/      # input data and intermediate files (dataset = pleroma or mastodon)
models/              # pretrained models (FastText language ID, DistilRoBERTa)
results/             # training outputs
figures/             # plots from the notebooks
```

## Data

The Pleroma dataset is not redistributed here. It was published by Agarwal et al.
(ICWSM 2024) and is available on request at
<https://netsys.surrey.ac.uk/datasets/dsnmod-pleroma/>.


```
@inproceedings{agarwal2024decentralised,
  title={Decentralised Moderation for Interoperable Social Networks: A Conversation-based Approach for Pleroma and the Fediverse},
  author={Agarwal, Vibhor and Raman, Aravindh and Sastry, Nishanth and Abdelmoniem, Ahmed M and Tyson, Gareth and Castro, Ignacio},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  volume={18},
  pages={2--14},
  year={2024}
}
```

The Mastodon dataset collected for this work will be available upon request.-- IN PROGRESS

## Citation

If you use this code, please cite:

```
@misc{leonidou2026detoxfed,
  title={DeTox-Fed: Detecting Toxic Conversations in the Fediverse with Federated Graph Neural Networks},
  author={Pantelitsa Leonidou and Nikos Salamanos and Sotiris Gypsiotis and Michael Sirivianos},
  year={2026},
  eprint={2605.21054},
  archivePrefix={arXiv},
  primaryClass={cs.SI},
  url={https://arxiv.org/abs/2605.21054}
}
```
