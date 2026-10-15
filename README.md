# DeTox-Fed: Detecting Toxic Conversations in the Fediverse with Federated Graph Neural Networks

This repository contains the code and resources for the DeTox-Fed paper:  
[DeTox-Fed: Detecting Toxic Conversations in the Fediverse with Federated Graph Neural Networks](https://arxiv.org/abs/2605.21054).

## Overview

Content moderation in the Fediverse is challenging because conversations, user activity, and moderation decisions are distributed across independently operated instances. **DeTox-Fed** is a federated graph-learning framework for **conversation-level toxicity detection** that enables instances to collaboratively train a moderation model while keeping raw conversations and moderation labels local.

<p align="center">
  <img src="illustrations/DeTox-Fed-Framework-Illustration.svg" width="800" alt="DeTox-Fed framework">
</p>

<p align="center">
  <em>
    Overview of the DeTox-Fed framework.
  </em>
</p>

DeTox-Fed operates in four main stages:

**(A) Raw conversations.** The raw data consist of conversation trees from different DSN instances, where each tree contains an initial post and its replies. Toots from the same user are shown with the same color.

**(B) Conversation graphs.** For each DSN instance, we construct a weighted conversation graph in which nodes represent conversation trees and edge weights represent the number of users shared between two conversations. For example, Trees A and B share three users (Blue, Green, and Red), so the weight of the edge between them is 3. Each node is represented by a feature vector combining tree-level structural embeddings, user-related features, sentiment-based features, and other conversation-level statistics.

**(C) Graph backboning.** We reduce noisy and weak connections by extracting the backbone of each conversation graph, keeping only statistically significant edges.

**(D) Federated toxicity detection.** The federated graph-learning phase, where each instance trains a local Graph Neural Network (GNN) model and shares model updates with the FL coordinator.

We evaluate DeTox-Fed on conversation datasets from **Pleroma** and **Mastodon**, using both Detoxify-derived and GPT-derived conversation labels. We compare DeTox-Fed with a federated version of **GraphNLI** and with **seven LLM models** under few-shot prompting.

## Repository structure

| Folder | Content |
|---|---|
| [`DeTox-Fed_experiments/`](DeTox-Fed_experiments/README.md) | Data preparation, conversation graphs, backboning, and federated training of DeTox-Fed |
| [`Graphnli_baseline_experiments/`](Graphnli_baseline_experiments/README.md) | Federated GraphNLI baseline, built on the data prepared by `DeTox-Fed_experiments` |
| [`LLM_experiments/`](LLM_experiments/README.md) | Few-shot LLM baseline, with the per-seed scores used for the results |

Each folder has its own README with setup and usage instructions.

The raw data are not part of the repository. All code reads from and writes to these folders at the repository root:

```text
data/<dataset>/      # input data and intermediate files (dataset = pleroma or mastodon)
models/              # pretrained models (FastText language ID, DistilRoBERTa)
results/             # training outputs
figures/             # plots from the notebooks
```

## Data

The **Pleroma dataset** introduced by **Bin Zia et al. (2022)** and made available upon request:
<https://netsys.surrey.ac.uk/datasets/dsnmod-pleroma/>
```bibtex
@article{binzia2022toxicity,
  title={Toxicity in the Decentralized Web and the Potential for Model Sharing},
  author={Bin Zia, H. and Raman, A. and Castro, I. and Hassan Anaobi, I. and De Cristofaro, E. and Sastry, N. and Tyson, G.},
  journal={Proceedings of the ACM on Measurement and Analysis of Computing Systems},
  volume={6},
  number={2},
  year={2022}
}
```

The **Mastodon dataset** was collected specifically for this paper. It is available upon request.

## Citation

If you use this code, please cite:

```bibtex
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
