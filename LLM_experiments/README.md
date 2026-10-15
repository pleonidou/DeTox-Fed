# LLM Experiments

This folder holds the LLM baseline from the DeTox-Fed paper. We test whether a Fediverse moderator could simply prompt an LLM with a few labeled examples instead of training DeTox-Fed.

## Setup

* **Seeds:** 1, 42, 999. Results are the mean ± standard deviation over the three seeds.
* **Instances:** each seed samples 10 instances. Every instance gets its own classifier.
* **Prompt:** each instance builds its prompt from 20 of its own conversations, 10 toxic and 10 non-toxic.
* **Local test set:** 100 of the instance's own conversations, 50 toxic and 50 non-toxic.
* **Shared global test set:** 100 conversations drawn from all 10 instances. Every instance scores the same set.
* **Conversations:** only conversations with at least 10 toots are used (≥10).
* **Labels:** derived automatically from Detoxify toot scores.
* **Prompt variants:** the top tier models are run twice. **Base** is the standard prompt. **Rule** adds a sentence describing the labeling rule, without the threshold values.

Each run is 6,000 classifications per model and dataset (10 instances × 200 conversations × 3 seeds).

## Models

| Model | Provider | Context window | Prompts | How it was run |
|---|---|---|---|---|
| **Top tier** | | | | |
| Claude Opus 5 | Anthropic | 1M | Base, Rule | Batch API |
| GPT-5.6 Sol | OpenAI | 1.05M | Base, Rule | Batch API |
| Gemini 3.7 Flash | Google | 1M | Base, Rule | Batch API |
| **Lightweight** | | | | |
| Claude Haiku 4.5 | Anthropic | 200K | Base | API, 50 requests/min |
| GPT-5.4 mini | OpenAI | 400K | Base | API, 50 requests/min |
| Gemini 3.1 Flash-Lite | Google | 1M | Base | API, 50 requests/min |
| Dolphin 3 (8B) | Open-source | 128K | Base | Colab T4 GPU (no API cost) |

Context windows are the providers' advertised maximums (tokens). Our prompts use only a small part of them.

## Results

All metrics are in %. Bold marks the best value per column within each tier.

### Pleroma

| Model | Prompt | Test set | Macro-F1 | Toxic precision | Toxic recall |
|---|---|---|---|---|---|
| **Top tier** | | | | | |
| Claude Opus 5 | Base | Local | 66.20 ± 0.09 | 77.35 ± 2.10 | 49.13 ± 1.97 |
|  |  | Shared Global | 66.19 ± 1.09 | 72.19 ± 4.15 | 55.13 ± 6.18 |
|  | Rule | Local | 62.61 ± 1.37 | 76.50 ± 1.54 | 42.07 ± 3.69 |
|  |  | Shared Global | 65.51 ± 2.09 | 72.35 ± 3.95 | 53.27 ± 8.62 |
| GPT-5.6 Sol | Base | Local | **74.19 ± 1.07** | 81.68 ± 1.38 | 63.33 ± 3.89 |
|  |  | Shared Global | 73.62 ± 0.97 | 76.34 ± 4.16 | **69.40 ± 5.20** |
|  | Rule | Local | 69.51 ± 1.22 | 82.86 ± 1.46 | 51.97 ± 3.50 |
|  |  | Shared Global | 69.62 ± 1.50 | 75.83 ± 5.22 | 59.52 ± 5.47 |
| Gemini 3.7 Flash | Base | Local | 70.37 ± 0.30 | **84.33 ± 1.77** | 52.64 ± 1.26 |
|  |  | Shared Global | 70.29 ± 1.02 | 78.25 ± 5.73 | 58.43 ± 6.44 |
|  | Rule | Local | 66.65 ± 1.69 | 84.15 ± 2.86 | 45.36 ± 1.84 |
|  |  | Shared Global | 66.17 ± 1.53 | 77.24 ± 6.87 | 50.16 ± 7.88 |
| **Lightweight** | | | | | |
| Dolphin 3 (8B) | Base | Local | 65.17 ± 1.58 | 64.27 ± 0.12 | 68.93 ± 7.75 |
|  |  | Shared Global | 68.34 ± 5.31 | 66.14 ± 3.80 | **76.00 ± 12.17** |
| Claude Haiku 4.5 | Base | Local | 62.87 ± 1.34 | **79.35 ± 3.44** | 41.07 ± 4.79 |
|  |  | Shared Global | 65.51 ± 5.28 | 73.98 ± 6.02 | 51.33 ± 11.81 |
| GPT-5.4 mini | Base | Local | 69.56 ± 0.83 | 75.66 ± 2.21 | 59.13 ± 5.08 |
|  |  | Shared Global | **72.05 ± 2.00** | 74.06 ± 5.72 | 69.40 ± 5.37 |
| Gemini 3.1 Flash-Lite | Base | Local | 66.09 ± 1.96 | 77.04 ± 1.23 | 49.22 ± 5.60 |
|  |  | Shared Global | 67.44 ± 2.80 | 71.73 ± 6.45 | 60.15 ± 8.98 |

### Mastodon

| Model | Prompt | Test set | Macro-F1 | Toxic precision | Toxic recall |
|---|---|---|---|---|---|
| **Top tier** | | | | | |
| Claude Opus 5 | Base | Local | 75.69 ± 1.85 | 76.30 ± 0.69 | 74.53 ± 4.22 |
|  |  | Shared Global | 74.16 ± 4.98 | 75.96 ± 2.05 | 70.73 ± 10.90 |
|  | Rule | Local | 73.50 ± 2.17 | 77.54 ± 2.33 | 66.53 ± 2.80 |
|  |  | Shared Global | 71.38 ± 4.90 | 76.60 ± 1.76 | 62.33 ± 10.33 |
| GPT-5.6 Sol | Base | Local | 79.64 ± 2.21 | 77.97 ± 2.04 | 82.73 ± 3.58 |
|  |  | Shared Global | **80.64 ± 6.24** | 78.45 ± 5.03 | **84.40 ± 7.90** |
|  | Rule | Local | 78.55 ± 2.17 | 80.82 ± 1.22 | 74.92 ± 3.82 |
|  |  | Shared Global | 77.77 ± 5.25 | 80.69 ± 3.92 | 73.04 ± 8.65 |
| Gemini 3.7 Flash | Base | Local | 73.87 ± 0.77 | 84.92 ± 1.76 | 59.53 ± 1.60 |
|  |  | Shared Global | 72.41 ± 4.02 | 83.88 ± 6.61 | 57.33 ± 3.75 |
|  | Rule | Local | 71.43 ± 1.71 | **85.33 ± 2.65** | 54.13 ± 2.72 |
|  |  | Shared Global | 68.26 ± 2.76 | 84.83 ± 6.03 | 48.20 ± 2.09 |
| **Lightweight** | | | | | |
| Dolphin 3 (8B) | Base | Local | 61.27 ± 2.45 | 59.14 ± 1.57 | **86.80 ± 1.64** |
|  |  | Shared Global | 65.78 ± 2.85 | 62.34 ± 1.79 | 85.27 ± 6.09 |
| Claude Haiku 4.5 | Base | Local | 70.29 ± 2.42 | 69.93 ± 2.14 | 71.20 ± 3.34 |
|  |  | Shared Global | 72.58 ± 3.57 | 73.17 ± 0.65 | 71.47 ± 9.66 |
| GPT-5.4 mini | Base | Local | 73.04 ± 3.60 | 69.54 ± 3.52 | 83.33 ± 3.00 |
|  |  | Shared Global | **73.68 ± 2.98** | 70.60 ± 3.22 | 82.20 ± 4.33 |
| Gemini 3.1 Flash-Lite | Base | Local | 71.52 ± 1.96 | 75.62 ± 2.57 | 64.13 ± 3.63 |
|  |  | Shared Global | 71.53 ± 7.19 | **76.29 ± 7.72** | 62.93 ± 8.29 |

## Cost and runtime

Each row is one full run of 6,000 classifications. Tokens include input, cached and output tokens.

* **Lightweight models** were called one request at a time, capped at 50 requests per minute. Runtime is the wall clock time of the whole run.
* **Top tier models** were sent through each provider's batch API, which charges 50% of the normal price. Runtime is how long the provider took to return the batch, and both prompts were sent together, so it is shown once per model.
* **Dolphin 3** ran on a Colab GPU, so it has no API cost.

Bold marks the lowest value per column within each tier.

### Pleroma

| Model | Prompt | Cost ($) | Tokens | $ per 1M tokens | Runtime (h) |
|---|---|---|---|---|---|
| **Lightweight** | | | | | |
| Claude Haiku 4.5 | Base | 12.33 | 34.31M | 0.36 | ~2.5 |
| GPT-5.4 mini | Base | **5.38** | **30.28M** | **0.18** | ~2.5 |
| Gemini 3.1 Flash-Lite | Base | 10.72 | 32.30M | 0.33 | ~3.0 |
| **Top tier** | | | | | |
| Claude Opus 5 | Base | 46.83 | 46.84M | 1.00 | ~0.25 |
|  | Rule | 48.44 | 47.38M | 1.02 |  |
| GPT-5.6 Sol | Base | 41.82 | **30.57M** | 1.37 | ~3.5 |
|  | Rule | 42.41 | 30.91M | 1.37 |  |
| Gemini 3.7 Flash | Base | **11.92** | 32.73M | **0.36** | ~0.6 |
|  | Rule | 12.12 | 33.07M | 0.37 |  |

### Mastodon

| Model | Prompt | Cost ($) | Tokens | $ per 1M tokens | Runtime (h) |
|---|---|---|---|---|---|
| **Lightweight** | | | | | |
| Claude Haiku 4.5 | Base | 9.32 | 43.25M | 0.22 | ~2.5 |
| GPT-5.4 mini | Base | **6.54** | **38.03M** | **0.17** | ~2.5 |
| Gemini 3.1 Flash-Lite | Base | 11.80 | 42.45M | 0.28 | ~4.0 |
| **Top tier** | | | | | |
| Claude Opus 5 | Base | 43.32 | 61.23M | 0.71 | ~0.25 |
|  | Rule | 40.24 | 61.76M | 0.65 |  |
| GPT-5.6 Sol | Base | 52.44 | **38.06M** | 1.38 | ~3.5 |
|  | Rule | 52.92 | 38.38M | 1.38 |  |
| Gemini 3.7 Flash | Base | **14.21** | 40.16M | **0.35** | ~0.6 |
|  | Rule | 14.52 | 40.50M | 0.36 |  |

**Total spend:** $56.09 for the lightweight models and $421.19 for the top tier models.