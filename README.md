# Reasoning LLMs — From the Ground Up

Hands-on implementations and experiments exploring how reasoning models work: from text
generation fundamentals through inference-time compute scaling, reinforcement learning with
verifiable rewards (RLVR), and knowledge distillation.

Each notebook builds understanding of a core technique. Training scripts are the production
versions of those experiments — runnable from the terminal on real hardware.

---

## Topics

| Topic | Notebook | Training Script |
|---|---|---|
| Text generation & autoregressive sampling | [notebooks/text_generation.ipynb](notebooks/text_generation.ipynb) | — |
| Sampling strategies (top-K, top-P, temperature) | [notebooks/sampling_strategies.ipynb](notebooks/sampling_strategies.ipynb) | — |
| Inference-time compute scaling | [notebooks/inference_time_scaling.ipynb](notebooks/inference_time_scaling.ipynb) | — |
| Self-refinement | [notebooks/self_refinement.ipynb](notebooks/self_refinement.ipynb) | — |
| Reinforcement learning for LLMs | [notebooks/reinforcement_learning.ipynb](notebooks/reinforcement_learning.ipynb) | — |
| RLVR with GRPO | [notebooks/rlvr_with_grpo.ipynb](notebooks/rlvr_with_grpo.ipynb) | [scripts/rlvr_grpo_original_no_kl.py](scripts/rlvr_grpo_original_no_kl.py) |
| Improving GRPO (tracking, KL divergence, clip ratio, format rewards) | [notebooks/improving_grpo.ipynb](notebooks/improving_grpo.ipynb) | [scripts/7_3_plus_tracking.py](scripts/7_3_plus_tracking.py) · [7_4](scripts/7_4_plus_clip_ratio.py) · [7_5](scripts/7_5_plus_kl.py) · [7_6](scripts/7_6_plus_format_reward.py) |
| Knowledge distillation | [notebooks/distillation.ipynb](notebooks/distillation.ipynb) | [scripts/distill.py](scripts/distill.py) |
| Evaluation | [notebooks/eval_reasoning.ipynb](notebooks/eval_reasoning.ipynb) | [evaluation/evaluate_math500.py](evaluation/evaluate_math500.py) · [evaluate_json.py](evaluation/evaluate_json.py) |

---

## Concepts

Write-ups on the technical ideas behind each technique, separate from the notebook code:

- [Model Distillation](concepts/model_distillation.md) — hard vs. soft distillation, answer-only loss masking, same-family teacher advantage, results on MATH-500

More to come as the project grows.

---

## Repository Layout

```
notebooks/      Jupyter notebooks — one per topic
scripts/        Terminal-runnable training scripts
evaluation/     Evaluation scripts and notebook
concepts/       Technical write-ups on core ideas
utils/          Shared utilities (data, model, display)
metrics/        Training run outputs (gitignored)
assets/         Reference PDFs
```

---

## Setup

```bash
python3 -m venv venv-reasoning
source venv-reasoning/bin/activate
pip install -r requirements.txt
```

Install the Jupyter kernel for this environment:

```bash
python3 -m ipykernel install --user --name=venv-reasoning
jupyter notebook
```

### Running training scripts

Scripts are designed to be run from the repo root:

```bash
# Example: RLVR with GRPO
python scripts/rlvr_grpo_original_no_kl.py --help

# Example: knowledge distillation
python scripts/distill.py --help
```

---

## Acknowledgements

Implementations follow and extend Sebastian Raschka's
*[Build a Reasoning Model From Scratch](https://mng.bz/lZ5B)*.
