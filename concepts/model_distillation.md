# Model Distillation

Model distillation is a training-time technique for transferring reasoning ability from a large,
expensive **teacher** model into a smaller, deployable **student** model. It is one of the two
core techniques (alongside RLVR) behind most production reasoning models today — the flagship
model is trained with RL, then distilled down for cost-effective deployment.

---

## Hard vs. Soft Distillation

| | Soft Distillation | Hard Distillation |
|---|---|---|
| Trains on | Teacher's full token probability distributions (logits) | Teacher's generated text tokens only |
| Requires | Live access to teacher at training time | Pre-generated synthetic dataset |
| Information richness | Higher — captures uncertainty across vocabulary | Lower — only the sampled sequence |
| Practical? | Expensive; requires co-locating teacher and student | Yes — equivalent to SFT on synthetic data |

Hard distillation is the practical choice: generate teacher traces offline once, then train the
student on them as a standard supervised fine-tuning job. This is the approach used here.

---

## Distillation vs. RLVR

| | RLVR | Distillation |
|---|---|---|
| Supervision signal | Reward from an automated verifier (correct/incorrect) | Direct imitation of teacher's step-by-step solution |
| Dataset | Generated on-the-fly during training | Pre-generated offline |
| Training speed | Slow — active rollout loop per step | Fast — fixed dataset, standard SFT loop |
| Data efficiency | Learns from sparse reward signal | Learns from dense token-level targets |

Distillation is faster to iterate on. RLVR is what creates the strong teacher in the first place.

---

## The Distillation Pipeline

```
Teacher Model (DeepSeek-R1 671B or Qwen3-235B)
        │
        │  generates <think>reasoning trace</think> + final answer
        ▼
Raw Dataset (12,000 MATH problems + teacher solutions)
        │
        │  tokenize → filter sequences > 2,048 tokens → train/val split
        ▼
Filtered Dataset (6,695 examples — 44% discarded)
        │
        │  answer-only cross-entropy loss (prompt tokens masked)
        ▼
Student Model (Qwen3 0.6B, 2 epochs, AdamW lr=5e-6)
        │
        │  evaluated on held-out MATH-500 test set
        ▼
Evaluation Results
```

### Why 44% of data gets discarded

Reasoning models can produce extremely verbose traces — the raw dataset has an average length
of 2,946 tokens per example and a maximum of 42,005 tokens. Filtering to ≤ 2,048 tokens is not
about removing bad problems; it removes examples where the teacher over-explained. The resulting
average drops to 1,180 tokens, and training on this cleaner subset still outperforms using the
full noisy set.

---

## Dataset Format

Each training example is a chat-template sequence:

```
<|im_start|>user
You are a helpful math assistant...
Question: [problem]
Answer: <|im_end|>
<|im_start|>assistant
<think>
[teacher's step-by-step reasoning]
</think>

[final answer]
<|im_end|>
```

The `<think>...</think>` tags explicitly separate reasoning from the answer. This lets
interfaces hide the verbose thinking trace from end users while keeping it as a training target.

---

## Answer-Only Loss Masking

Loss is computed **only on the answer tokens** — everything up to and including the user prompt
is masked out. The model is trained to produce the reasoning trace and answer, not to reconstruct
the input question it already received.

```
[prompt tokens → masked, no loss]  [answer tokens → loss computed here]
```

This is what makes the student learn *how* to reason rather than memorize the question format.

---

## Training Setup

| Hyperparameter | Value |
|---|---|
| Base model | Qwen3 0.6B (pre-trained, not RL-trained) |
| Teacher | DeepSeek-R1 671B or Qwen3-235B |
| Optimizer | AdamW |
| Learning rate | 5e-6 |
| Epochs | 2 |
| Gradient clipping | norm = 1.0 |
| Loss | Answer-only cross-entropy |

Starting from a pre-trained base model (not an RL-trained one) ensures that any reasoning
improvement is directly attributable to the distillation dataset, not prior RL training.

---

## Results on MATH-500

| Student Model | Teacher | MATH-500 Accuracy |
|---|---|---|
| Qwen3 0.6B baseline | — | 15.2% |
| Qwen3 0.6B distilled | DeepSeek-R1 671B (cross-family) | 33.6% (+18.4 pts) |
| Qwen3 0.6B distilled | Qwen3-235B (same-family) | **45.0% (+29.8 pts)** |

### Why same-family distillation wins

A 235B same-family teacher outperforms a 671B cross-family teacher on the same 0.6B student.
The gap is **compatibility, not capacity**:

- **Shared tokenizer** — same tokens for math symbols, LaTeX, and reasoning phrases
- **Shared prompting conventions** — the student already understands the teacher's format
- **Shared output style** — the student's priors align with the teacher's training targets

The teacher's traces become much easier to imitate when the student and teacher speak the same
"dialect" of tokens and formatting.

---

## Key Takeaways

1. Hard distillation = supervised fine-tuning on teacher-generated reasoning traces.
2. Filtering by token length removed 44% of the data — accuracy still doubled from baseline.
3. Answer-only loss masking is critical: the student learns to reason, not to echo the question.
4. Same-family distillation (Qwen3-235B → Qwen3-0.6B) achieves 45.0% on MATH-500, nearly 3× the 15.2% baseline.
5. The production recipe: RLVR trains the flagship teacher; distillation deploys the student.
