# Distillation Visuals

## 1. The Distillation Pipeline

```mermaid
flowchart LR
    A["🎓 Teacher Model\nDeepSeek-R1 671B\nor Qwen3-235B"]:::teacher
    B["📚 Raw Dataset\n12,000 MATH problems\n+ reasoning traces\n+ answers"]:::data
    C["✂️ Filtered Dataset\n6,695 examples\n≤ 2,048 tokens\n44% discarded"]:::filtered
    D["🔢 Token Sequences\nprompt_ids + answer_ids\nLoss masked on prompt"]:::tokens
    E["🎯 Student Model\nQwen3 0.6B\n2 epochs · AdamW lr=5e-6"]:::student
    F["📊 MATH-500\nEvaluation"]:::eval

    A -->|"generates\n<think>trace</think> + answer"| B
    B -->|"tokenize\nformat_distilled_answer()"| C
    C -->|"build_examples()\nprompt_len stored"| D
    D -->|"answer-only\ncross-entropy loss"| E
    E -->|"scored on\nheld-out test set"| F

    classDef teacher fill:#4a90d9,color:#fff,stroke:#2c6fad
    classDef data    fill:#8e44ad,color:#fff,stroke:#6c3483
    classDef filtered fill:#e67e22,color:#fff,stroke:#ca6f1e
    classDef tokens  fill:#16a085,color:#fff,stroke:#0e7065
    classDef student fill:#27ae60,color:#fff,stroke:#1e8449
    classDef eval    fill:#c0392b,color:#fff,stroke:#922b21
```

---

## 2. Token Length Distribution (before and after filtering)

```
Token length distribution — DeepSeek-R1 traces (12,000 examples)

 ≤  512  ▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░
 ≤ 1024  ▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░
 ≤ 2048  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░  ← cutoff (6,695 kept)
 ≤ 4096  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░
 ≤ 8192  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░
 >  8k   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░  longest: 42,005 tokens

Average (raw):      2,946 tokens
Average (filtered): 1,180 tokens
Removed:            5,305 examples  (44%)
```

---

## 3. MATH-500 Accuracy Comparison

```
MATH-500 Benchmark — Qwen3 0.6B student model

Baseline (no distillation)
████░░░░░░░░░░░░░░░░░░░░░░░░░░  15.2%

After DeepSeek-R1 distillation (cross-family, 671B teacher)
████████████░░░░░░░░░░░░░░░░░░  33.6%   +18.4 pts

After Qwen3-235B distillation (same-family teacher)
████████████████████░░░░░░░░░░  45.0%   +29.8 pts  ← 3× baseline

                               0%       50%      100%

Key: same-family teacher outperforms a 3× larger cross-family teacher.
     The gain is compatibility, not capacity.
```

---

## 4. What Gets Masked During Training

```
Full token sequence fed to the model:

┌─────────────────────────────────────────────────────────┐
│  <|im_start|>user                                       │
│  You are a helpful math assistant...                    │  ← PROMPT
│  Question: Sam is hired for a 20-day period...          │  ← masked
│  Answer: <|im_end|>                                     │  (no loss)
│  <|im_start|>assistant                                  │
├─────────────────────────────────────────────────────────┤
│  <think>                                                │
│  Okay, let me think. x + y = 20...                      │  ← ANSWER
│  ...so y = 6. Let me verify: 14×60 - 6×30 = 660 ✓     │  ← loss
│  </think>                                               │  computed
│                                                         │  here
│  Sam did not work \boxed{6} days.                       │
│  <|im_end|>                                             │
└─────────────────────────────────────────────────────────┘

Loss is computed only on tokens after prompt_len.
The model learns to generate reasoning, not to reconstruct the question.
```
