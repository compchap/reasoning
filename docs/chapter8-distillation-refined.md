# Model Distillation: Teaching Small Models to Reason
**Chapter 8 — LLM Study Club**

**Type:** Lecture / peer teaching
**Duration:** 30 min talk + 20 min discussion
**Audience:** Study club — familiar with SFT, cross-entropy loss, RLVR, GRPO

---

## Slide 1 — Title

**Model Distillation: Teaching Small Models to Reason**
*Chapter 8 · LLM Study Club*

**Empowerment promise (spoken, not on slide):**
> "By the end of the next 30 minutes, you'll understand how to transfer deep
> reasoning ability from a 671-billion-parameter model into a 0.6B model that
> runs on your laptop — and you'll see exactly why distillation is the technique
> powering most deployed reasoning models today."

*(No joke. No "good morning." Start with the promise.)*

---

## Slide 2 — Quick Recap: Where We Left Off

**Core idea:** Ground the audience before introducing anything new.

- RLVR trains a model by giving it rewards for correct answers
- GRPO estimates advantages across a group of generated solutions
- Result: models that *learn to reason* — but only if you can afford to run them

**Fence (spoken):**
> "Last session we got a model that can reason. Today's question is: how do
> you make that reasoning cheap enough to actually ship?"

**Verbal punctuation:**
> "That's the setup. Now — distillation."

---

## Slide 3 — What Is Distillation?

**Core idea (Cycle 1):** A small student model learns by imitating a large teacher's reasoning traces.

- **Teacher:** A large, capable model (e.g. DeepSeek-R1 671B) generates step-by-step solutions
- **Student:** A smaller model (e.g. Qwen3 0.6B) is trained to reproduce those solutions
- **Key insight:** The student learns *how* the teacher thinks, not just *what* it answers

**Hard vs. Soft distillation (fence):**

| | Soft Distillation | Hard Distillation |
|---|---|---|
| Trains on | Teacher's full token probabilities (logits) | Teacher's generated text only |
| Requires | Access to teacher at training time | Pre-generated dataset |
| Practical? | Expensive, logistically complex | Yes — just SFT on synthetic data |

**Cycle 2 (spoken):**
> "Hard distillation is supervised fine-tuning on synthetic data. That's all it is."

**Verbal punctuation:**
> "Clear on the setup? Let's look at the data."

---

## Slide 4 — Building the Dataset

**Core idea:** Generate reasoning traces from the teacher offline; store them as training targets.

- **Source:** 12,000 problems from the MATH dataset (MATH-500 held out — no leakage)
- **Teacher generates:** step-by-step reasoning + final answer
- **Format:** reasoning wrapped in `<think>...</think>` tags, answer follows

**[VISUAL: show a live example from the notebook]**
*→ Demo cell: `pprint(math_train[4])` — show `message_thinking` and `message_content`*
*→ Then show `format_distilled_answer()` output — the `<think>` format*

**Cycle 3 (spoken, pointing to the thinking field):**
> "The teacher's internal monologue becomes the student's training target.
> That's the whole idea — borrow the thinking, shrink the model."

*(This is your slogan — repeat it.)*

---

## Slide 5 — Tokenizing & Filtering

**Core idea:** Build token sequences, then aggressively filter outliers.

**Pipeline:**
1. Encode: `[prompt_ids] + [answer_ids] + [eos]`
2. Mask: loss computed **only** on answer tokens — prompt is ignored
3. Filter: remove sequences longer than 2,048 tokens

**The surprising number:**
```
Original:  12,000 examples
Filtered:   6,695 examples   ← 44% discarded
```

**[VISUAL: token length distribution — mark this for a histogram from notebook]**
*→ The outlier story: avg 2,946 tokens, longest 42,005 tokens*
*→ After filter: avg 1,180 tokens, max 2,048*

**Fence:**
> "We didn't throw away bad problems — we threw away problems where the
> teacher over-explained. Long traces are noisier training targets."

**[VISUAL: see `docs/distillation-pipeline.md` for pipeline diagram]**

---

## Slide 6 — The Training Loop

**Core idea (cycling back):** Standard supervised learning — what makes it work is *what* you're supervising on.

**Training setup:**
- Optimizer: AdamW, lr = 5e-6
- Epochs: 2 passes through the filtered 6,695 examples
- Gradient clipping: norm = 1.0
- Loss: answer-only cross-entropy

**[VISUAL: training curve from notebook — show this live]**
*→ Demo: `plot_distill_metrics("deepseek-r1-2048_distill_metrics.csv")`*
*→ Point out: train loss noisy, val loss smooth — **val loss is the signal***

**Audience question (pause ~7 seconds):**
> "We're running 2 epochs. What would you expect to happen to validation loss
> if we kept training for 10 epochs?"

*(Looking for: overfitting — val loss would start rising even as train loss drops.)*

**Verbal punctuation:**
> "Two epochs, answer-only loss. Now the part everyone cares about — does it work?"

---

## Slide 7 — Results: The Payoff

**Core idea:** Distillation works — and same-family distillation works dramatically better.

**[VISUAL: accuracy comparison chart — see `docs/distillation-pipeline.md`]**

| Model | MATH-500 Accuracy |
|---|---|
| Qwen3 0.6B (baseline) | 15.2% |
| + DeepSeek-R1 distillation | 33.6% |
| + Qwen3-235B distillation | **45.0%** |

**Cycle 1 — the surprise:**
> "We more than doubled accuracy using the cross-family teacher.
> But using a same-family teacher, we nearly tripled it from baseline."

**Cycle 2 — why it matters:**
> "45% accuracy on MATH-500 from a 0.6B model. That model fits on a phone."

**Verbal punctuation:**
> "But *why* does same-family do so much better? That's the last thing to understand."

---

## Slide 8 — Why Same-Family Distillation Wins

**Core idea (final cycle of the key insight):** Shared vocabulary and conventions make teacher targets easier to imitate.

- **Shared tokenizer:** same tokens for math symbols, LaTeX, reasoning phrases
- **Shared prompting conventions:** the student already "speaks the format"
- **Shared output style:** the student's priors match the teacher's targets

**Fence:**
> "This isn't about the teacher being stronger. A 235B same-family teacher
> outperforms a 671B cross-family teacher on a 0.6B student.
> The gap is compatibility, not capacity."

**Cycle 3 (closing the loop):**
> "Borrow the thinking from the right teacher, and you don't need to run a
> 671B model. You shrink the model without losing the reasoning."

---

## Slide 9 — Contributions

*(This slide stays on screen during the 20-minute discussion.)*

**What we covered today:**

1. Hard distillation = supervised fine-tuning on teacher-generated reasoning traces
2. Filtering by token length removes 44% of data — and the result still outperforms baselines
3. Answer-only loss is critical: the student learns to reason, not to reconstruct prompts
4. Same-family distillation (Qwen3-235B → Qwen3-0.6B): **15.2% → 45.0%** on MATH-500
5. The production recipe: RLVR to train the flagship teacher; distillation to deploy

*(Not "Thank you." Not "Questions?" Not "Future Directions.")*

---

## Final Words (choose one)

**Option A — Salute the group:**
> "We've now closed the loop — RLVR built the teacher, distillation ships
> the student. That's the full recipe this book has been building toward."

**Option B — Leave a provocation for discussion:**
> "The wild part? We threw away half the dataset and still got here.
> Imagine what happens when we stop throwing it away."

*(Once applause or discussion starts, you can mouth "thank you" — but don't make it the final act.)*

---

## Delivery Notes

- **Slides are condiments.** The notebook cells *are* the talk on slides 4–6. Let the code run live; don't paste it onto slides.
- **No laser pointer.** Point to the screen directly, or use inline annotations in the notebook.
- **Stand near your display.** Don't let a gap open between you and the screen.
- **Slow down on Slide 7.** The 45% number is the salient moment — pause after you say it.

---

## Winston's Star Check

- [x] **Symbol:** The `<think>...</think>` tag — a visible marker for "borrowed reasoning"
- [x] **Slogan:** *"Borrow the thinking, shrink the model"*
- [x] **Surprise:** Same-family teacher beats a model 3× larger; 44% of data discarded and accuracy still jumps
- [x] **Salient idea:** Same-family distillation as a compatibility story, not a scale story
- [x] **Story:** We took a 0.6B model scoring 15%. Fed it 6,695 traces from a larger same-family teacher. After 2 epochs, it scores 45%.
