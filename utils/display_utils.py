from IPython.display import display, Latex, Math

import matplotlib.pyplot as plt
import torch
 
def plot_scores_bar(
    next_token_logits, 
    start=19_800, 
    end=19_900,
    arrow=True, 
    ylabel="Logit value"
):
    # Select vocabulary subsection
    x = torch.arange(start, end)
    # .cpu() is a shortcut for to(torch.device("cpu"))
    logits_section = next_token_logits[0, start:end].float().cpu()

    # Plot the logits (scores) within the selected range
    plt.bar(x, logits_section)
    plt.xlabel("Vocabulary index")

    # Draw an arrow to highlight the largest score
    if arrow:
        max_idx = torch.argmax(logits_section)
        plt.annotate(
            "Berlin",
            xy=(x[max_idx], logits_section[max_idx]),
            xytext=(x[max_idx] - 25, logits_section[max_idx] - 2),
            arrowprops={
                "facecolor": "black", "arrowstyle": "->", "lw": 2.0
            },
            fontsize=8,
        )
 
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
 
# plot_scores_bar(next_token_logits)


import matplotlib.pyplot as plt

def plot_brevity_curve(brevity_bonus, max_len=2048):
    lengths = torch.arange(1, max_len)
    scores = 1.5 * torch.exp(-lengths / brevity_bonus)
 
    plt.figure(figsize=(4, 3))
    plt.plot(lengths, scores)
    plt.xlabel("Text length (number of characters)")
    plt.ylabel("Score contribution")
    plt.tight_layout()
    #plt.savefig("brevity_curve.pdf")
    plt.show()
 
# plot_brevity_curve(500)
 