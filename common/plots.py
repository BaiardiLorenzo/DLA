"""Grafici salvati su disco, utilizzabili anche senza desktop grafico."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def curves(rows, path, keys, xlabel="epoch", title=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in keys:
        samples = [row for row in rows if row.get(key) is not None]
        if samples:
            ax.plot([row[xlabel] for row in samples], [row[key] for row in samples], label=key)
    ax.set(xlabel=xlabel, title=title)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
