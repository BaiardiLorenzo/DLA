"""Confronto quantitativo del flusso dei gradienti con e senza residuali."""
from common.plots import plt
from common.runtime import parser, positive_int, setup
from lab1.ex1_2 import run_comparison


def main():
    p = parser(__doc__, epochs=30, batch_size=128, checkpoint=False)
    p.add_argument("--blocks", type=positive_int, nargs="+", default=[2, 4])
    args = p.parse_args()
    device, out = setup(args, "lab1", "ex2_1")
    runs = run_comparison(args, device, out, gradients=True)
    for number in (1, args.epochs):
        fig, ax = plt.subplots(figsize=(12, 5))
        for name, rows in runs.items():
            selected = [row for row in rows if row["epoch"] == number]
            ax.plot(range(len(selected)), [max(row["mean_abs_gradient"], 1e-15) for row in selected], label=name)
        ax.set(xlabel="Strato con pesi (ordine forward)", ylabel="Media |gradiente|", yscale="log", title=f"Epoca {number}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"gradients_epoch_{number}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
