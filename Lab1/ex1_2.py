"""Confronto di CNN plain e residuali su CIFAR-10."""
from common.plots import curves
from common.runtime import parser, positive_int, seed_everything, setup, write_csv
from lab1.data import image_loaders
from lab1.models import ImageCNN
from lab1.training import fit


def run_comparison(args, device, out, gradients=False):
    rows, summary, gradients_by_model = [], [], {}
    for blocks in args.blocks:
        for residual in (False, True):
            name = f"{'residual' if residual else 'plain'}_{blocks}"
            seed_everything(args.seed)
            loaders = image_loaders("cifar10", args.data_dir, args.batch_size, args.seed)
            config = dict(blocks=blocks, residual=residual)
            history, gradient_rows = fit(ImageCNN(**config).to(device), loaders, args, device,
                                        out / name, "image_cnn", config, gradients)
            rows.extend(dict(epoch=row["epoch"], **{name: row["val_accuracy"]}) for row in history)
            summary.append(dict(model=name, convolutions=3 + 6 * blocks,
                                best_val_accuracy=max(row["val_accuracy"] for row in history),
                                final_train_loss=history[-1]["train_loss"]))
            gradients_by_model[name] = gradient_rows
    write_csv(out / "comparison.csv", summary)
    curves(rows, out / "comparison.png", [row["model"] for row in summary], title="CNN CIFAR-10: validation")
    return gradients_by_model


def main():
    p = parser(__doc__, epochs=30, batch_size=128, checkpoint=False)
    p.add_argument("--blocks", type=positive_int, nargs="+", default=[1, 2, 4])
    args = p.parse_args()
    device, out = setup(args, "lab1", "ex1_2")
    run_comparison(args, device, out)


if __name__ == "__main__":
    main()
