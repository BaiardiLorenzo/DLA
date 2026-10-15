"""Confronto di MLP di profondità diversa su MNIST."""
from common.plots import curves
from common.runtime import parser, positive_int, seed_everything, setup, write_csv
from lab1.data import image_loaders
from lab1.models import MLP
from lab1.training import fit


def main():
    p = parser(__doc__, checkpoint=False)
    p.add_argument("--depths", type=positive_int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--width", type=positive_int, default=64)
    args = p.parse_args()
    device, out = setup(args, "lab1", "ex1_1")
    comparisons, summary = [], []
    for depth in args.depths:
        seed_everything(args.seed)
        loaders = image_loaders("mnist", args.data_dir, args.batch_size, args.seed)
        config = dict(width=args.width, depth=depth)
        history, _ = fit(MLP(**config).to(device), loaders, args, device, out / f"depth_{depth}", "mlp", config)
        comparisons.extend(dict(epoch=row["epoch"], **{f"depth_{depth}": row["val_accuracy"]}) for row in history)
        summary.append(dict(depth=depth, best_val_accuracy=max(row["val_accuracy"] for row in history)))
    curves(comparisons, out / "comparison.png", [f"depth_{depth}" for depth in args.depths], title="MLP MNIST: validation")
    write_csv(out / "comparison.csv", summary)


if __name__ == "__main__":
    main()
