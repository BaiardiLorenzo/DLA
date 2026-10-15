"""CNN CIFAR-10: comportamento OOD a diverse epoche con FakeData e CIFAR-100."""
from common.runtime import positive_int, setup, write_csv, write_json
from lab4.analysis import analyze_ood
from lab4.cli import ood_parser
from lab4.data import OOD_CLASSES, loaders, ood_loaders
from lab4.models import CifarCNN
from lab4.training import fit, load_model


def main():
    p = ood_parser(__doc__)
    p.add_argument("--snapshot-epochs", type=positive_int, nargs="+", default=[1, 5, 10, 20])
    args = p.parse_args()
    device, out = setup(args, "lab4", "intro")
    data = loaders(args)
    if args.checkpoint:
        paths = [("loaded", args.checkpoint)]
    else:
        fit(CifarCNN().to(device), data, args, device, out / "training", snapshots=args.snapshot_epochs)
        epochs = sorted(set([number for number in args.snapshot_epochs if number <= args.epochs] + [args.epochs]))
        paths = [(str(number), out / "training" / f"epoch_{number}.pt") for number in epochs]
    summary = []
    for source in ("fake", "cifar100"):
        ood = ood_loaders(args, source)[1]
        for epoch, path in paths:
            model = load_model(path, device)
            for method in ("max_logit", "msp"):
                metrics = analyze_ood(model, data[2], ood, device, out / source / f"epoch_{epoch}" / method, method)
                summary.append(dict(epoch=epoch, source=source, method=method, **metrics))
    write_csv(out / "epoch_comparison.csv", summary)
    write_json(out / "cifar100_classes.json", OOD_CLASSES)


if __name__ == "__main__":
    main()
