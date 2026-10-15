"""Valutazione dei classificatori Lab 1 sul test ufficiale separato."""
from pathlib import Path

from common.runtime import load_checkpoint, parser, setup, write_csv
from lab1.data import image_loaders
from lab1.models import DigitCNN, ImageCNN, MLP
from lab1.training import epoch


def main():
    p = parser(__doc__, training=False)
    args = p.parse_args()
    if args.checkpoint is None:
        p.error("--checkpoint deve essere un file o una directory con best.pt")
    device, out = setup(args, "lab1", "evaluation")
    paths = [args.checkpoint] if args.checkpoint.is_file() else sorted(args.checkpoint.rglob("best.pt"))
    if not paths:
        p.error("Nessun best.pt trovato")
    classes = {"mlp": MLP, "image_cnn": ImageCNN, "digit_cnn": DigitCNN}
    rows = []
    for path in paths:
        saved = load_checkpoint(path, device)
        kind = saved["kind"]
        if kind not in classes:
            p.error(f"Classificatore Lab1 non supportato: {kind}")
        model = classes[kind](**saved["config"]).to(device)
        model.load_state_dict(saved["model"])
        dataset = "cifar10" if kind == "image_cnn" else "mnist"
        loader = image_loaders(dataset, args.data_dir, args.batch_size, args.seed, normalize=kind != "digit_cnn")[2]
        loss, accuracy = epoch(model, loader, device)
        row = dict(checkpoint=str(path), dataset=dataset, test_samples=len(loader.dataset), loss=loss, accuracy=accuracy)
        rows.append(row)
        print(row)
        write_csv(out / "test_metrics.csv", rows)


if __name__ == "__main__":
    main()
