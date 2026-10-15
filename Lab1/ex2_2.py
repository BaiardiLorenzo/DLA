"""Network surgery di un classificatore MNIST e localizzazione su canvas."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from torchvision.ops import nms

from common.plots import plt
from common.runtime import load_checkpoint, parser, positive_int, setup, write_json
from lab1.data import image_loaders
from lab1.models import DigitCNN, FullyConvolutionalDigits
from lab1.training import fit


def main():
    p = parser(__doc__, epochs=10)
    p.add_argument("--digits", type=positive_int, default=9)
    p.add_argument("--confidence", type=float, default=0.8)
    p.add_argument("--canvas-run", type=Path, help="Riusa canvas e ground truth di una precedente esecuzione")
    args = p.parse_args()
    if not 0 < args.confidence < 1:
        p.error("--confidence deve essere tra 0 e 1")
    device, out = setup(args, "lab1", "ex2_2")
    loaders = image_loaders("mnist", args.data_dir, args.batch_size, args.seed, normalize=False)
    classifier = DigitCNN().to(device)
    if args.checkpoint:
        classifier.load_state_dict(load_checkpoint(args.checkpoint, device, "digit_cnn")["model"])
    else:
        fit(classifier, loaders, args, device, out / "classifier", "digit_cnn", {})
        classifier.load_state_dict(load_checkpoint(out / "classifier" / "best.pt", device, "digit_cnn")["model"])
    dense = FullyConvolutionalDigits(classifier).to(device).eval()
    side = int(args.digits ** 0.5 + 0.999)
    size = side * 40 + 8
    canvas = torch.zeros(1, size, size)
    truth = []
    dataset = loaders[2].dataset
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[:args.digits]
    for i, index in enumerate(indices):
        image, label = dataset[int(index)]
        x, y = 8 + (i % side) * 40, 8 + (i // side) * 40
        canvas[:, y:y + 28, x:x + 28] = image
        truth.append(dict(label=int(label), box=[x, y, x + 28, y + 28]))
    if args.canvas_run:
        canvas = torch.load(args.canvas_run / "class_maps.pt", map_location="cpu", weights_only=True)["canvas"]
        truth = json.loads((args.canvas_run / "detections.json").read_text(encoding="utf-8"))["truth"]
    with torch.no_grad():
        scores = dense(canvas[None].to(device)).softmax(1)[0].cpu()
    pixel_scores = F.interpolate(scores[None], size=canvas.shape[-2:], mode="bilinear", align_corners=False)[0]
    torch.save({"grid_scores": scores, "pixel_scores": pixel_scores, "canvas": canvas}, out / "class_maps.pt")
    confidence, labels = scores.max(0)
    foreground = F.avg_pool2d(canvas[None], dense.receptive_field, dense.stride)[0, 0]
    complete_ink = foreground >= 0.9 * F.max_pool2d(foreground[None, None], 7, stride=1, padding=3)[0, 0]
    coordinates = torch.arange(26, dtype=canvas.dtype)
    mass = foreground * (26 * 26)
    centroid_x = F.conv2d(canvas[None], coordinates[None, None, None].expand(1, 1, 26, 26), stride=4)[0, 0] / mass.clamp_min(1e-8)
    centroid_y = F.conv2d(canvas[None], coordinates[None, None, :, None].expand(1, 1, 26, 26), stride=4)[0, 0] / mass.clamp_min(1e-8)
    centrality = torch.exp(-((centroid_x - 13.5).square() + (centroid_y - 13.5).square()) / 18)
    # Il classificatore MNIST vede cifre centrate: penalizza finestre parziali
    # e decentrate senza usare le posizioni o le classi ground truth.
    ys, xs = torch.where((confidence >= args.confidence) & (foreground > 0.03) & complete_ink & (centrality >= 0.4))
    boxes = torch.stack([xs * 4, ys * 4, xs * 4 + 26, ys * 4 + 26], 1).float()
    strengths = confidence[ys, xs] * centrality[ys, xs]
    keep = nms(boxes, strengths, 0.25) if len(boxes) else []
    predictions = [dict(label=int(labels[ys[i], xs[i]]), confidence=float(confidence[ys[i], xs[i]]),
                        detection_score=float(strengths[i]), box=boxes[i].tolist()) for i in keep]
    matched = set()
    for prediction in predictions:
        px1, py1, px2, py2 = prediction["box"]
        for index, actual in enumerate(truth):
            if index in matched or prediction["label"] != actual["label"]:
                continue
            x1, y1, x2, y2 = actual["box"]
            intersection = max(0, min(px2, x2) - max(px1, x1)) * max(0, min(py2, y2) - max(py1, y1))
            union = (px2 - px1) * (py2 - py1) + (x2 - x1) * (y2 - y1) - intersection
            if intersection / union >= 0.5:
                matched.add(index)
                break
    write_json(out / "detections.json", {"truth": truth, "predictions": predictions,
                                         "precision_iou_0_5": len(matched) / len(predictions) if predictions else None,
                                         "recall_iou_0_5": len(matched) / len(truth),
                                         "note": "Un canvas sintetico; filtro di centratura/completezza del tratto, campo ricettivo 26, stride 4; nessuna classe background appresa."})
    torch.save({"kind": "digit_fcn", "config": {}, "model": dense.state_dict()}, out / "fcn.pt")
    fig, axes = plt.subplots(2, 6, figsize=(16, 6))
    axes[0, 0].imshow(canvas[0], cmap="gray")
    axes[0, 0].set_title("Canvas e localizzazioni")
    for item in predictions:
        x1, y1, x2, y2 = item["box"]
        axes[0, 0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red"))
        axes[0, 0].text(x1, y1, str(item["label"]), color="red")
    for digit, ax in zip(range(10), list(axes.flat)[1:11]):
        ax.imshow(pixel_scores[digit], cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"Classe {digit}")
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "localization.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
