"""Grad-CAM sul punteggio della classe, con hook rimossi dopo l'uso."""
import torch
from torch.nn import functional as F

from common.plots import plt
from common.runtime import load_checkpoint, parser, positive_int, setup
from lab1.data import image_loaders
from lab1.models import ImageCNN


def gradcam(model, image, target=None):
    activation = []
    handle = model.features[-2].register_forward_hook(lambda module, inputs, output: activation.append(output))
    try:
        logits = model(image)
        target = int(logits.argmax(1)[0]) if target is None else target
        features = activation[0]
        gradient = torch.autograd.grad(logits[0, target], features)[0]
        weights = gradient.mean((2, 3), keepdim=True)
        heatmap = (features * weights).sum(1, keepdim=True).relu()
        heatmap = F.interpolate(heatmap, image.shape[-2:], mode="bilinear", align_corners=False)
        heatmap = heatmap / heatmap.amax((2, 3), keepdim=True).clamp_min(1e-12)
        return heatmap[0, 0].detach().cpu(), target
    finally:
        handle.remove()


def main():
    p = parser(__doc__, training=False)
    p.add_argument("--samples", type=positive_int, default=8)
    p.add_argument("--class-id", type=int, choices=range(10))
    args = p.parse_args()
    device, out = setup(args, "lab1", "ex2_3")
    state = load_checkpoint(args.checkpoint, device, "image_cnn")
    model = ImageCNN(**state["config"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    dataset = image_loaders("cifar10", args.data_dir, args.batch_size, args.seed)[2].dataset
    mean = torch.tensor([0.4914, 0.4822, 0.4465])[:, None, None]
    std = torch.tensor([0.247, 0.243, 0.262])[:, None, None]
    count = min(args.samples, len(dataset))
    columns = min(4, count)
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(rows, 2 * columns, figsize=(5 * columns, 2.8 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i in range(count):
        image, label = dataset[i]
        heatmap, target = gradcam(model, image[None].to(device), args.class_id)
        rgb = (image * std + mean).clamp(0, 1).permute(1, 2, 0)
        original, overlay = axes[i // columns, 2 * (i % columns):2 * (i % columns) + 2]
        original.imshow(rgb)
        original.set_title(f"Vero: {dataset.classes[label]}")
        overlay.imshow(rgb)
        overlay.imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        overlay.set_title(f"Grad-CAM: {dataset.classes[target]}")
    fig.tight_layout()
    fig.savefig(out / "gradcam.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
