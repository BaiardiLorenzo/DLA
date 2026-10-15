import numpy as np
import torch
from sklearn import metrics

from common.plots import plt
from common.runtime import write_csv, write_json
from lab4.attacks import fgsm, odin


def scores(model, loader, device, method="msp", temperature=1.0, epsilon=0.0):
    model.eval()
    result = []
    for images, _ in loader:
        images = images.to(device)
        if method == "odin":
            values = odin(model, images, temperature, epsilon)
        else:
            with torch.no_grad():
                logits = model(images)
                values = logits.max(1).values if method == "max_logit" else logits.softmax(1).max(1).values
        result.extend(values.detach().cpu().tolist())
    return np.asarray(result)


def ood_metrics(id_scores, ood_scores):
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    values = np.concatenate([id_scores, ood_scores])
    precision_id, recall_id, _ = metrics.precision_recall_curve(labels, values)
    precision_ood, recall_ood, _ = metrics.precision_recall_curve(1 - labels, -values)
    return dict(auroc_id=metrics.roc_auc_score(labels, values),
                aupr_id=metrics.auc(recall_id, precision_id), aupr_ood=metrics.auc(recall_ood, precision_ood),
                average_precision_id=metrics.average_precision_score(labels, values),
                average_precision_ood=metrics.average_precision_score(1 - labels, -values),
                id_samples=len(id_scores), ood_samples=len(ood_scores))


def analyze_ood(model, id_loader, ood_loader, device, out, method="msp", temperature=1.0, epsilon=0.0):
    out.mkdir(parents=True, exist_ok=True)
    id_scores = scores(model, id_loader, device, method, temperature, epsilon)
    ood_scores = scores(model, ood_loader, device, method, temperature, epsilon)
    result = ood_metrics(id_scores, ood_scores)
    write_json(out / "metrics.json", {**result, "score_direction": "maggiore = ID; score OOD = -score ID"})
    write_csv(out / "scores.csv", [dict(is_id=label, score=float(value)) for label, group in ((1, id_scores), (0, ood_scores)) for value in group])
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    values = np.concatenate([id_scores, ood_scores])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(id_scores, bins=40, density=True, alpha=0.5, label="ID")
    axes[0].hist(ood_scores, bins=40, density=True, alpha=0.5, label="OOD")
    axes[0].set(xlabel="Score ID", ylabel="Densità")
    fpr, tpr, _ = metrics.roc_curve(labels, values)
    axes[1].plot(fpr, tpr, label=f"AUROC ID={result['auroc_id']:.3f}")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate")
    for name, binary, score in (("ID", labels, values), ("OOD", 1 - labels, -values)):
        precision, recall, _ = metrics.precision_recall_curve(binary, score)
        axes[2].plot(recall, precision, label=name)
    axes[2].set(xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "ood.png", dpi=150)
    plt.close(fig)
    return result


@torch.no_grad()
def clean_accuracy(model, loader, device):
    model.eval()
    correct = count = 0
    for images, labels in loader:
        correct += (model(images.to(device)).argmax(1).cpu() == labels).sum().item()
        count += len(labels)
    return correct / count


def attack_analysis(model, loader, device, out, epsilons, target=None, examples=8):
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    rows = []
    for epsilon in epsilons:
        clean_correct = adversarial_correct = success = eligible = count = 0
        original_examples, attacked_examples, labels_examples, predictions_examples = [], [], [], []
        max_delta = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                clean_predictions = model(images).argmax(1)
            targets = torch.full_like(labels, target) if target is not None else labels
            attacked = fgsm(model, images, targets, epsilon, targeted=target is not None)
            with torch.no_grad():
                predictions = model(attacked).argmax(1)
            clean = clean_predictions == labels
            candidates = clean & (labels != target) if target is not None else clean
            outcome = predictions == target if target is not None else predictions != labels
            clean_correct += clean.sum().item()
            adversarial_correct += (predictions == labels).sum().item()
            eligible += candidates.sum().item()
            success += (outcome & candidates).sum().item()
            count += len(labels)
            max_delta = max(max_delta, (attacked - images).abs().max().item())
            take = min(examples - len(original_examples), len(labels))
            if take > 0:
                original_examples.extend(images[:take].detach().cpu())
                attacked_examples.extend(attacked[:take].cpu())
                labels_examples.extend(labels[:take].cpu().tolist())
                predictions_examples.extend(predictions[:take].cpu().tolist())
        rows.append(dict(epsilon=epsilon, clean_accuracy=clean_correct / count,
                         adversarial_accuracy=adversarial_correct / count,
                         attack_success_rate=success / eligible if eligible else None,
                         eligible_samples=eligible, max_pixel_delta=max_delta, target=target))
        fig, axes = plt.subplots(len(original_examples), 2, figsize=(6, 2.5 * len(original_examples)), squeeze=False)
        for i, (original, attacked) in enumerate(zip(original_examples, attacked_examples)):
            axes[i, 0].imshow(original.permute(1, 2, 0))
            axes[i, 0].set_title(f"Originale: {labels_examples[i]}")
            axes[i, 1].imshow(attacked.permute(1, 2, 0))
            axes[i, 1].set_title(f"Predizione: {predictions_examples[i]}, eps={epsilon}")
            for ax in axes[i]:
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"examples_eps_{epsilon}.png", dpi=150)
        plt.close(fig)
    write_csv(out / "attacks.csv", rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([row["epsilon"] for row in rows], [row["adversarial_accuracy"] for row in rows], label="Accuratezza sotto attacco")
    ax.plot([row["epsilon"] for row in rows], [row["clean_accuracy"] for row in rows], label="Accuratezza clean")
    ax.set(xlabel="Epsilon in [0,1]", ylabel="Accuratezza", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "robustness.png", dpi=150)
    plt.close(fig)
    return rows
