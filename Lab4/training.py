import torch
from torch.nn import functional as F
from tqdm import tqdm

from common.plots import curves
from common.runtime import load_checkpoint, write_csv
from lab4.attacks import fgsm
from lab4.models import Adaptor, CifarCNN, Discriminator


def load_model(checkpoint, device):
    saved = load_checkpoint(checkpoint, device, "ood_cnn")
    model = CifarCNN().to(device)
    model.load_state_dict(saved["model"])
    return model


def classification_epoch(model, loader, device, optimizer=None, epsilon=None):
    model.train(optimizer is not None)
    total = correct = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if optimizer:
                if epsilon is not None:
                    adversarial = fgsm(model, images, labels, epsilon)
                    loss = 0.5 * (loss + F.cross_entropy(model(adversarial), labels))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
    return total / count, correct / count


def jarn_epoch(model, adaptor, discriminator, loader, device, optimizer, adaptor_optimizer, disc_optimizer,
               coefficient, noise_epsilon):
    model.train()
    adaptor.train()
    discriminator.train()
    loss_sum = adversarial_sum = correct = count = 0
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        noisy = (images + torch.empty_like(images).uniform_(-noise_epsilon, noise_epsilon)).clamp(0, 1).detach().requires_grad_(True)
        logits = model(noisy)
        # Somma sui campioni: il Jacobiano di ciascuna immagine non cambia con batch_size.
        per_example = F.cross_entropy(logits, labels, reduction="none")
        jacobian = torch.autograd.grad(per_example.sum(), noisy, create_graph=True)[0]
        generated = adaptor(jacobian)
        real = 2 * noisy.detach() - 1  # dominio [-1,1], coerente con tanh dell'adattatore

        # Discriminatore: immagini reali vs Jacobiani adattati, gradienti isolati.
        discriminator.requires_grad_(True)
        real_logits, fake_logits = discriminator(real), discriminator(generated.detach())
        disc_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        disc_loss += F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        disc_optimizer.zero_grad(set_to_none=True)
        disc_loss.backward()
        disc_optimizer.step()

        # Classificatore e adattatore minimizzano log(1-D(J')): obiettivo minimax del paper.
        # Softplus(-logit) è -log(sigmoid(logit)); qui serve log(1-sigmoid(logit)).
        discriminator.requires_grad_(False)
        fake_logits = discriminator(generated)
        adversarial_loss = -F.softplus(fake_logits).mean()
        loss = per_example.mean() + coefficient * adversarial_loss
        optimizer.zero_grad(set_to_none=True)
        adaptor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        adaptor_optimizer.step()
        discriminator.requires_grad_(True)
        loss_sum += per_example.mean().item() * len(labels)
        adversarial_sum += adversarial_loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        count += len(labels)
    return loss_sum / count, correct / count, adversarial_sum / count


def fit(model, loaders, args, device, out, mode="clean", snapshots=()):
    out.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    adaptor = discriminator = None
    if mode == "jarn":
        adaptor, discriminator = Adaptor().to(device), Discriminator().to(device)
        adaptor_optimizer = torch.optim.Adam(adaptor.parameters(), lr=args.lr)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    rows, best = [], float("inf")
    for number in range(1, args.epochs + 1):
        if mode == "jarn":
            train_loss, train_accuracy, regularizer = jarn_epoch(
                model, adaptor, discriminator, loaders[0], device, optimizer, adaptor_optimizer,
                disc_optimizer, args.jarn_weight, args.noise_epsilon)
        else:
            train_loss, train_accuracy = classification_epoch(model, loaders[0], device, optimizer,
                                                               args.train_epsilon if mode == "adversarial" else None)
            regularizer = None
        val_loss, val_accuracy = classification_epoch(model, loaders[1], device)
        row = dict(epoch=number, train_loss=train_loss, train_accuracy=train_accuracy,
                   val_loss=val_loss, val_accuracy=val_accuracy, jarn_regularizer=regularizer)
        rows.append(row)
        print(row)
        write_csv(out / "metrics.csv", rows)
        state = dict(kind="ood_cnn", config={"training": mode}, model=model.state_dict(), epoch=number)
        if adaptor is not None:
            state.update(adaptor=adaptor.state_dict(), discriminator=discriminator.state_dict())
        if val_loss < best:
            best = val_loss
            torch.save(state, out / "best.pt")
        if number in snapshots or number == args.epochs:
            torch.save(state, out / f"epoch_{number}.pt")
    curves(rows, out / "loss.png", ["train_loss", "val_loss"])
    curves(rows, out / "accuracy.png", ["train_accuracy", "val_accuracy"])
    model.load_state_dict(load_checkpoint(out / "best.pt", device, "ood_cnn")["model"])
    return rows
