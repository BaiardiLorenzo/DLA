import torch
from torch import nn
from tqdm import tqdm

from common.plots import curves
from common.runtime import write_csv


def epoch(model, loader, device, optimizer=None, gradients=None):
    model.train(optimizer is not None)
    loss_sum = correct = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if gradients is not None:
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None and parameter.ndim > 1:
                            gradients.setdefault(name, []).append(parameter.grad.detach().abs().mean())
                optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)
    return loss_sum / count, correct / count


def fit(model, loaders, args, device, out, kind, config, track_gradients=False):
    out.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history, gradient_rows = [], []
    best = float("inf")
    for number in range(1, args.epochs + 1):
        gradients = {} if track_gradients else None
        train_loss, train_acc = epoch(model, loaders[0], device, optimizer, gradients)
        val_loss, val_acc = epoch(model, loaders[1], device)
        row = dict(epoch=number, train_loss=train_loss, train_accuracy=train_acc, val_loss=val_loss, val_accuracy=val_acc)
        history.append(row)
        print(row)
        write_csv(out / "metrics.csv", history)
        if gradients is not None:
            gradient_rows.extend(dict(epoch=number, layer=layer, mean_abs_gradient=torch.stack(values).mean().item())
                                 for layer, values in gradients.items())
            write_csv(out / "gradients.csv", gradient_rows)
        if val_loss < best:
            best = val_loss
            torch.save({"kind": kind, "config": config, "model": model.state_dict(), "epoch": number}, out / "best.pt")
    curves(history, out / "loss.png", ["train_loss", "val_loss"])
    curves(history, out / "accuracy.png", ["train_accuracy", "val_accuracy"])
    return history, gradient_rows
