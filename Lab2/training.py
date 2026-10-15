import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from common.plots import curves
from common.runtime import write_csv, write_json


def supervised_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total, count, truth, predictions = 0.0, 0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for raw in tqdm(loader, leave=False):
            batch = {key: value.to(device) for key, value in raw.items()}
            labels = batch.pop("labels")
            logits = model(**batch)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item() * len(labels)
            count += len(labels)
            truth.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(-1).detach().cpu().tolist())
    if not count:
        raise ValueError("Lo split è vuoto")
    return dict(loss=total / count, accuracy=accuracy_score(truth, predictions),
                macro_f1=f1_score(truth, predictions, average="macro", zero_division=0))


def fit_supervised(model, loaders, args, device, out, kind, config):
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr)
    best, rows = float("inf"), []
    for number in range(1, args.epochs + 1):
        training = supervised_epoch(model, loaders[0], device, optimizer)
        validation = supervised_epoch(model, loaders[1], device)
        row = dict(epoch=number, **{f"train_{k}": v for k, v in training.items()}, **{f"val_{k}": v for k, v in validation.items()})
        rows.append(row)
        print(row)
        write_csv(out / "metrics.csv", rows)
        if validation["loss"] < best:
            best = validation["loss"]
            torch.save({"kind": kind, "config": config, "model": model.state_dict(), "epoch": number}, out / "best.pt")
    state = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    if loaders[2] is not None:
        write_json(out / "test_metrics.json", supervised_epoch(model, loaders[2], device))
    curves(rows, out / "loss.png", ["train_loss", "val_loss"])
    curves(rows, out / "accuracy.png", ["train_accuracy", "val_accuracy"])
    return rows
