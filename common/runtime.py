"""Configurazione delle esecuzioni e salvataggio locale dei risultati."""

import argparse
import csv
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Il valore deve essere positivo")
    return value


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Il valore deve essere positivo e finito")
    return value


def parser(description, epochs=20, batch_size=64, training=True, checkpoint=True):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    if training:
        p.add_argument("--epochs", type=positive_int, default=epochs)
        p.add_argument("--lr", type=positive_float, default=1e-3)
    p.add_argument("--batch-size", type=positive_int, default=batch_size)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--output-dir", type=Path)
    if checkpoint:
        p.add_argument("--checkpoint", type=Path)
    return p


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile: installa la build GPU o usa --device cpu")
    return torch.device("cuda" if name == "cuda" or (name == "auto" and torch.cuda.is_available()) else "cpu")


def setup(args, lab, exercise):
    seed_everything(args.seed)
    args.data_dir = args.data_dir.resolve()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.data_dir / "huggingface"))
    name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
    out = (args.output_dir or ROOT / "outputs" / lab / exercise / name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        raise FileExistsError(f"Directory già utilizzata: {out}. Scegli un nuovo --output-dir")
    device = device_for(args.device)
    write_json(out / "config.json", {**vars(args), "resolved_device": str(device), "torch": torch.__version__,
                                     "python": sys.version, "torch_threads": torch.get_num_threads(),
                                     "cuda_runtime": torch.version.cuda,
                                     "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    print(f"Device: {device}; risultati: {out}")
    return device, out


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path, device="cpu", kind=None):
    if path is None or not Path(path).is_file():
        raise FileNotFoundError("Specificare --checkpoint con un checkpoint prodotto dall'esercizio prerequisito")
    state = torch.load(path, map_location=device, weights_only=True)
    if kind and state.get("kind") != kind:
        raise ValueError(f"Checkpoint incompatibile: atteso {kind}, ricevuto {state.get('kind')}")
    return state
