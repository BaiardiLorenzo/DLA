"""GPT a caratteri per l'Inferno di Dante, scritto in PyTorch."""
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import trange

from common.plots import curves
from common.runtime import ROOT, load_checkpoint, parser, positive_int, setup, write_csv, write_json
from lab2.data import character_batch, clean_dante
from lab2.models import TinyGPT


def main():
    p = parser(__doc__, epochs=10)
    p.set_defaults(lr=3e-4)
    p.add_argument("--corpus", type=Path, default=ROOT / "archive" / "Lab2" / "commedia.txt")
    p.add_argument("--block-size", type=positive_int, default=128)
    p.add_argument("--width", type=positive_int, default=192)
    p.add_argument("--heads", type=positive_int, default=6)
    p.add_argument("--layers", type=positive_int, default=4)
    p.add_argument("--steps-per-epoch", type=positive_int, default=500)
    p.add_argument("--val-batches", type=positive_int, default=50)
    p.add_argument("--generate", type=positive_int, default=1000)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--prompt", default="Nel mezzo del cammin ")
    args = p.parse_args()
    if args.width % args.heads or args.temperature <= 0:
        p.error("--width deve essere divisibile per --heads e --temperature deve essere positiva")
    device, out = setup(args, "lab2", "ex1")
    if args.checkpoint:
        saved = load_checkpoint(args.checkpoint, device, "tiny_gpt")
        vocabulary, config = saved["vocabulary"], saved["config"]
        model = TinyGPT(**config).to(device)
        model.load_state_dict(saved["model"])
    else:
        text = clean_dante(args.corpus)
        (out / "corpus_clean.txt").write_text(text, encoding="utf-8")
        vocabulary = sorted(set(text))
        encoder = {char: i for i, char in enumerate(vocabulary)}
        tokens = torch.tensor([encoder[char] for char in text], dtype=torch.long)
        cut = int(0.9 * len(tokens))
        training, validation = tokens[:cut], tokens[cut:]
        if min(len(training), len(validation)) <= args.block_size:
            p.error("Corpus troppo piccolo per --block-size")
        config = dict(vocab_size=len(vocabulary), block_size=args.block_size, width=args.width,
                      heads=args.heads, layers=args.layers, dropout=0.1)
        model = TinyGPT(**config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        generator = torch.Generator().manual_seed(args.seed)
        rows, best = [], float("inf")
        for number in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            for _ in trange(args.steps_per_epoch, desc=f"Epoca {number}"):
                x, y = character_batch(training, args.batch_size, args.block_size, generator)
                logits = model(x.to(device))
                loss = F.cross_entropy(logits.reshape(-1, len(vocabulary)), y.to(device).reshape(-1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.item()
            model.eval()
            val_total = 0.0
            val_generator = torch.Generator().manual_seed(args.seed + 1)
            with torch.no_grad():
                for _ in range(args.val_batches):
                    x, y = character_batch(validation, args.batch_size, args.block_size, val_generator)
                    val_total += F.cross_entropy(model(x.to(device)).reshape(-1, len(vocabulary)), y.to(device).reshape(-1)).item()
            row = dict(epoch=number, train_loss=total / args.steps_per_epoch, val_loss=val_total / args.val_batches)
            rows.append(row)
            print(row)
            write_csv(out / "metrics.csv", rows)
            if row["val_loss"] < best:
                best = row["val_loss"]
                torch.save(dict(kind="tiny_gpt", config=config, vocabulary=vocabulary, model=model.state_dict()), out / "best.pt")
        curves(rows, out / "loss.png", ["train_loss", "val_loss"])
        model.load_state_dict(load_checkpoint(out / "best.pt", device, "tiny_gpt")["model"])
    encoder = {char: i for i, char in enumerate(vocabulary)}
    missing = sorted(set(args.prompt) - set(vocabulary))
    if missing or not args.prompt:
        p.error(f"Prompt vuoto o caratteri fuori vocabolario: {missing}")
    prompt = torch.tensor([[encoder[char] for char in args.prompt]], device=device)
    result = model.generate(prompt, args.generate, args.temperature)[0].tolist()
    generated = "".join(vocabulary[index] for index in result)
    (out / "generated.txt").write_text(generated, encoding="utf-8")
    write_json(out / "vocabulary.json", vocabulary)
    print(generated)


if __name__ == "__main__":
    main()
