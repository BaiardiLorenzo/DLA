"""Classificazione irony/emoji su TweetEval con DistilBERT."""
from torch.utils.data import DataLoader

from common.runtime import parser, setup, load_checkpoint, write_json
from lab2.data import TextCollator
from lab2.models import TextClassifier
from lab2.training import fit_supervised, supervised_epoch


def main():
    p = parser(__doc__, epochs=5, batch_size=32)
    p.set_defaults(lr=None)
    p.add_argument("--task", choices=["irony", "emoji"], default="irony")
    p.add_argument("--model", default="distilbert/distilbert-base-uncased")
    p.add_argument("--fine-tune", action="store_true")
    p.add_argument("--max-length", type=int, default=128)
    args = p.parse_args()
    if args.max_length < 2 or args.max_length > 512:
        p.error("--max-length deve essere tra 2 e 512")
    if args.lr is None:
        args.lr = 2e-5 if args.fine_tune else 1e-3
    device, out = setup(args, "lab2", "ex3_1")
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer
    saved = load_checkpoint(args.checkpoint, device, "text_classifier") if args.checkpoint else None
    config = saved["config"] if saved else dict(model=args.model, task=args.task, frozen=not args.fine_tune, max_length=args.max_length)
    dataset = load_dataset("cardiffnlp/tweet_eval", config["task"], cache_dir=str(args.data_dir / "datasets"))
    config["classes"] = dataset["train"].features["label"].num_classes
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    collator = TextCollator(tokenizer, config["max_length"])
    loaders = [DataLoader(dataset[split], batch_size=args.batch_size, shuffle=split == "train", collate_fn=collator)
               for split in ("train", "validation", "test")]
    model = TextClassifier(AutoModel.from_pretrained(config["model"]), config["classes"], config["frozen"]).to(device)
    if saved:
        model.load_state_dict(saved["model"])
        write_json(out / "test_metrics.json", supervised_epoch(model, loaders[2], device))
    else:
        fit_supervised(model, loaders, args, device, out, "text_classifier", config)
    tokenizer.save_pretrained(out / "tokenizer")


if __name__ == "__main__":
    main()
