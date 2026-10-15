"""Multiple-choice su ARC-Easy o SWAG, con numero variabile di opzioni."""
from torch.utils.data import DataLoader

from common.runtime import load_checkpoint, parser, setup, write_json
from lab2.data import ChoiceCollator
from lab2.models import MultipleChoice
from lab2.training import fit_supervised, supervised_epoch


def main():
    p = parser(__doc__, epochs=4, batch_size=8)
    p.set_defaults(lr=2e-5)
    p.add_argument("--dataset", choices=["arc", "swag"], default="arc")
    p.add_argument("--model", default="google-bert/bert-base-uncased")
    p.add_argument("--frozen", action="store_true")
    p.add_argument("--max-length", type=int, default=192)
    args = p.parse_args()
    if not 2 <= args.max_length <= 512:
        p.error("--max-length deve essere tra 2 e 512")
    device, out = setup(args, "lab2", "ex3_2")
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer
    saved = load_checkpoint(args.checkpoint, device, "multiple_choice") if args.checkpoint else None
    config = saved["config"] if saved else dict(model=args.model, dataset=args.dataset, frozen=args.frozen, max_length=args.max_length)
    if config["dataset"] == "arc":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", cache_dir=str(args.data_dir / "datasets"))
        splits = ("train", "validation", "test")
    else:
        dataset = load_dataset("allenai/swag", "regular", cache_dir=str(args.data_dir / "datasets"))
        # Il test pubblico SWAG non ha risposte: validation resta lo holdout finale.
        partition = dataset["train"].train_test_split(test_size=0.1, seed=args.seed)
        dataset["train"], dataset["development"] = partition["train"], partition["test"]
        splits = ("train", "development", "validation")
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    collator = ChoiceCollator(tokenizer, config["dataset"], config["max_length"])
    loaders = [DataLoader(dataset[split], batch_size=args.batch_size, shuffle=i == 0, collate_fn=collator) for i, split in enumerate(splits)]
    model = MultipleChoice(AutoModel.from_pretrained(config["model"]), config["frozen"]).to(device)
    if saved:
        model.load_state_dict(saved["model"])
        write_json(out / "test_metrics.json", supervised_epoch(model, loaders[2], device))
    else:
        fit_supervised(model, loaders, args, device, out, "multiple_choice", config)
    tokenizer.save_pretrained(out / "tokenizer")


if __name__ == "__main__":
    main()
