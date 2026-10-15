"""Adversarial training FGSM online e confronto con il modello clean."""
from common.runtime import seed_everything, setup, write_json
from lab4.analysis import analyze_ood, attack_analysis, clean_accuracy
from lab4.cli import attack_arguments, ood_parser, validate_epsilons
from lab4.data import loaders, ood_loaders
from lab4.training import fit, load_model


def main():
    p = ood_parser(__doc__)
    attack_arguments(p)
    p.add_argument("--mode", choices=["adversarial", "clean"], default="adversarial",
                   help="clean produce il controllo con le stesse epoche aggiuntive")
    p.add_argument("--train-epsilon", type=float, default=0.03)
    args = p.parse_args()
    validate_epsilons(p, [*args.epsilons, args.train_epsilon])
    device, out = setup(args, "lab4", "ex2_2")
    baseline = load_model(args.checkpoint, device)
    data = loaders(args)
    robust = load_model(args.checkpoint, device)
    seed_everything(args.seed)
    fit(robust, data, args, device, out / "training", mode=args.mode)
    ood = ood_loaders(args, args.ood)[1]
    summary = {}
    for name, model in (("baseline", baseline), (args.mode, robust)):
        summary[name] = dict(clean_accuracy=clean_accuracy(model, data[2], device),
                             attacks=attack_analysis(model, data[2], device, out / name / "attacks", args.epsilons, examples=args.examples),
                             ood=analyze_ood(model, data[2], ood, device, out / name / "ood"))
    write_json(out / "comparison.json", summary)


if __name__ == "__main__":
    main()
