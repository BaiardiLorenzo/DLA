"""JARN: regolarizzazione avversaria dei Jacobiani con doppia backpropagation."""
from common.runtime import setup, write_json
from lab4.analysis import analyze_ood, attack_analysis, clean_accuracy
from lab4.cli import attack_arguments, ood_parser, validate_epsilons
from lab4.data import loaders, ood_loaders
from lab4.training import fit, load_model


def main():
    p = ood_parser(__doc__)
    attack_arguments(p)
    p.add_argument("--jarn-weight", type=float, default=0.01)
    p.add_argument("--noise-epsilon", type=float, default=0.03)
    args = p.parse_args()
    validate_epsilons(p, [*args.epsilons, args.noise_epsilon])
    if args.jarn_weight < 0:
        p.error("jarn-weight non può essere negativo")
    device, out = setup(args, "lab4", "ex3_2")
    baseline, regularized = load_model(args.checkpoint, device), load_model(args.checkpoint, device)
    data = loaders(args)
    fit(regularized, data, args, device, out / "training", mode="jarn")
    ood = ood_loaders(args, args.ood)[1]
    summary = {}
    for name, model in (("baseline", baseline), ("jarn", regularized)):
        summary[name] = dict(clean_accuracy=clean_accuracy(model, data[2], device),
                             attacks=attack_analysis(model, data[2], device, out / name / "attacks", args.epsilons, examples=args.examples),
                             ood=analyze_ood(model, data[2], ood, device, out / name / "ood"))
    write_json(out / "comparison.json", summary)


if __name__ == "__main__":
    main()
