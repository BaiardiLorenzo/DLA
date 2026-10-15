"""ODIN: selezione T/epsilon sulla validation, valutazione finale sul test."""
from common.runtime import setup, write_csv, write_json
from lab4.analysis import analyze_ood, ood_metrics, scores
from lab4.cli import ood_parser, validate_epsilons
from lab4.data import loaders, ood_loaders
from lab4.training import load_model


def main():
    p = ood_parser(__doc__, training=False)
    p.add_argument("--temperatures", nargs="+", type=float, default=[1, 10, 100, 1000])
    p.add_argument("--epsilons", nargs="+", type=float, default=[0, 0.001, 0.002, 0.004])
    args = p.parse_args()
    validate_epsilons(p, args.epsilons)
    if any(value <= 0 for value in args.temperatures):
        p.error("Temperature positive richieste")
    device, out = setup(args, "lab4", "ex3_1")
    model = load_model(args.checkpoint, device)
    data = loaders(args)
    ood_validation, ood_test = ood_loaders(args, args.ood)
    grid = []
    for temperature in args.temperatures:
        for epsilon in args.epsilons:
            id_scores = scores(model, data[1], device, "odin", temperature, epsilon)
            ood_scores = scores(model, ood_validation, device, "odin", temperature, epsilon)
            row = dict(temperature=temperature, epsilon=epsilon, **ood_metrics(id_scores, ood_scores))
            grid.append(row)
            print(row)
            write_csv(out / "validation_grid.csv", grid)
    best = max(grid, key=lambda row: row["auroc_id"])
    write_json(out / "selected.json", best)
    analyze_ood(model, data[2], ood_test, device, out / "test", "odin", best["temperature"], best["epsilon"])
    analyze_ood(model, data[2], ood_test, device, out / "baseline_msp", "msp")


if __name__ == "__main__":
    main()
