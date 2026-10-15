"""FGSM mirato: minimizza la loss della classe bersaglio."""
from common.runtime import setup
from lab4.analysis import attack_analysis
from lab4.cli import attack_arguments, ood_parser, validate_epsilons
from lab4.data import loaders
from lab4.training import load_model


def main():
    p = ood_parser(__doc__, training=False)
    attack_arguments(p)
    p.add_argument("--target", type=int, choices=range(10), default=0)
    args = p.parse_args()
    validate_epsilons(p, args.epsilons)
    device, out = setup(args, "lab4", "ex3_3")
    model = load_model(args.checkpoint, device)
    attack_analysis(model, loaders(args)[2], device, out, args.epsilons, target=args.target, examples=args.examples)


if __name__ == "__main__":
    main()
