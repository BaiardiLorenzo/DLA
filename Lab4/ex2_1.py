"""FGSM non mirato: esempi e accuratezza in funzione di epsilon."""
from common.runtime import setup
from lab4.analysis import attack_analysis
from lab4.cli import attack_arguments, ood_parser, validate_epsilons
from lab4.data import loaders
from lab4.training import load_model


def main():
    p = ood_parser(__doc__, training=False)
    attack_arguments(p)
    args = p.parse_args()
    validate_epsilons(p, args.epsilons)
    device, out = setup(args, "lab4", "ex2_1")
    model = load_model(args.checkpoint, device)
    attack_analysis(model, loaders(args)[2], device, out, args.epsilons, examples=args.examples)


if __name__ == "__main__":
    main()
