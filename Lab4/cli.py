from common.runtime import parser, positive_int


def ood_parser(description, training=True):
    p = parser(description, epochs=20, batch_size=128, training=training)
    p.add_argument("--ood", choices=["fake", "cifar100"], default="cifar100")
    p.add_argument("--ood-size", type=positive_int, default=5000)
    return p


def attack_arguments(p):
    p.add_argument("--epsilons", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.1])
    p.add_argument("--examples", type=positive_int, default=8)


def validate_epsilons(p, epsilons):
    if any(value < 0 or value > 1 for value in epsilons):
        p.error("Epsilon deve essere tra 0 e 1 nello spazio dei pixel")
