"""Pipeline OOD CIFAR-10 con score continui e istogrammi."""
from common.runtime import setup, write_json
from lab4.analysis import analyze_ood, clean_accuracy
from lab4.cli import ood_parser
from lab4.data import OOD_CLASSES, loaders, ood_loaders
from lab4.models import CifarCNN
from lab4.training import fit, load_model


def main(exercise="ex1_1"):
    p = ood_parser(__doc__)
    args = p.parse_args()
    device, out = setup(args, "lab4", exercise)
    data = loaders(args)
    if args.checkpoint:
        model = load_model(args.checkpoint, device)
    else:
        model = CifarCNN().to(device)
        fit(model, data, args, device, out / "training")
    ood = ood_loaders(args, args.ood)[1]
    results = {method: analyze_ood(model, data[2], ood, device, out / method, method) for method in ("max_logit", "msp")}
    results["clean_accuracy"] = clean_accuracy(model, data[2], device)
    results["ood_source"] = args.ood
    if args.ood == "cifar100":
        results["ood_classes"] = OOD_CLASSES
    write_json(out / "summary.json", results)


if __name__ == "__main__":
    main()
