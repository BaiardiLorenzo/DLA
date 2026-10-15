import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from common.images import cifar_dataset

# Classi nominalmente distinte dalle dieci categorie CIFAR-10.
# La distinzione semantica non implica separazione perfetta delle distribuzioni.
OOD_CLASSES = ["apple", "mushroom", "orchid", "orange", "pear", "rose", "sunflower", "tulip",
               "maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree", "bottle", "bowl",
               "can", "cup", "plate", "clock", "keyboard", "lamp", "telephone", "television",
               "bed", "chair", "couch", "table", "wardrobe"]


def loaders(args):
    transform = transforms.ToTensor()
    training = cifar_dataset("cifar10", args.data_dir, True, transform)
    testing = cifar_dataset("cifar10", args.data_dir, False, transform)
    order = torch.randperm(len(training), generator=torch.Generator().manual_seed(args.seed)).tolist()
    cut = int(0.9 * len(order))
    split_train, split_val = Subset(training, order[:cut]), Subset(training, order[cut:])
    train_loader = DataLoader(split_train, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(split_val, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(testing, batch_size=args.batch_size, num_workers=0)
    return train_loader, val_loader, test_loader


def ood_loaders(args, source):
    if source == "fake":
        validation = datasets.FakeData(size=args.ood_size, image_size=(3, 32, 32), num_classes=10,
                                       transform=transforms.ToTensor(), random_offset=args.seed)
        testing = datasets.FakeData(size=args.ood_size, image_size=(3, 32, 32), num_classes=10,
                                    transform=transforms.ToTensor(), random_offset=args.seed + args.ood_size)
    else:
        validation = cifar_dataset("cifar100", args.data_dir, True, transforms.ToTensor())
        testing = cifar_dataset("cifar100", args.data_dir, False, transforms.ToTensor())
        class_ids = {validation.class_to_idx[name] for name in OOD_CLASSES}
        def selected(dataset, seed):
            indices = [i for i, label in enumerate(dataset.targets) if label in class_ids]
            order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed)).tolist()
            return Subset(dataset, [indices[i] for i in order[:args.ood_size]])
        validation, testing = selected(validation, args.seed), selected(testing, args.seed + 1)
    return tuple(DataLoader(dataset, batch_size=args.batch_size, num_workers=0) for dataset in (validation, testing))
