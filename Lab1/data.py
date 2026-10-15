import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from common.images import cifar_dataset


def image_loaders(name, root, batch_size, seed, normalize=True):
    transform = transforms.ToTensor()
    if normalize:
        mean, std = ((0.1307,), (0.3081,)) if name == "mnist" else ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262))
        transform = transforms.Compose([transform, transforms.Normalize(mean, std)])
    if name == "mnist":
        train = datasets.MNIST(root, train=True, download=True, transform=transform)
        test = datasets.MNIST(root, train=False, download=True, transform=transform)
    else:
        train = cifar_dataset("cifar10", root, True, transform)
        test = cifar_dataset("cifar10", root, False, transform)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed)).tolist()
    cut = int(0.9 * len(order))
    training, validation = Subset(train, order[:cut]), Subset(train, order[cut:])
    return (
        DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=0,
                   generator=torch.Generator().manual_seed(seed)),
        DataLoader(validation, batch_size=batch_size, num_workers=0),
        DataLoader(test, batch_size=batch_size, num_workers=0),
    )
