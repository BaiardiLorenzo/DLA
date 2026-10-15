import torch
from torch.nn import functional as F


def fgsm(model, images, labels, epsilon, targeted=False):
    if epsilon < 0:
        raise ValueError("epsilon non può essere negativo")
    mode = model.training
    model.eval()
    try:
        with torch.enable_grad():
            x = images.detach().clone().requires_grad_(True)
            loss = F.cross_entropy(model(x), labels)
            gradient = torch.autograd.grad(loss, x)[0]
            direction = -1 if targeted else 1
            return (x + direction * epsilon * gradient.sign()).clamp(0, 1).detach()
    finally:
        model.train(mode)


def odin(model, images, temperature, epsilon):
    if temperature <= 0 or epsilon < 0:
        raise ValueError("ODIN richiede temperatura positiva ed epsilon non negativo")
    mode = model.training
    model.eval()
    try:
        with torch.enable_grad():
            x = images.detach().clone().requires_grad_(True)
            logits = model(x) / temperature
            predicted = logits.detach().argmax(1)
            loss = F.cross_entropy(logits, predicted)
            gradient = torch.autograd.grad(loss, x)[0]
            perturbed = (x - epsilon * gradient.sign()).clamp(0, 1)
        with torch.no_grad():
            return (model(perturbed) / temperature).softmax(1).max(1).values
    finally:
        model.train(mode)
