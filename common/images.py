"""CIFAR originale se già presente; copia pubblica HF per il download."""
import numpy as np
from filelock import FileLock
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets


class CifarMirror(Dataset):
    def __init__(self, name, root, train, transform):
        from datasets import load_dataset
        config = "plain_text" if name == "cifar10" else "cifar100"
        self.label_key = "label" if name == "cifar10" else "fine_label"
        self.store = load_dataset(f"uoft-cs/{name}", config, split="train" if train else "test",
                                  cache_dir=str(root / "datasets"))
        self.transform = transform
        self.targets = list(self.store[self.label_key])
        self.classes = self.store.features[self.label_key].names
        self.class_to_idx = {label: index for index, label in enumerate(self.classes)}
        self.source = f"huggingface:uoft-cs/{name}"
        # Decodifica Arrow/PNG una volta; gli epoch successivi leggono RGB uint8.
        # Lock e rename atomico evitano cache parziali con due code di training.
        cache = root / "cifar_rgb" / name
        cache.mkdir(parents=True, exist_ok=True)
        split = "train" if train else "test"
        path = cache / f"{split}.npy"
        with FileLock(str(path) + ".lock"):
            if not path.exists():
                temporary = cache / f"{split}.tmp.npy"
                images = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8,
                                                   shape=(len(self.store), 32, 32, 3))
                for index, item in enumerate(self.store):
                    images[index] = np.asarray(item["img"].convert("RGB"), dtype=np.uint8)
                images.flush()
                del images
                temporary.replace(path)
        self.images = np.load(path, mmap_mode="r", allow_pickle=False)
        if self.images.shape != (len(self.store), 32, 32, 3) or self.images.dtype != np.uint8:
            raise ValueError(f"Cache CIFAR non valida: {path}")
        print(f"Dataset {self.source}: {'train' if train else 'test'}, {len(self)} immagini", flush=True)

    def __len__(self):
        return len(self.store)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index])
        return self.transform(image) if self.transform else image, int(self.targets[index])


def cifar_dataset(name, root, train, transform):
    factory = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    try:
        return factory(root, train=train, download=False, transform=transform)
    except RuntimeError as error:
        if "Dataset not found" not in str(error):
            raise
        return CifarMirror(name, root, train, transform)
