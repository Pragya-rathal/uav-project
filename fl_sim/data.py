from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def load_mnist(data_root: str = "./data"):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    return train_set, test_set


def dirichlet_split_noniid(
    labels: torch.Tensor,
    num_devices: int,
    alpha: float,
    seed: int,
    min_size: int = 100,
) -> Dict[int, List[int]]:
    rng = np.random.default_rng(seed)
    labels_np = labels.numpy()
    num_classes = int(labels.max().item() + 1)

    while True:
        idx_by_client = {i: [] for i in range(num_devices)}
        for c in range(num_classes):
            idx_c = np.where(labels_np == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(np.repeat(alpha, num_devices))
            cut_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            splits = np.split(idx_c, cut_points)
            for i, split in enumerate(splits):
                idx_by_client[i].extend(split.tolist())
        sizes = [len(idx_by_client[i]) for i in range(num_devices)]
        if min(sizes) >= min_size:
            break

    for i in range(num_devices):
        rng.shuffle(idx_by_client[i])
    return idx_by_client


def make_client_loaders(
    train_set,
    client_indices: Dict[int, List[int]],
    batch_size: int,
    seed: int,
) -> Dict[int, DataLoader]:
    loaders = {}
    gen = torch.Generator().manual_seed(seed)
    for cid, idxs in client_indices.items():
        subset = Subset(train_set, idxs)
        loaders[cid] = DataLoader(
            subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, generator=gen
        )
    return loaders


def make_test_loader(test_set, batch_size: int = 256) -> DataLoader:
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


def summarize_client_distribution(labels: torch.Tensor, client_indices: Dict[int, List[int]]) -> Dict[int, Dict[int, int]]:
    summary = {}
    labels_np = labels.numpy()
    for cid, idxs in client_indices.items():
        hist = defaultdict(int)
        for idx in idxs:
            hist[int(labels_np[idx])] += 1
        summary[cid] = dict(hist)
    return summary
