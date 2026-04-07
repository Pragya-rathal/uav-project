from typing import Dict, Tuple

import torch
from torch import nn

from fl_sim.utils import clone_state_dict


def local_train(
    model: nn.Module,
    global_state: Dict[str, torch.Tensor],
    loader,
    device: torch.device,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> Tuple[Dict[str, torch.Tensor], float]:
    model.load_state_dict(global_state)
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_batches = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1

    state = clone_state_dict(model.state_dict())
    return state, total_loss / max(1, total_batches)


@torch.no_grad()
def evaluate(model: nn.Module, state_dict: Dict[str, torch.Tensor], test_loader, device: torch.device):
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for x, y in test_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

    return total_loss / total_samples, total_correct / total_samples
