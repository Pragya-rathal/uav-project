from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch


class Compressor(ABC):
    @abstractmethod
    def compress(
        self, delta: Dict[str, torch.Tensor], device_id: int
    ) -> Tuple[Dict[str, torch.Tensor], int, int]:
        """Returns decompressed delta for aggregation, parameters sent, bytes sent."""


class NoCompression(Compressor):
    def compress(self, delta, device_id):
        params = sum(v.numel() for v in delta.values())
        bytes_sent = params * 4
        return delta, params, bytes_sent


class TopKErrorFeedback(Compressor):
    def __init__(self, ratio: float = 0.08):
        self.ratio = ratio
        self.residuals: Dict[int, Dict[str, torch.Tensor]] = {}

    def _init_residual(self, delta: Dict[str, torch.Tensor], device_id: int) -> None:
        if device_id not in self.residuals:
            self.residuals[device_id] = {k: torch.zeros_like(v) for k, v in delta.items()}

    def compress(self, delta, device_id):
        self._init_residual(delta, device_id)
        residual = self.residuals[device_id]
        reconstructed = {}
        total_kept = 0
        for k, t in delta.items():
            u = t + residual[k]
            flat = u.flatten()
            k_keep = max(1, int(self.ratio * flat.numel()))
            if k_keep >= flat.numel():
                mask = torch.ones_like(flat, dtype=torch.bool)
            else:
                top_idx = torch.topk(flat.abs(), k_keep).indices
                mask = torch.zeros_like(flat, dtype=torch.bool)
                mask[top_idx] = True
            compressed_flat = torch.zeros_like(flat)
            compressed_flat[mask] = flat[mask]
            residual[k] = (flat - compressed_flat).view_as(t)
            reconstructed[k] = compressed_flat.view_as(t)
            total_kept += int(mask.sum().item())

        # Send value + index for sparse encoding.
        bytes_sent = total_kept * (4 + 4)
        return reconstructed, total_kept, bytes_sent


class QSGD(Compressor):
    def __init__(self, levels: int = 8):
        self.levels = levels

    def compress(self, delta, device_id):
        reconstructed = {}
        sent_params = 0
        total_bytes = 0
        for k, t in delta.items():
            flat = t.flatten()
            norm = torch.norm(flat, p=2)
            if norm.item() == 0:
                reconstructed[k] = torch.zeros_like(t)
                continue
            abs_v = flat.abs() / norm
            scaled = abs_v * self.levels
            lower = torch.floor(scaled)
            prob = scaled - lower
            rnd = torch.rand_like(prob)
            q = lower + (rnd < prob).float()
            q = torch.clamp(q, max=self.levels)
            recon = torch.sign(flat) * norm * q / self.levels
            reconstructed[k] = recon.view_as(t)
            sent_params += flat.numel()
            total_bytes += flat.numel()  # 1 byte quant level
            total_bytes += 4  # norm per tensor
        return reconstructed, sent_params, total_bytes


class SignSGD(Compressor):
    def __init__(self, use_scale: bool = True):
        self.use_scale = use_scale

    def compress(self, delta, device_id):
        reconstructed = {}
        sent_params = 0
        total_bytes = 0
        for k, t in delta.items():
            flat = t.flatten()
            sign = torch.sign(flat)
            sign[sign == 0] = 1
            if self.use_scale:
                scale = flat.abs().mean()
                recon = sign * scale
                total_bytes += 4
            else:
                recon = sign
            reconstructed[k] = recon.view_as(t)
            sent_params += flat.numel()
            total_bytes += (flat.numel() + 7) // 8  # packed bits
        return reconstructed, sent_params, total_bytes
