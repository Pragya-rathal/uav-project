from dataclasses import dataclass


@dataclass
class SimulationConfig:
    seed: int = 42
    num_devices: int = 24
    num_clusters: int = 4
    rounds: int = 8
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    dirichlet_alpha: float = 0.5
    participation_rate: float = 0.8
    device_compute_min: float = 0.5
    device_compute_max: float = 2.0
    bandwidth_min_mbps: float = 1.0
    bandwidth_max_mbps: float = 10.0
    clustering_coeff_min: float = 0.3
    clustering_coeff_max: float = 1.0
    topk_ratio: float = 0.08
    qsgd_levels: int = 8
    signsgd_scale: bool = True
    base_compute_per_sample_sec: float = 8e-4
    aggregation_time_sec: float = 0.2
    results_dir: str = "results"
    device: str = "cuda"
