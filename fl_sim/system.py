from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class DeviceProfile:
    device_id: int
    cluster_id: int
    compute_power: float
    bandwidth_mbps: float
    clustering_coefficient: float
    distance_to_uav: float
    channel_quality: float


@dataclass
class Cluster:
    cluster_id: int
    device_ids: List[int]
    head_id: int


def create_device_profiles(
    num_devices: int,
    num_clusters: int,
    compute_min: float,
    compute_max: float,
    bandwidth_min: float,
    bandwidth_max: float,
    coeff_min: float,
    coeff_max: float,
    seed: int,
) -> tuple[Dict[int, DeviceProfile], Dict[int, Cluster]]:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, 100, size=(num_devices, 2))
    uav_pos = np.array([50.0, 50.0])
    distances = np.linalg.norm(positions - uav_pos, axis=1)

    shuffled = np.arange(num_devices)
    rng.shuffle(shuffled)
    cluster_splits = np.array_split(shuffled, num_clusters)

    profiles: Dict[int, DeviceProfile] = {}
    clusters: Dict[int, Cluster] = {}

    for cid, split in enumerate(cluster_splits):
        device_ids = split.tolist()
        scores = []
        for did in device_ids:
            compute_power = float(rng.uniform(compute_min, compute_max))
            bandwidth = float(rng.uniform(bandwidth_min, bandwidth_max))
            coeff = float(rng.uniform(coeff_min, coeff_max))
            dist = float(distances[did])
            channel = 1.0 / (1.0 + dist / 100.0)
            score = 0.6 * compute_power + 0.4 * coeff
            scores.append((did, score))
            profiles[did] = DeviceProfile(
                device_id=did,
                cluster_id=cid,
                compute_power=compute_power,
                bandwidth_mbps=bandwidth,
                clustering_coefficient=coeff,
                distance_to_uav=dist,
                channel_quality=channel,
            )
        head_id = max(scores, key=lambda x: x[1])[0]
        clusters[cid] = Cluster(cluster_id=cid, device_ids=device_ids, head_id=head_id)

    return profiles, clusters
