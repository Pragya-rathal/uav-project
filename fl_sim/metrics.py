from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class RoundMetrics:
    round_idx: int
    train_loss: float
    test_loss: float
    test_accuracy: float
    communication_mb: float
    latency_sec: float
    active_devices: int


def metrics_to_df(metrics: List[RoundMetrics]) -> pd.DataFrame:
    return pd.DataFrame([m.__dict__ for m in metrics])
