import argparse
import os
from collections import defaultdict

import pandas as pd
import torch

from fl_sim.compression import NoCompression, QSGD, SignSGD, TopKErrorFeedback
from fl_sim.config import SimulationConfig
from fl_sim.data import (
    dirichlet_split_noniid,
    load_mnist,
    make_client_loaders,
    make_test_loader,
    summarize_client_distribution,
)
from fl_sim.metrics import RoundMetrics, metrics_to_df
from fl_sim.models import MNISTCNN
from fl_sim.system import create_device_profiles
from fl_sim.trainer import evaluate, local_train
from fl_sim.utils import (
    add_state_dict,
    clone_state_dict,
    scale_state_dict,
    set_seed,
    subtract_state_dict,
    weighted_average_state_dict,
)
from fl_sim.visualization import make_plots


def run_method(method_name, compressor, cfg, base_context):
    device = base_context["torch_device"]
    profiles = base_context["profiles"]
    clusters = base_context["clusters"]
    client_loaders = base_context["client_loaders"]
    test_loader = base_context["test_loader"]
    client_indices = base_context["client_indices"]

    model = MNISTCNN()
    global_state = clone_state_dict(model.state_dict())

    metrics = []
    total_comm_mb = 0.0
    rng = torch.Generator().manual_seed(cfg.seed + hash(method_name) % 1000)

    for r in range(1, cfg.rounds + 1):
        active_devices = []
        for did in sorted(client_loaders.keys()):
            if torch.rand(1, generator=rng).item() < cfg.participation_rate:
                active_devices.append(did)

        if not active_devices:
            active_devices = [int(torch.randint(len(client_loaders), (1,), generator=rng).item())]

        cluster_payloads = defaultdict(list)
        train_losses = []
        cluster_latencies = {}
        round_comm_bytes = 0

        for did in active_devices:
            local_model = MNISTCNN()
            local_state, train_loss = local_train(
                local_model,
                global_state,
                client_loaders[did],
                device,
                cfg.local_epochs,
                cfg.lr,
                cfg.momentum,
                cfg.weight_decay,
            )
            train_losses.append(train_loss)
            delta = subtract_state_dict(local_state, global_state)
            rec_delta, sent_params, sent_bytes = compressor.compress(delta, did)
            round_comm_bytes += sent_bytes

            sample_count = len(client_indices[did])
            cluster_id = profiles[did].cluster_id
            cluster_payloads[cluster_id].append((did, rec_delta, sample_count))

            num_samples = len(client_indices[did])
            base_compute = cfg.base_compute_per_sample_sec * num_samples * cfg.local_epochs
            compute_time = base_compute / profiles[did].compute_power
            model_mb = sent_bytes / (1024 * 1024)
            upload_time = (model_mb * 8.0) / profiles[did].bandwidth_mbps
            device_time = compute_time + upload_time
            cluster_latencies.setdefault(cluster_id, []).append(device_time)

        head_updates = []
        head_weights = []
        for cid, payloads in cluster_payloads.items():
            updates = [p[1] for p in payloads]
            weights = [p[2] for p in payloads]
            cluster_agg = weighted_average_state_dict(updates, weights)
            head_updates.append(cluster_agg)
            head_weights.append(sum(weights))

        if head_updates:
            server_delta = weighted_average_state_dict(head_updates, head_weights)
            global_state = add_state_dict(global_state, server_delta)

        round_latency = max(max(v) for v in cluster_latencies.values()) + cfg.aggregation_time_sec
        total_comm_mb += round_comm_bytes / (1024 * 1024)

        test_loss, test_acc = evaluate(MNISTCNN(), global_state, test_loader, device)
        metrics.append(
            RoundMetrics(
                round_idx=r,
                train_loss=float(sum(train_losses) / max(1, len(train_losses))),
                test_loss=float(test_loss),
                test_accuracy=float(test_acc),
                communication_mb=float(round_comm_bytes / (1024 * 1024)),
                latency_sec=float(round_latency),
                active_devices=len(active_devices),
            )
        )

        print(
            f"[{method_name}] Round {r:02d} | acc={test_acc:.4f} | "
            f"loss={test_loss:.4f} | comm={round_comm_bytes/(1024*1024):.3f} MB | "
            f"lat={round_latency:.3f}s | active={len(active_devices)}"
        )

    df = metrics_to_df(metrics)
    summary = {
        "method": method_name,
        "final_accuracy": float(df["test_accuracy"].iloc[-1]),
        "avg_latency": float(df["latency_sec"].mean()),
        "total_comm": float(df["communication_mb"].sum()),
    }
    return df, summary


def main():
    parser = argparse.ArgumentParser(description="Clustered Federated Learning Compression Simulation")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--num-devices", type=int, default=24)
    parser.add_argument("--num-clusters", type=int, default=4)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--topk", type=float, default=0.08)
    parser.add_argument("--qsgd-levels", type=int, default=8)
    args = parser.parse_args()

    cfg = SimulationConfig(
        rounds=args.rounds,
        num_devices=args.num_devices,
        num_clusters=args.num_clusters,
        local_epochs=args.local_epochs,
        dirichlet_alpha=args.alpha,
        batch_size=args.batch_size,
        topk_ratio=args.topk,
        qsgd_levels=args.qsgd_levels,
    )
    os.makedirs(cfg.results_dir, exist_ok=True)
    set_seed(cfg.seed)

    torch_device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch_device}")

    train_set, test_set = load_mnist()
    client_indices = dirichlet_split_noniid(
        train_set.targets, num_devices=cfg.num_devices, alpha=cfg.dirichlet_alpha, seed=cfg.seed
    )
    dist_summary = summarize_client_distribution(train_set.targets, client_indices)
    pd.DataFrame.from_dict(dist_summary, orient="index").fillna(0).to_csv(
        os.path.join(cfg.results_dir, "client_class_distribution.csv")
    )

    client_loaders = make_client_loaders(train_set, client_indices, batch_size=cfg.batch_size, seed=cfg.seed)
    test_loader = make_test_loader(test_set)

    profiles, clusters = create_device_profiles(
        num_devices=cfg.num_devices,
        num_clusters=cfg.num_clusters,
        compute_min=cfg.device_compute_min,
        compute_max=cfg.device_compute_max,
        bandwidth_min=cfg.bandwidth_min_mbps,
        bandwidth_max=cfg.bandwidth_max_mbps,
        coeff_min=cfg.clustering_coeff_min,
        coeff_max=cfg.clustering_coeff_max,
        seed=cfg.seed,
    )

    profile_df = pd.DataFrame([vars(p) for p in profiles.values()]).sort_values("device_id")
    profile_df.to_csv(os.path.join(cfg.results_dir, "device_profiles.csv"), index=False)
    cluster_df = pd.DataFrame(
        [
            {
                "cluster_id": c.cluster_id,
                "head_id": c.head_id,
                "device_ids": " ".join(map(str, c.device_ids)),
            }
            for c in clusters.values()
        ]
    )
    cluster_df.to_csv(os.path.join(cfg.results_dir, "clusters.csv"), index=False)

    context = {
        "torch_device": torch_device,
        "profiles": profiles,
        "clusters": clusters,
        "client_loaders": client_loaders,
        "test_loader": test_loader,
        "client_indices": client_indices,
    }

    methods = {
        "no_compression": NoCompression(),
        "topk_error_feedback": TopKErrorFeedback(ratio=cfg.topk_ratio),
        "qsgd": QSGD(levels=cfg.qsgd_levels),
        "signsgd": SignSGD(use_scale=cfg.signsgd_scale),
    }

    all_dfs = {}
    summaries = []
    for name, comp in methods.items():
        df, summary = run_method(name, comp, cfg, context)
        df.to_csv(os.path.join(cfg.results_dir, f"{name}_metrics.csv"), index=False)
        all_dfs[name] = df
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(cfg.results_dir, "final_comparison.csv"), index=False)
    make_plots(all_dfs, cfg.results_dir)

    print("\nmethod | final_accuracy | avg_latency | total_comm")
    for _, row in summary_df.iterrows():
        print(f"{row['method']} | {row['final_accuracy']:.4f} | {row['avg_latency']:.4f} | {row['total_comm']:.4f}")


if __name__ == "__main__":
    main()
