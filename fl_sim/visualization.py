import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def make_plots(method_dfs: Dict[str, pd.DataFrame], results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)

    def save_plot(fig, name: str):
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, name), dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for m, df in method_dfs.items():
        ax.plot(df["round_idx"], df["test_accuracy"], marker="o", label=m)
    ax.set_title("Accuracy vs Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Test Accuracy")
    ax.legend()
    save_plot(fig, "accuracy_vs_rounds.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for m, df in method_dfs.items():
        ax.plot(df["round_idx"], df["latency_sec"], marker="o", label=m)
    ax.set_title("Latency vs Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Latency (s)")
    ax.legend()
    save_plot(fig, "latency_vs_rounds.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for m, df in method_dfs.items():
        ax.plot(df["round_idx"], df["communication_mb"].cumsum(), marker="o", label=m)
    ax.set_title("Communication vs Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Communication (MB)")
    ax.legend()
    save_plot(fig, "communication_vs_rounds.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for m, df in method_dfs.items():
        ax.scatter(df["latency_sec"], df["test_accuracy"], label=m)
    ax.set_title("Accuracy vs Latency")
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Test Accuracy")
    ax.legend()
    save_plot(fig, "accuracy_vs_latency.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for m, df in method_dfs.items():
        ax.scatter(df["communication_mb"].cumsum(), df["test_accuracy"], label=m)
    ax.set_title("Accuracy vs Communication")
    ax.set_xlabel("Cumulative Communication (MB)")
    ax.set_ylabel("Test Accuracy")
    ax.legend()
    save_plot(fig, "accuracy_vs_communication.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for m, df in method_dfs.items():
        ax.hist(df["latency_sec"], bins=8, alpha=0.4, label=m)
    ax.set_title("Latency Distribution")
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Frequency")
    ax.legend()
    save_plot(fig, "latency_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for m, df in method_dfs.items():
        ax.plot(df["round_idx"], df["active_devices"], marker="o", label=m)
    ax.set_title("Participation per Round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Active Devices")
    ax.legend()
    save_plot(fig, "participation_per_round.png")
