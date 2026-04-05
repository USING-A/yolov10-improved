from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"
OUT_DIR = ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    """Configure a clean publication-style plot theme."""
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_csv(EXP_DIR / "baseline.csv")
    screening = pd.read_csv(EXP_DIR / "module_screening.csv")
    combo = pd.read_csv(EXP_DIR / "combination_ablation.csv")
    tuning = pd.read_csv(EXP_DIR / "hyperparam_tuning.csv")
    return baseline, screening, combo, tuning


def draw_module_screening(baseline: pd.DataFrame, screening: pd.DataFrame) -> None:
    base_map95 = float(
        baseline.loc[baseline["run_name"] == "baseline_bauto_nbs128", "map50_95"].iloc[0]
    )

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    plot_df = screening[["module_name", "map50_95"]].sort_values("map50_95", ascending=True)
    ax.barh(
        plot_df["module_name"],
        plot_df["map50_95"],
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.axvline(
        base_map95,
        color="#D62728",
        linestyle="--",
        linewidth=1.2,
        label=f"Baseline mAP50-95 = {base_map95:.5f}",
    )
    ax.set_xlabel("mAP50-95")
    ax.set_ylabel("Single Module")
    ax.set_title("Single-Module Screening vs Baseline")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)
    for i, value in enumerate(plot_df["map50_95"]):
        ax.text(value + 0.0005, i, f"{value:.5f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_module_screening_map95.png", bbox_inches="tight")
    plt.close(fig)


def draw_combination_ablation(combo: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    plot_df = combo[["run_name", "map50_95", "pareto_rank"]].copy()
    plot_df["label"] = plot_df["run_name"].str.replace("combo_", "", regex=False)
    plot_df = plot_df.sort_values("map50_95", ascending=True)
    colors = ["#72B7B2" if rank == "A" else "#F58518" for rank in plot_df["pareto_rank"]]

    ax.barh(
        plot_df["label"],
        plot_df["map50_95"],
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xlabel("mAP50-95")
    ax.set_ylabel("Combination Run")
    ax.set_title("Combination Ablation (150 Epochs)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    for i, value in enumerate(plot_df["map50_95"]):
        ax.text(value + 0.0005, i, f"{value:.5f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_combination_ablation_map95.png", bbox_inches="tight")
    plt.close(fig)


def draw_hparam_comparison(tuning: pd.DataFrame) -> None:
    comp = tuning[tuning["epochs"] == 100].copy()
    comp = comp.sort_values("map50_95", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    plot_df = comp[["run_name", "map50", "map50_95"]].copy()
    plot_df["label"] = plot_df["run_name"].str.replace("tune_cbam_eca_", "", regex=False)

    x = range(len(plot_df))
    width = 0.36
    ax.bar(
        [i - width / 2 for i in x],
        plot_df["map50"],
        width=width,
        color="#54A24B",
        edgecolor="black",
        linewidth=0.6,
        label="mAP50",
    )
    ax.bar(
        [i + width / 2 for i in x],
        plot_df["map50_95"],
        width=width,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.6,
        label="mAP50-95",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["label"], rotation=10, ha="right")
    ax.set_ylabel("Metric Value")
    ax.set_title("Final 100-Epoch Hyperparameter Comparison")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_hparam_100e_comparison.png", bbox_inches="tight")
    plt.close(fig)


def draw_phase_trajectory(
    baseline: pd.DataFrame, screening: pd.DataFrame, combo: pd.DataFrame, tuning: pd.DataFrame
) -> None:
    base_map95 = float(
        baseline.loc[baseline["run_name"] == "baseline_bauto_nbs128", "map50_95"].iloc[0]
    )
    best_single = screening.loc[screening["map50_95"].idxmax()]
    best_combo = combo.loc[combo["map50_95"].idxmax()]
    best_tune_100 = tuning[tuning["epochs"] == 100].loc[
        tuning[tuning["epochs"] == 100]["map50_95"].idxmax()
    ]

    phase_df = pd.DataFrame(
        {
            "phase": ["Baseline", "Single Module", "Combination", "Final Mainline (100e)"],
            "run": [
                "baseline_bauto_nbs128",
                best_single["run_name"],
                best_combo["run_name"],
                best_tune_100["run_name"],
            ],
            "map50_95": [
                base_map95,
                float(best_single["map50_95"]),
                float(best_combo["map50_95"]),
                float(best_tune_100["map50_95"]),
            ],
        }
    )

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(phase_df["phase"], phase_df["map50_95"], marker="o", color="#B279A2", linewidth=1.8)
    for i, value in enumerate(phase_df["map50_95"]):
        ax.text(i, value + 0.0007, f"{value:.5f}", ha="center", fontsize=9)
    ax.set_ylabel("mAP50-95")
    ax.set_title("Best-Per-Phase mAP50-95 Trajectory")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_phase_best_map95.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_plot_style()
    baseline, screening, combo, tuning = load_data()
    draw_module_screening(baseline, screening)
    draw_combination_ablation(combo)
    draw_hparam_comparison(tuning)
    draw_phase_trajectory(baseline, screening, combo, tuning)

    print("Generated figures:")
    for path in sorted(OUT_DIR.glob("fig_*.png")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
