import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_experiments(experiments):
    plt.figure(figsize=(5, 3))
    sns.set_style("whitegrid")
    colors = sns.color_palette("colorblind")

    for idx, name in enumerate(experiments):
        path = Path(f"output/logs/{name}/metrics.csv")
        if not path.exists(): continue
        
        df = pd.read_csv(path)
        if "params" not in df.columns: continue

        acc_col = "test_acc" if "test_acc" in df.columns else "val_acc"
        df = df.groupby("params")[acc_col].max().reset_index()
        df["params_mil"] = df["params"] / 1e6
        df = df.sort_values("params_mil", ascending=False)

        color = colors[idx % len(colors)]
        if len(df) == 1:
            plt.scatter(df["params_mil"], df[acc_col], s=100, label=name, color=color, edgecolors='k', zorder=5)
        else:
            plt.plot(df["params_mil"], df[acc_col], marker='o', markersize=5, label=name, color=color)

    plt.xlabel("Parameters (Millions)", fontsize=10)
    plt.ylabel("Test Accuracy", fontsize=10)
    #plt.ylim(0.0, 1.0)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("final_results.png", dpi=150)
    print("Plot saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', nargs='+', default=["default", "imp_baseline"])
    args = parser.parse_args()
    plot_experiments(args.experiments)