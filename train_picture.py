import os
import argparse
from typing import Dict, List, Tuple, Optional

import torch
import matplotlib.pyplot as plt



def resolve_ckpt_path(path: str) -> str:
    """
    Resolve checkpoint path robustly:
    - try given path
    - try swapping 'bandwidth' <-> 'bandwith' (common typo)
    - try with/without leading './'
    """
    candidates = [path]

    
    if path.startswith("./"):
        candidates.append(path[2:])
    else:
        candidates.append("./" + path)

    
    if "bandwidth" in path:
        candidates.append(path.replace("bandwidth", "bandwith"))
    if "bandwith" in path:
        candidates.append(path.replace("bandwith", "bandwidth"))

    
    extra = []
    for p in list(candidates):
        if "bandwidth" in p:
            extra.append(p.replace("bandwidth", "bandwith"))
        if "bandwith" in p:
            extra.append(p.replace("bandwith", "bandwidth"))
    candidates.extend(extra)

    
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    for c in uniq:
        if os.path.exists(c):
            return c

    raise FileNotFoundError(
        "Checkpoint not found. Tried:\n" + "\n".join(uniq) +
        f"\nCWD={os.getcwd()}"
    )


def load_ckpt(path: str, map_location: str = "cpu") -> Dict:
    path = resolve_ckpt_path(path)
    
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint is not a dict: {path}")
    ckpt["_resolved_path"] = path
    return ckpt


def extract_series(ckpt: Dict) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Expect keys saved by your training script:
      - train_losses
      - test_losses
      - accuracies
      - epoch_times
    """
    def get_list(key: str) -> List[float]:
        v = ckpt.get(key, None)
        if v is None:
            raise KeyError(f"Missing key '{key}' in ckpt: {ckpt.get('_resolved_path','<unknown>')}")
        return [float(x) for x in v]

    tr = get_list("train_losses")
    te = get_list("test_losses")
    acc = get_list("accuracies")
    t = get_list("epoch_times")
    
    n = min(len(tr), len(te), len(acc), len(t))
    return tr[:n], te[:n], acc[:n], t[:n]


#helpers
def plot_one_axis(ax, x, y, label, marker, markevery=1):
    ax.plot(
        x, y,
        marker=marker,
        markevery=markevery,
        linewidth=1.8,
        markersize=5.5,
        label=label
    )


def make_2x4_figure(
    datasets: Dict[str, List[Tuple[str, str]]],
    out_path: str,
    dpi: int = 220
):
    """
    datasets:
      {
        "mnist": [(label, ckpt_path), (label, ckpt_path), (label, ckpt_path)],
        "cifar10": [...]
      }
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), constrained_layout=True)

    
    markers = ["o", "^", "x"]  # circle, triangle, cross

    row_order = ["mnist", "cifar10"]
    col_titles = ["Train Loss", "Test Loss", "Accuracy", "Epoch Time (s)"]

    for r, ds_name in enumerate(row_order):
        models = datasets[ds_name]

        
        loaded = []
        for (label, path) in models:
            ckpt = load_ckpt(path, map_location="cpu")
            tr, te, acc, t = extract_series(ckpt)
            loaded.append((label, tr, te, acc, t, ckpt["_resolved_path"]))

        
        for c in range(4):
            ax = axes[r, c]
            ax.grid(True, alpha=0.25)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=12)

            if c == 0:
                ax.set_ylabel(ds_name.upper(), fontsize=12, fontweight="bold")

            for i, (label, tr, te, acc, t, resolved) in enumerate(loaded):
                series = [tr, te, acc, t][c]
                x = list(range(1, len(series) + 1))
                plot_one_axis(
                    ax=ax,
                    x=x,
                    y=series,
                    label=label,
                    marker=markers[i % len(markers)],
                    markevery=max(1, len(series)//12)  # reduce marker clutter
                )

            ax.set_xlabel("Epoch")

        
        ax0 = axes[r, 0]
        handles, labels = ax0.get_legend_handles_labels()
        ax0.legend(
            handles,
            labels,
            loc="upper right",
            frameon=True,
            fontsize=9
        )

    # save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="./final/training_curves_2x4.png")
    args = parser.parse_args()

    # checkpoints"
    datasets = {
        "mnist": [
            ("Baseline (dense)", "./final/mnist-baseline.pt"),
            ("Variant (blockdiag2x2, bw=4)", "./final/mnist-variant-blockdiag2x2-bandwidth4.pt"),
            ("Variant (blockdiag2x2, bw=2)", "./final/mnist-variant-blockdiag2x2.pt"),
        ],
        "cifar10": [
            ("Baseline (dense)", "./final/cifar10-baseline.pt"),
            ("Variant (blockdiag2x2, bw=4)", "./final/cifar10-variant-blockdiag2x2-bandwidth4.pt"),
            ("Variant (blockdiag2x2, bw=2)", "./final/cifar10-variant-blockdiag2x2.pt"),
        ]
    }

    make_2x4_figure(datasets=datasets, out_path=args.out)


if __name__ == "__main__":
    main()
