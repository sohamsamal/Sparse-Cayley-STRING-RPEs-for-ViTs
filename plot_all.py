import os, argparse
import torch
import matplotlib.pyplot as plt

def load_hist(path):
    ckpt = torch.load(path, map_location="cpu")
    meta = ckpt.get("meta", {})
    cfg = meta.get("config", {})
    dataset = meta.get("dataset", "mnist" if os.path.basename(path).startswith("mnist") else "cifar10")

    # label
    if "cayley_topk" in cfg:
        k = cfg["cayley_topk"]
        label = f"TopK(k={k})" if k is not None else "Baseline(dense)"
    elif "cayley_bandwidth" in cfg and "cayley_mode" not in cfg:
        bw = cfg["cayley_bandwidth"]
        label = f"Banded(bw={bw})" if bw is not None else "Baseline(dense)"
    elif "cayley_mode" in cfg:
        label = "Baseline(dense)" if cfg["cayley_mode"] == "baseline" else "BlockDiag2x2"
    else:
        label = os.path.basename(path).replace(".pt","")

    return dataset, label, ckpt.get("train_losses", []), ckpt.get("test_losses", []), ckpt.get("accuracies", []), ckpt.get("epoch_times", [])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, default="final")
    ap.add_argument("--out", type=str, default="final/training_all.png")
    args = ap.parse_args()

    paths = [os.path.join(args.ckpt_dir, f) for f in os.listdir(args.ckpt_dir) if f.endswith(".pt")]
    items = [load_hist(p) for p in sorted(paths)]

    by_ds = {"mnist": [], "cifar10": []}
    for it in items:
        by_ds[it[0]].append(it)

    fig, axes = plt.subplots(2, 4, figsize=(18, 6), dpi=150)
    cols = ["Train Loss", "Test Loss", "Accuracy", "Epoch Time (s)"]
    for r, ds in enumerate(["mnist", "cifar10"]):
        for c in range(4):
            axes[r, c].set_title(cols[c])
            axes[r, c].set_xlabel("Epoch")
        axes[r,0].set_ylabel(ds.upper())

        for (dataset, label, tr, te, acc, et) in by_ds[ds]:
            x = list(range(1, len(tr)+1))
            if tr:
                axes[r,0].plot(x, tr, marker="o", linewidth=1, label=label)
            if te:
                axes[r,1].plot(x, te, marker="o", linewidth=1, label=label)
            if acc:
                axes[r,2].plot(x, acc, marker="o", linewidth=1, label=label)
            if et:
                axes[r,3].plot(list(range(1, len(et)+1)), et, marker="o", linewidth=1, label=label)

        for c in range(4):
            axes[r,c].grid(True, alpha=0.3)
            if c != 3:
                axes[r,c].legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
