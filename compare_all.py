import os, time, argparse
from typing import Dict, Tuple, List

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pandas as pd

def make_test_loader(dataset: str, batch_size: int, num_workers: int = 2):
    if dataset == "mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        ds = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tfm)
    elif dataset == "cifar10":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        ds = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=tfm)
    else:
        raise ValueError(dataset)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

def detect_impl(config: Dict) -> str:
    if "cayley_mode" in config:
        return "blockdiag_or_dense_main"
    if "cayley_topk" in config:
        return "topk"
    if "cayley_bandwidth" in config:
        return "banded_or_dense"
    return "unknown"

def build_model(impl: str, config: Dict):
    if impl == "blockdiag_or_dense_main":
        import main as M
        return M.ViT(config)
    if impl == "banded_or_dense":
        import banded_variant as M
        return M.ViT(config)
    if impl == "topk":
        import top_k_variant as M
        return M.ViT(config)
    raise ValueError(f"Unknown impl: {impl}")

@torch.no_grad()
def eval_acc_loss(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        bs = x.size(0)
        total += bs
        correct += (logits.argmax(dim=-1) == y).sum().item()
        loss_sum += loss.item() * bs
    return correct / total, loss_sum / total

@torch.no_grad()
def infer_ms(model, x, device, iters=200, warmup=50):
    model.eval()
    x = x.to(device, non_blocking=True)

    if device.startswith("cuda"):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))  # ms
        return sum(times) / len(times)
    else:
        for _ in range(warmup):
            _ = model(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        t1 = time.perf_counter()
        return 1000.0 * (t1 - t0) / iters

def label_from_ckpt(path: str, meta: Dict, config: Dict) -> str:
    base = os.path.basename(path).replace(".pt","")
    impl = detect_impl(config)
    if impl == "topk":
        k = config.get("cayley_topk", None)
        return f"topk(k={k})" if k is not None else "baseline(dense)"
    if impl == "banded_or_dense":
        bw = config.get("cayley_bandwidth", None)
        return f"banded(bw={bw})" if bw is not None else "baseline(dense)"
    if impl == "blockdiag_or_dense_main":
        mode = config.get("cayley_mode", "baseline")
        if mode == "baseline":
            return "baseline(dense)"
        # main.py's "variant" is blockdiag2x2 in your codebase
        return "blockdiag2x2"
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, default="final")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--infer_batch", type=int, default=128)
    args = ap.parse_args()

    rows = []
    for fn in sorted(os.listdir(args.ckpt_dir)):
        if not fn.endswith(".pt"):
            continue
        path = os.path.join(args.ckpt_dir, fn)
        ckpt = torch.load(path, map_location="cpu")
        meta = ckpt.get("meta", {})
        config = meta.get("config", {})
        dataset = meta.get("dataset", None)
        if dataset is None:
            # fallback: infer from filename prefix
            dataset = "mnist" if fn.startswith("mnist") else "cifar10"

        impl = detect_impl(config)
        model = build_model(impl, config)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(args.device)

        test_loader = make_test_loader(dataset, batch_size=args.batch_size)
        acc, loss = eval_acc_loss(model, test_loader, args.device)

        xb, _ = next(iter(test_loader))
        xb = xb[:args.infer_batch]
        ms_batch = infer_ms(model, xb, args.device)
        ms_img = ms_batch / xb.size(0)

        params = sum(p.numel() for p in model.parameters()) / 1e6

        epoch_times = ckpt.get("epoch_times", [])
        train_s_epoch = (sum(epoch_times) / len(epoch_times)) if len(epoch_times) else None
        train_s_total = (sum(epoch_times)) if len(epoch_times) else None

        rows.append({
            "dataset": dataset,
            "model": label_from_ckpt(path, meta, config),
            "impl": impl,
            "params_M": round(params, 3),
            "best_acc_ckpt": float(ckpt.get("best_acc", acc)),
            "test_acc_now": acc,
            "test_loss_now": loss,
            "train_s_epoch": train_s_epoch,
            "train_s_total": train_s_total,
            "ms_batch": ms_batch,
            "ms_img": ms_img,
            "ckpt": path,
        })

    df = pd.DataFrame(rows).sort_values(["dataset","model"])
    print(df.to_string(index=False))
    df.to_csv(os.path.join(args.ckpt_dir, "comparison.csv"), index=False)
    print(f"\nSaved: {os.path.join(args.ckpt_dir, 'comparison.csv')}")

if __name__ == "__main__":
    main()
