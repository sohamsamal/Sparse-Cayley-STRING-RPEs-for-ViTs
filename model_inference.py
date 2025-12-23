import time
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import torch
import torchvision
import torchvision.transforms as transforms

from main import ViT  # ✅ 你说的导入方式

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_test_loader(dataset: str, batch_size: int = 128):
    dataset = dataset.lower()
    if dataset == "mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        ds = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tfm)
    elif dataset == "cifar10":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        ds = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=tfm)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(DEVICE == "cuda")
    )
    return loader


def load_ckpt(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # ✅ 更安全：优先 weights_only=True（新版本 pytorch 支持）
    try:
        return torch.load(str(p), map_location="cpu", weights_only=True)
    except TypeError:
        # 老版本 torch 没有 weights_only 参数
        return torch.load(str(p), map_location="cpu")
    except Exception:
        # 如果 ckpt 里有非 tensor 对象，weights_only=True 可能不允许 -> fallback
        return torch.load(str(p), map_location="cpu")


def build_model_from_ckpt(ckpt: Dict[str, Any]):
    config = ckpt.get("meta", {}).get("config", None)
    if config is None:
        raise KeyError("Checkpoint missing ckpt['meta']['config']")

    model = ViT(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return model


def count_params(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def extract_training_time(ckpt: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns:
      avg_epoch_s: average training seconds per epoch (from epoch_times)
      total_s: total training seconds (sum(epoch_times))
    If not available, returns (None, None).
    """
    epoch_times = ckpt.get("epoch_times", None)

    # 有些人会把它放在 meta 里（兜底）
    if epoch_times is None:
        epoch_times = ckpt.get("meta", {}).get("epoch_times", None)

    if epoch_times is None:
        return None, None

    # epoch_times 可能是 list[float] / tensor / numpy array
    try:
        if isinstance(epoch_times, torch.Tensor):
            epoch_times = epoch_times.detach().cpu().tolist()
        else:
            epoch_times = list(epoch_times)
    except Exception:
        return None, None

    if len(epoch_times) == 0:
        return None, None

    total_s = float(sum(epoch_times))
    avg_epoch_s = total_s / len(epoch_times)
    return avg_epoch_s, total_s


@torch.no_grad()
def measure_infer(model, loader, warmup: int = 5, max_batches: int = 50):
    # warmup
    it = iter(loader)
    for _ in range(warmup):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        x = x.to(DEVICE, non_blocking=True)
        _ = model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    times = []
    n_images = 0
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(DEVICE, non_blocking=True)

        t0 = time.time()
        _ = model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

        times.append(t1 - t0)
        n_images += x.size(0)

    avg_batch_s = sum(times) / len(times)
    imgs_per_batch = n_images / len(times)
    ms_per_batch = avg_batch_s * 1000.0
    ms_per_img = (avg_batch_s / imgs_per_batch) * 1000.0
    return ms_per_batch, ms_per_img


def main():
    runs = [
        ("mnist",   "baseline",          "./final/mnist-baseline.pt"),
        ("mnist",   "blockdiag2x2-bw2",   "./final/mnist-variant-blockdiag2x2.pt"),
        ("mnist",   "blockdiag2x2-bw4",   "./final/mnist-variant-blockdiag2x2-bandwidth4.pt"),

        ("cifar10", "baseline",          "./final/cifar10-baseline.pt"),
        ("cifar10", "blockdiag2x2-bw2",   "./final/cifar10-variant-blockdiag2x2.pt"),
        ("cifar10", "blockdiag2x2-bw4",   "./final/cifar10-variant-blockdiag2x2-bandwidth4.pt"),
    ]

    loaders = {
        "mnist": load_test_loader("mnist", batch_size=128),
        "cifar10": load_test_loader("cifar10", batch_size=128),
    }

    print(f"Device: {DEVICE}\n")
    print("dataset | model                | params(M) | best_acc | best_epoch | train_s/epoch | train_s_total | ms/batch | ms/img")
    print("-" * 126)

    for ds, tag, path in runs:
        ckpt = load_ckpt(path)
        model = build_model_from_ckpt(ckpt)

        total, _ = count_params(model)
        total_m = total / 1e6

        best_acc = ckpt.get("best_acc", None)
        best_epoch = ckpt.get("best_epoch", None)

        # ✅ 从 ckpt 读训练时间
        avg_epoch_s, total_train_s = extract_training_time(ckpt)

        ms_batch, ms_img = measure_infer(model, loaders[ds])

        best_acc_str = f"{best_acc:.4f}" if isinstance(best_acc, (float, int)) else "NA"
        best_epoch_str = f"{best_epoch}" if isinstance(best_epoch, (int,)) else "NA"
        avg_epoch_str = f"{avg_epoch_s:.2f}" if isinstance(avg_epoch_s, (float, int)) else "NA"
        total_train_str = f"{total_train_s:.1f}" if isinstance(total_train_s, (float, int)) else "NA"

        print(f"{ds:7s} | {tag:20s} | {total_m:8.3f} | "
              f"{best_acc_str:7s} | {best_epoch_str:10s} | "
              f"{avg_epoch_str:11s} | {total_train_str:12s} | "
              f"{ms_batch:7.2f} | {ms_img:6.3f}")


if __name__ == "__main__":
    main()
