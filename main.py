import math
import time
import argparse
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

#dataset
def load_dataset(dataset: str, batch_size: int, train: bool):
    dataset = dataset.lower()

    if dataset == "mnist":
        if train:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(
                    (28, 28), scale=(0.8, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=2
                ),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

        ds = torchvision.datasets.MNIST(
            root="./data", train=train, download=True, transform=transform
        )
        image_size, num_channels = 28, 1

    elif dataset == "cifar10":
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                )
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                )
            ])

        ds = torchvision.datasets.CIFAR10(
            root="./data", train=train, download=True, transform=transform
        )
        image_size, num_channels = 32, 3

    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from ['mnist', 'cifar10'].")

    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=(DEVICE == "cuda")
    )
    return loader, image_size, num_channels


#helpers
def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
    else:
        raise ValueError(f"Unexpected freqs_cis shape {freqs_cis.shape} for x shape {x.shape}")
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


#block diagonal 2x2 closed form
class CayleyOrthogonal(nn.Module):
    """
    Per-head Cayley transform:
        P = (I - S)(I + S)^(-1)

    baseline: dense skew-symmetric S via S_raw, using torch.linalg.solve
    variant : block-diagonal skew-symmetric S with 2x2 blocks, closed-form (no solve)

    API kept compatible:
      - cayley_mode: "baseline" or "variant"
      - bandwidth arg is accepted but ignored for block-diagonal variant (for fair CLI compatibility)
    """
    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        eps: float = 1e-5,
        cayley_mode: str = "baseline",
        bandwidth=None,  # kept for compatibility; ignored by block-diagonal variant
    ):
        super().__init__()
        assert cayley_mode in ("baseline", "variant")
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.eps = eps
        self.cayley_mode = cayley_mode
        self.bandwidth = bandwidth

        self.register_buffer("I", torch.eye(head_dim), persistent=False)

        if self.cayley_mode == "baseline":
            # dense generator
            self.S_raw = nn.Parameter(torch.zeros(num_heads, head_dim, head_dim))
        else:
            # block-diagonal 2x2 generator: each block parameter a => [[0,a],[-a,0]]
            # number of 2x2 blocks
            n_blocks = head_dim // 2
            self.n_blocks = n_blocks
            self.has_tail = (head_dim % 2 == 1)  # if odd D, last dim left unchanged
            # one 'a' per head per block
            self.a = nn.Parameter(torch.zeros(num_heads, n_blocks))

    def _skew_dense(self):
        A = self.S_raw
        return A - A.transpose(-1, -2)

    def _apply_blockdiag(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply block-diagonal Cayley P to x without forming matrices.
        x: (B, H, N, D)
        Returns y with same shape.
        """
        B, H, N, D = x.shape
        assert H == self.num_heads and D == self.head_dim

        # pairs: (0,1), (2,3), ...
        n_blocks = self.n_blocks
        # (B,H,N,2*n_blocks)
        x_pair = x[..., :2 * n_blocks]

        # reshape into (B,H,N,n_blocks,2)
        x_pair = x_pair.view(B, H, N, n_blocks, 2)
        x0 = x_pair[..., 0]  # (B,H,N,n_blocks)
        x1 = x_pair[..., 1]  # (B,H,N,n_blocks)

        # a: (H,n_blocks) -> broadcast to (B,H,N,n_blocks)
        a = self.a.view(1, H, 1, n_blocks)

        # Cayley of 2x2 skew block:
        # P = 1/(1+a^2) * [[1-a^2, -2a], [ 2a ,  1-a^2]]
        denom = 1.0 + a * a
        c = (1.0 - a * a) / denom
        s = (2.0 * a) / denom

        y0 = c * x0 - s * x1
        y1 = s * x0 + c * x1

        y_pair = torch.stack([y0, y1], dim=-1).view(B, H, N, 2 * n_blocks)

        if self.has_tail:
            # keep last dimension unchanged (or you could learn a 1x1 "rotation", but that's just ±1)
            tail = x[..., 2 * n_blocks:]
            return torch.cat([y_pair, tail], dim=-1)
        else:
            return y_pair

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, N, D)
        """
        if self.cayley_mode == "variant":
            return self._apply_blockdiag(x)

        #  baseline dense Cayley with solve
        B, H, N, D = x.shape
        assert H == self.num_heads and D == self.head_dim

        S = self._skew_dense()
        I = self.I.to(x.device, x.dtype).expand(H, -1, -1)

        A = (I + S.to(x.dtype)) + self.eps * I
        Bmat = I - S.to(x.dtype)

        rhs = x.permute(1, 0, 2, 3).reshape(H, B * N, D).transpose(1, 2)

        solve_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        y0 = torch.linalg.solve(A.to(solve_dtype), rhs.to(solve_dtype))
        y = (Bmat.to(solve_dtype) @ y0).to(x.dtype)

        y = y.transpose(1, 2).reshape(H, B, N, D).permute(1, 0, 2, 3)
        return y


class CayleyStringAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qkv_bias: bool = False,
        rope_theta: float = 10.0,
        cayley_mode: str = "baseline",
        cayley_bandwidth=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.rope_theta = rope_theta

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)

        self.cayley = CayleyOrthogonal(
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            cayley_mode=cayley_mode,
            bandwidth=cayley_bandwidth,  # accepted for API compatibility
        )

    def forward(self, x):
        batch_size, num_tokens, hidden_size = x.shape

        qkv = (self.qkv(x)
               .reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        #Axial RoPE
        num_patches = num_tokens - 1
        w = int(math.sqrt(num_patches))
        h = int(math.sqrt(num_patches))

        freqs_cis = compute_axial_cis(
            dim=self.head_dim, end_x=w, end_y=h, theta=self.rope_theta
        ).to(x.device)

        q_rope, k_rope = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)
        q = torch.cat([q[:, :, :1], q_rope], dim=2)
        k = torch.cat([k[:, :, :1], k_rope], dim=2)

        #  Apply Cayley Transform
        q_cls, q_tok = q[:, :, :1], q[:, :, 1:]
        k_cls, k_tok = k[:, :, :1], k[:, :, 1:]

        q_tok = self.cayley(q_tok)
        k_tok = self.cayley(k_tok)

        q = torch.cat([q_cls, q_tok], dim=2)
        k = torch.cat([k_cls, k_tok], dim=2)

        # Attention 
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, hidden_size)
        x = self.proj(x)
        return x


#ViT
class GELUActivation(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size,
                                    kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = GELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])

    def forward(self, x):
        return self.dense_2(self.activation(self.dense_1(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CayleyStringAttention(
            hidden_size=config["hidden_size"],
            num_heads=config["num_attention_heads"],
            qkv_bias=config["qkv_bias"],
            rope_theta=config.get("rope_theta", 10.0),
            cayley_mode=config.get("cayley_mode", "baseline"),
            cayley_bandwidth=config.get("cayley_bandwidth", None),
        )
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x):
        x = x + self.attention(self.layernorm_1(x))
        x = x + self.mlp(self.layernorm_2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config["num_hidden_layers"])])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(config["hidden_size"], config["num_classes"])
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return self.classifier(x[:, 0, :])

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)



# Train / eval with epoch_times + best model
def train_epoch(model, optimizer, loss_fn, device, trainLoader):
    model.train()
    total_loss = 0.0
    for images, labels in trainLoader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(trainLoader.dataset)


@torch.no_grad()
def evaluate(model, loss_fn, device, testLoader):
    model.eval()
    total_loss = 0.0
    correct = 0
    for images, labels in testLoader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()

    acc = correct / len(testLoader.dataset)
    avg_loss = total_loss / len(testLoader.dataset)
    return acc, avg_loss


def train(model, optimizer, loss_fn, device, trainLoader, testLoader, epochs,
          save_best_path: str, meta: dict):

    model.to(device)
    train_losses, test_losses, accuracies, epoch_times = [], [], [], []

    best_acc = -1.0
    best_epoch = -1

    for epoch in range(epochs):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        tr_loss = train_epoch(model, optimizer, loss_fn, device, trainLoader)
        acc, te_loss = evaluate(model, loss_fn, device, testLoader)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        epoch_time = t1 - t0

        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        accuracies.append(acc)
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch+1:02d} | "
              f"train_loss {tr_loss:.4f} | "
              f"test_loss {te_loss:.4f} | "
              f"acc {acc:.4f} | "
              f"time {epoch_time:.2f}s")

        # save best
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_acc": best_acc,
                "best_epoch": best_epoch,
                "meta": meta,
                "train_losses": train_losses,
                "test_losses": test_losses,
                "accuracies": accuracies,
                "epoch_times": epoch_times,
            }, save_best_path)

    return train_losses, test_losses, accuracies, epoch_times, best_acc, best_epoch



# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"])
    parser.add_argument("--mode", type=str, default="variant", choices=["baseline", "variant"],
                        help="baseline=dense Cayley+solve, variant=block-diagonal Cayley (2x2) closed-form")
    parser.add_argument("--bandwidth", type=int, default=4,
                        help="kept for compatibility; ignored by block-diagonal variant")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--rope_theta", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # data
    trainLoader, image_size, num_channels = load_dataset(args.dataset, args.batch_size, train=True)
    testLoader, _, _ = load_dataset(args.dataset, args.batch_size, train=False)

    # config
    cayley_mode = "baseline" if args.mode == "baseline" else "variant"

    config = {
        "image_size": image_size,
        "num_channels": num_channels,
        "num_classes": 10,
        "patch_size": 4,
        "hidden_size": 48,
        "intermediate_size": 4 * 48,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "initializer_range": 0.02,
        "qkv_bias": True,
        "rope_theta": args.rope_theta,
        "cayley_mode": cayley_mode,
        "cayley_bandwidth": None if args.mode == "baseline" else args.bandwidth,  # ignored by block-diagonal
    }

    model = ViT(config)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    tag = f"{args.dataset}-{args.mode}"
    if args.mode == "variant":
        tag += "-blockdiag2x2"

    save_best = f"{tag}-best.pt"

    meta = {
        "dataset": args.dataset,
        "mode": args.mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "config": config,
        "device": DEVICE,
    }

    train_losses, test_losses, accuracies, epoch_times, best_acc, best_epoch = train(
        model, optimizer, loss_fn, DEVICE,
        trainLoader, testLoader,
        epochs=args.epochs,
        save_best_path=save_best,
        meta=meta
    )

    print(f"\nSaved best checkpoint: {save_best}")
    print(f"Best acc: {best_acc:.4f} at epoch {best_epoch}")
    print(f"Avg epoch time: {sum(epoch_times)/len(epoch_times):.3f}s")


if __name__ == "__main__":
    main()
