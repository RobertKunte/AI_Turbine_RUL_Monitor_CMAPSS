import numpy as np
import torch


def _to_np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu()
        return x.numpy()
    return np.asarray(x)


def tensor_stats(name, x, max_items=5):
    a = _to_np(x)
    finite = np.isfinite(a)
    fin_ratio = float(finite.mean()) if a.size > 0 else 0.0
    a_fin = a[finite] if finite.any() else a.reshape(-1)[:0]
    if a_fin.size > 0:
        msg = (
            f"[dbg][{name}] shape={tuple(a.shape)} "
            f"finite={fin_ratio:.4f} "
            f"mean={float(a_fin.mean()):.6f} std={float(a_fin.std()):.6f} "
            f"min={float(a_fin.min()):.6f} max={float(a_fin.max()):.6f}"
        )
    else:
        msg = f"[dbg][{name}] shape={tuple(a.shape)} finite={fin_ratio:.4f} (no finite values)"
    print(msg)


def batch_time_std(name, x):
    # x: [B,T,1] or [B,T]
    if isinstance(x, torch.Tensor):
        t = x.detach().float()
        if t.dim() == 3 and t.size(-1) == 1:
            t = t[..., 0]
        std_over_batch = t.std(dim=0).mean().item()
        std_over_time = t.std(dim=1).mean().item()
        print(
            f"[dbg][{name}] std_over_batch_mean={std_over_batch:.8f} std_over_time_mean={std_over_time:.8f}"
        )
    else:
        a = _to_np(x)
        if a.ndim == 3 and a.shape[-1] == 1:
            a = a[..., 0]
        std_over_batch = float(a.std(axis=0).mean())
        std_over_time = float(a.std(axis=1).mean())
        print(
            f"[dbg][{name}] std_over_batch_mean={std_over_batch:.8f} std_over_time_mean={std_over_time:.8f}"
        )


def compare_two_samples(name, x, t_steps=5):
    # compare sample 0 vs 1 for first t_steps
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(_to_np(x))
    t = x.detach().float().cpu()
    if t.dim() == 3 and t.size(-1) == 1:
        t = t[..., 0]
    a0 = t[0, :t_steps].numpy()
    a1 = t[1, :t_steps].numpy() if t.size(0) > 1 else None
    print(f"[dbg][{name}] sample0[:{t_steps}]={a0}")
    if a1 is not None:
        print(f"[dbg][{name}] sample1[:{t_steps}]={a1}")
        print(f"[dbg][{name}] sample0==sample1(all)={np.allclose(a0, a1)}")

