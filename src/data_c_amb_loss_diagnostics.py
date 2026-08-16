"""Data C diagnostics: Multires λ=3 with/without best-ambiguity pulse loss.

Two models (same Data C protocol, λ=3, Multires 2K, physical FROGNet):
  A) pulse L1 on raw reconstruction
  B) pulse L1 on reconstruction after best-ambiguity alignment to GT

Validation traces use AWGN with SNR ~ U[-10, 30] dB (same range for both models).

Per-epoch logs: pulse L1 raw/amb (train+val), trace L1 (train+val),
gradient norms of L_data and L_reg, and CUDA/CPU timing breakdown.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from data_generation import stochastic_pulse_config_data_c
from dataset_utils import build_stochastic_frog_dataloaders
from frog_reconstruction_model import extract_pulse_prediction
from frognet import FROGNet
from pulse_metrics import (
    _ambiguity_bases,
    _best_shift_by_amplitude,
    _shift_field_zeros,
    align_pred_best_l1_ambiguity_torch_fast,
    best_l1_ambiguity,
    global_phase_min_l1,
    pack_complex_field,
    pulse_packed_l1_loss_torch,
    unpack_packed_field,
)
from trace_noise import add_trace_noise_awgn
from train import build_model

PulseLossMode = Literal["raw", "best_ambiguity"]
AmbiguityBackend = Literal["legacy", "fast"]
TraceLossRef = Literal["clean", "noisy"]
LoaderBuilder = Literal["data_c", "filtered_c1"]


def trace_l1_sum_batch_torch(i_pred: torch.Tensor, i_ref: torch.Tensor) -> torch.Tensor:
    return (i_pred - i_ref).abs().flatten(1).sum(dim=-1).mean()


def _trace_ref_for_batch(
    I_clean: torch.Tensor,
    I_noisy: torch.Tensor,
    trace_loss_ref: TraceLossRef,
) -> torch.Tensor:
    return I_noisy if trace_loss_ref == "noisy" else I_clean


def _sample_snr_db(
    snr_db_range: tuple[float, float],
    snr_db_values: list[float] | tuple[float, ...] | np.ndarray | None,
    *,
    values_name: str = "snr_db_values",
) -> float:
    """Sample one SNR: discrete grid if given, else continuous Uniform[lo, hi]."""
    if snr_db_values is not None:
        vals = np.asarray(snr_db_values, dtype=float).ravel()
        if vals.size == 0:
            raise ValueError(f"{values_name} must be non-empty")
        return float(vals[int(np.random.randint(0, vals.size))])
    snr_lo, snr_hi = map(float, snr_db_range)
    return float(np.random.uniform(snr_lo, snr_hi))


def _sample_train_snr_db(
    train_snr_db_range: tuple[float, float],
    train_snr_db_values: list[float] | tuple[float, ...] | np.ndarray | None,
) -> float:
    """Sample one train SNR (wrapper around ``_sample_snr_db``)."""
    return _sample_snr_db(
        train_snr_db_range,
        train_snr_db_values,
        values_name="train_snr_db_values",
    )


def calibrate_trace_scale(
    model: torch.nn.Module,
    frog: FROGNet,
    loader,
    *,
    device: torch.device,
    n_batches: int = 8,
    trace_loss_ref: TraceLossRef = "clean",
    train_snr_db_range: tuple[float, float] = (0.0, 30.0),
    train_snr_db_values: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> float:
    model.eval()
    ratios: list[float] = []
    with torch.no_grad():
        for bi, (I_clean, E_true) in enumerate(loader):
            if bi >= n_batches:
                break
            I_clean = I_clean.to(device)
            E_true = E_true.to(device)
            if trace_loss_ref == "noisy":
                snr = _sample_train_snr_db(train_snr_db_range, train_snr_db_values)
                I_in = add_trace_noise_awgn(I_clean, snr)
                I_ref = I_in
            else:
                I_in = I_clean
                I_ref = I_clean
            E_pred = extract_pulse_prediction(model(I_in.unsqueeze(1)))
            p = float(pulse_packed_l1_loss_torch(E_pred, E_true).item())
            t = float(trace_l1_sum_batch_torch(frog(E_pred), I_ref).item())
            if p > 1e-8:
                ratios.append(t / p)
    if not ratios:
        return float(64 * 64 / (2 * 64))
    return float(np.median(ratios))


def subset_loader(base, n: int):
    from torch.utils.data import DataLoader, Subset

    n = min(int(n), len(base.dataset))
    return DataLoader(
        Subset(base.dataset, range(n)),
        batch_size=min(int(base.batch_size), n),
        shuffle=False,
    )


@dataclass
class EpochTiming:
    data_prep_sec: float = 0.0
    loss_data_fwd_sec: float = 0.0
    loss_reg_fwd_sec: float = 0.0
    total_backward_sec: float = 0.0
    optimizer_step_sec: float = 0.0
    n_batches: int = 0

    def add(self, other: "EpochTiming") -> None:
        self.data_prep_sec += other.data_prep_sec
        self.loss_data_fwd_sec += other.loss_data_fwd_sec
        self.loss_reg_fwd_sec += other.loss_reg_fwd_sec
        self.total_backward_sec += other.total_backward_sec
        self.optimizer_step_sec += other.optimizer_step_sec
        self.n_batches += other.n_batches

    def mean_per_batch(self) -> dict[str, float]:
        n = max(self.n_batches, 1)
        return {
            "data_prep_sec": self.data_prep_sec / n,
            "loss_data_fwd_sec": self.loss_data_fwd_sec / n,
            "loss_reg_fwd_sec": self.loss_reg_fwd_sec / n,
            "total_backward_sec": self.total_backward_sec / n,
            "optimizer_step_sec": self.optimizer_step_sec / n,
        }


@dataclass
class DiagnosticHistory:
    train_pulse_l1_raw: list[float] = field(default_factory=list)
    train_pulse_l1_amb: list[float] = field(default_factory=list)
    val_pulse_l1_raw: list[float] = field(default_factory=list)
    val_pulse_l1_amb: list[float] = field(default_factory=list)
    train_trace_l1: list[float] = field(default_factory=list)
    val_trace_l1: list[float] = field(default_factory=list)
    grad_norm_data: list[float] = field(default_factory=list)
    grad_norm_reg: list[float] = field(default_factory=list)
    grad_norm_total: list[float] = field(default_factory=list)
    timing_per_epoch: list[dict[str, float]] = field(default_factory=list)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def history_from_npz_dict(hist: dict[str, np.ndarray] | dict) -> DiagnosticHistory:
    """Rebuild ``DiagnosticHistory`` from ``history_to_npz_dict`` / train_state payload."""
    out = DiagnosticHistory()
    for key in (
        "train_pulse_l1_raw",
        "train_pulse_l1_amb",
        "val_pulse_l1_raw",
        "val_pulse_l1_amb",
        "train_trace_l1",
        "val_trace_l1",
        "grad_norm_data",
        "grad_norm_reg",
        "grad_norm_total",
    ):
        if key in hist:
            setattr(out, key, [float(x) for x in np.asarray(hist[key]).ravel()])
    timing_keys = [
        "data_prep_sec",
        "loss_data_fwd_sec",
        "loss_reg_fwd_sec",
        "total_backward_sec",
        "optimizer_step_sec",
    ]
    n_ep = len(out.train_pulse_l1_raw)
    for i in range(n_ep):
        row = {}
        for k in timing_keys:
            arr = hist.get(f"timing_{k}")
            if arr is None:
                row[k] = 0.0
            else:
                a = np.asarray(arr).ravel()
                row[k] = float(a[i]) if i < a.size else 0.0
        out.timing_per_epoch.append(row)
    return out


def _cpu_state_dict(state: dict) -> dict:
    return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in state.items()}


def _normalize_snap_e_pred(e_pred) -> np.ndarray:
    """Normalize one snapshot E_pred to shape (1, D) float32 for safe stacking."""
    arr = np.asarray(e_pred, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim >= 2:
        arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[0] != 1:
            # packed payload row or batch — keep first sample for fixed-val probe
            arr = arr[:1]
    else:
        raise ValueError(f"Unexpected snapshot E_pred ndim={arr.ndim}")
    return arr


def _stack_snap_e_pred(items: list) -> np.ndarray:
    if not items:
        return np.zeros((0, 128), dtype=np.float32)
    rows = [_normalize_snap_e_pred(x) for x in items]
    return np.concatenate(rows, axis=0).astype(np.float32)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _now() -> float:
    return time.perf_counter()


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        total += float(p.grad.detach().float().norm(2).item() ** 2)
    return float(np.sqrt(total))


def _shift_field_zeros_torch(e: torch.Tensor, shift: int) -> torch.Tensor:
    """Non-circular zero-pad shift for complex 1D tensor [N]."""
    n = e.numel()
    shift = int(shift)
    if shift == 0:
        return e
    if abs(shift) >= n:
        return torch.zeros_like(e)
    out = torch.zeros_like(e)
    if shift > 0:
        out[shift:] = e[: n - shift]
    else:
        k = -shift
        out[: n - k] = e[k:]
    return out


def _best_amb_params(e_rec: np.ndarray, e_true: np.ndarray) -> tuple[int, int, float]:
    """Return (base_kind, shift, phi) minimizing packed L1 vs truth."""
    e_true = np.asarray(e_true, dtype=np.complex128).ravel()
    true_packed = pack_complex_field(e_true)
    best_val = np.inf
    best = (0, 0, 0.0)
    for bi, base in enumerate(_ambiguity_bases(e_rec)):
        k = _best_shift_by_amplitude(base, e_true)
        c = _shift_field_zeros(base, k)
        phi = global_phase_min_l1(c, e_true)
        e_ph = c * np.exp(1j * phi)
        val = float(np.abs(pack_complex_field(e_ph) - true_packed).sum())
        if val < best_val:
            best_val = val
            best = (bi, k, float(phi))
    return best


def align_pred_best_l1_ambiguity_torch(
    E_pred: torch.Tensor,
    E_true: torch.Tensor,
    *,
    backend: AmbiguityBackend = "legacy",
) -> torch.Tensor:
    """
    Align each predicted packed field to GT via best FROG ambiguity.

    Discrete choice (base / shift / φ) is selected without grad;
    the chosen transforms are applied in torch so gradients flow through E_pred.

    backend:
      - ``legacy``: per-sample NumPy loop (original)
      - ``fast``: batched GPU + FFT |E| shift (``align_pred_best_l1_ambiguity_torch_fast``)
    """
    if backend == "fast":
        return align_pred_best_l1_ambiguity_torch_fast(E_pred, E_true)
    if E_pred.ndim != 2 or E_true.ndim != 2:
        raise ValueError("E_pred / E_true must be [B, 2N]")
    bsz, two_n = E_pred.shape
    n = two_n // 2
    out = []
    e_pred_np = E_pred.detach().cpu().numpy()
    e_true_np = E_true.detach().cpu().numpy()
    for i in range(bsz):
        e_r = unpack_packed_field(e_pred_np[i])
        e_t = unpack_packed_field(e_true_np[i])
        bi, k, phi = _best_amb_params(e_r, e_t)

        re = E_pred[i, :n]
        im = E_pred[i, n:]
        e = torch.complex(re, im)
        if bi == 1:
            e = torch.conj(e)
        elif bi == 2:
            e = torch.flip(torch.conj(e), dims=[0])
        e = _shift_field_zeros_torch(e, k)
        phase = torch.tensor(phi, device=e.device, dtype=re.dtype)
        e = e * torch.exp(1j * phase)
        out.append(torch.cat([e.real, e.imag], dim=0))
    return torch.stack(out, dim=0)


def _batch_mean_best_l1_ambiguity(E_pred: torch.Tensor, E_true: torch.Tensor) -> float:
    e_pred = E_pred.detach().cpu().numpy()
    e_true = E_true.detach().cpu().numpy()
    vals = [
        best_l1_ambiguity(unpack_packed_field(e_pred[i]), unpack_packed_field(e_true[i]))
        for i in range(e_pred.shape[0])
    ]
    return float(np.mean(vals)) if vals else float("nan")


def _pulse_loss_for_train(
    E_pred: torch.Tensor,
    E_true: torch.Tensor,
    mode: PulseLossMode,
    *,
    ambiguity_backend: AmbiguityBackend = "legacy",
) -> torch.Tensor:
    if mode == "raw":
        return pulse_packed_l1_loss_torch(E_pred, E_true)
    E_aligned = align_pred_best_l1_ambiguity_torch(
        E_pred, E_true, backend=ambiguity_backend
    )
    return pulse_packed_l1_loss_torch(E_aligned, E_true)

@torch.no_grad()
def _eval_epoch_metrics(
    model: torch.nn.Module,
    loader,
    frog: FROGNet,
    *,
    device: torch.device,
    snr_db_range: tuple[float, float],
    snr_db_values: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> dict[str, float]:
    model.eval()
    sum_raw = sum_amb = sum_tr = 0.0
    n_seen = 0
    for I_clean, E_true in loader:
        I_clean = I_clean.to(device)
        E_true = E_true.to(device)
        snr = _sample_snr_db(
            snr_db_range, snr_db_values, values_name="val_snr_db_values"
        )
        I_noisy = add_trace_noise_awgn(I_clean, snr)
        E_pred = extract_pulse_prediction(model(I_noisy.unsqueeze(1)))
        b = I_clean.shape[0]
        sum_raw += float(pulse_packed_l1_loss_torch(E_pred, E_true).item()) * b
        sum_amb += _batch_mean_best_l1_ambiguity(E_pred, E_true) * b
        sum_tr += float(trace_l1_sum_batch_torch(frog(E_pred), I_clean).item()) * b
        n_seen += b
    denom = max(n_seen, 1)
    return {
        "pulse_l1_raw": sum_raw / denom,
        "pulse_l1_amb": sum_amb / denom,
        "trace_l1": sum_tr / denom,
    }


def _measure_grad_norms(
    model: torch.nn.Module,
    frog: FROGNet,
    I_clean: torch.Tensor,
    E_true: torch.Tensor,
    *,
    lam: float,
    trace_scale: float,
    pulse_loss_mode: PulseLossMode,
    snr_db: float,
    ambiguity_backend: AmbiguityBackend = "legacy",
    trace_loss_ref: TraceLossRef = "clean",
) -> tuple[float, float, float]:
    """Compute ||∇L_data||, ||∇L_reg||, and ||∇(L_data + λ L_reg)|| on one batch."""
    model.train()
    I_noisy = add_trace_noise_awgn(I_clean, float(snr_db))
    I_ref = _trace_ref_for_batch(I_clean, I_noisy, trace_loss_ref)
    E_pred = extract_pulse_prediction(model(I_noisy.unsqueeze(1)))

    model.zero_grad(set_to_none=True)
    l_data = _pulse_loss_for_train(
        E_pred, E_true, pulse_loss_mode, ambiguity_backend=ambiguity_backend
    )
    l_data.backward(retain_graph=True)
    g_data = _grad_norm(model)

    model.zero_grad(set_to_none=True)
    # Fresh graph for L_reg = trace_L1 / scale (λ not included in this probe)
    E_pred2 = extract_pulse_prediction(model(I_noisy.unsqueeze(1)))
    l_reg = trace_l1_sum_batch_torch(frog(E_pred2), I_ref) / max(float(trace_scale), 1e-8)
    l_reg.backward(retain_graph=True)
    g_reg = _grad_norm(model)

    model.zero_grad(set_to_none=True)
    E_pred3 = extract_pulse_prediction(model(I_noisy.unsqueeze(1)))
    l_data3 = _pulse_loss_for_train(
        E_pred3, E_true, pulse_loss_mode, ambiguity_backend=ambiguity_backend
    )
    l_reg3 = trace_l1_sum_batch_torch(frog(E_pred3), I_ref) / max(float(trace_scale), 1e-8)
    if float(lam) > 0.0:
        loss_total = l_data3 + float(lam) * l_reg3
    else:
        loss_total = l_data3
    loss_total.backward()
    g_total = _grad_norm(model)

    model.zero_grad(set_to_none=True)
    return g_data, g_reg, g_total


def train_data_c_amb_diagnostics(
    *,
    pulse_loss_mode: PulseLossMode,
    lam: float = 3.0,
    n_train: int = 2048,
    n_val: int = 200,
    n_test: int = 512,
    batch_size: int = 64,
    seed: int = 0,
    max_epochs: int = 200,
    patience: int = 25,
    lr: float = 1e-3,
    train_snr_db_range: tuple[float, float] = (0.0, 30.0),
    train_snr_db_values: list[float] | tuple[float, ...] | np.ndarray | None = None,
    val_snr_db_range: tuple[float, float] = (-10.0, 30.0),
    val_snr_db_values: list[float] | tuple[float, ...] | np.ndarray | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
    ambiguity_backend: AmbiguityBackend = "legacy",
    trace_loss_ref: TraceLossRef = "clean",
    loader_builder: LoaderBuilder = "data_c",
    canonicalize_mode: str = "t0",
    max_steps: int | None = None,
    fixed_trace_scale: float | None = None,
    snapshot_every: int = 0,
    snapshot_snr_db: float = 10.0,
    snapshot_val_index: int = 0,
    snapshot_noise_seed: int = 12345,
    early_stop_mode: Literal["epochs", "frac_best_step", "steps"] = "epochs",
    early_stop_frac: float = 0.9,
    resume_train_state: dict | None = None,
    extension_steps: int | None = None,
    save_full_train_state: bool = False,
) -> dict:
    """
    Train Multires with full diagnostic logging.

    ``loader_builder``:
      - ``\"data_c\"``: Data C stochastic pulses (legacy Model A/B)
      - ``\"filtered_c1\"``: spectrally filtered C1 from ``c1_pulse_independent_NB``

    Early-stop / best-checkpoint selection use **validation best-ambiguity**
    pulse L1 for both ``raw`` and ``best_ambiguity`` training modes.

    Train SNR: continuous ``train_snr_db_range`` unless ``train_snr_db_values``
    is set (uniform sample from that discrete grid each batch).

    Val SNR: continuous ``val_snr_db_range`` unless ``val_snr_db_values`` is set
    (same discrete sampling as train).

    Resume / extension (Protocol v2 Band B/C):
      - ``resume_train_state``: full train-state dict (model_last, optimizer, RNG,
        best_*, history, …) from a prior screen stop.
      - ``extension_steps``: if set with resume, continue for exactly this many
        additional optimizer steps; early-stopping is disabled during extension.
      - ``save_full_train_state``: include a resumable ``train_state`` in the
        returned dict (also written by ``save_run_artifacts`` when present).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if extension_steps is not None and resume_train_state is None:
        raise ValueError("extension_steps requires resume_train_state")
    if extension_steps is not None and int(extension_steps) < 1:
        raise ValueError(f"extension_steps must be >= 1, got {extension_steps}")

    t_data0 = _now()
    if loader_builder == "filtered_c1":
        from data_generation import filtered_c1_pulse_config
        from dataset_utils import build_filtered_c1_frog_dataloaders

        bundle = build_filtered_c1_frog_dataloaders(
            n_train=n_train,
            n_val=max(n_val, 64),
            n_test=n_test,
            batch_size=batch_size,
            seed=seed,
            device=device,
            grid=filtered_c1_pulse_config(n=64),
            canonicalize_mode=canonicalize_mode,
        )
    elif loader_builder == "data_c":
        bundle = build_stochastic_frog_dataloaders(
            n_train=n_train,
            n_val=max(n_val, 64),
            n_test=n_test,
            batch_size=batch_size,
            seed=seed,
            device=device,
            grid=stochastic_pulse_config_data_c(n=64),
            canonicalize_mode=canonicalize_mode,
        )
    else:
        raise ValueError(f"Unknown loader_builder={loader_builder!r}")
    wall_time_data_sec = _now() - t_data0
    train_loader = bundle.train_loader
    val_loader = subset_loader(bundle.val_loader, n_val)

    model = build_model(n=64, device=device, model_name="multires")
    frog = FROGNet(num_delay_steps=64).to(device)
    frog.eval()
    for p in frog.parameters():
        p.requires_grad_(False)

    if resume_train_state is not None and resume_train_state.get("trace_scale") is not None:
        trace_scale = float(resume_train_state["trace_scale"])
    elif fixed_trace_scale is not None:
        trace_scale = float(fixed_trace_scale)
    else:
        trace_scale = calibrate_trace_scale(
            model,
            frog,
            train_loader,
            device=device,
            trace_loss_ref=trace_loss_ref,
            train_snr_db_range=train_snr_db_range,
            train_snr_db_values=train_snr_db_values,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = DiagnosticHistory()

    best_score = float("inf")
    best_epoch = 0
    best_step = 0
    best_state: dict | None = None
    epochs_no_improve = 0
    stopped_epoch = 0
    global_step = 0
    hit_max_steps = False
    hit_extension_budget = False
    extension_start_step = 0
    start_epoch = 0
    steps_per_epoch_est = max(1, (int(n_train) + int(batch_size) - 1) // int(batch_size))
    if early_stop_mode not in ("epochs", "frac_best_step", "steps"):
        raise ValueError(f"Unknown early_stop_mode={early_stop_mode!r}")
    if early_stop_mode == "frac_best_step" and not (0.0 < float(early_stop_frac) <= 10.0):
        raise ValueError(f"early_stop_frac out of range: {early_stop_frac}")
    if early_stop_mode == "steps" and int(patience) < 1:
        raise ValueError(f"patience (steps) must be >= 1, got {patience}")
    snapshots: dict[str, list] = {
        "epoch": [],
        "step": [],
        "E_pred": [],
    }

    if resume_train_state is not None:
        rts = resume_train_state
        model.load_state_dict(rts["model_last"])
        optimizer.load_state_dict(rts["optimizer"])
        # Ensure optimizer tensors are on the right device
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        if rts.get("model_best") is not None:
            best_state = {k: v.to(device) if torch.is_tensor(v) else v for k, v in rts["model_best"].items()}
        best_score = float(rts.get("best_score", float("inf")))
        best_epoch = int(rts.get("best_epoch", 0))
        best_step = int(rts.get("best_step", 0))
        epochs_no_improve = int(rts.get("epochs_no_improve", 0))
        global_step = int(rts.get("global_step", 0))
        start_epoch = int(rts.get("stopped_epoch", rts.get("epoch", 0)))
        if rts.get("history") is not None:
            history = history_from_npz_dict(rts["history"])
        prev_snap = rts.get("snapshots")
        if isinstance(prev_snap, dict) and "epoch" in prev_snap:
            ep = np.asarray(prev_snap["epoch"]).ravel()
            st = np.asarray(prev_snap.get("step", np.zeros_like(ep))).ravel()
            ep_pred = prev_snap.get("E_pred")
            for i in range(len(ep)):
                snapshots["epoch"].append(int(ep[i]))
                snapshots["step"].append(int(st[i]) if i < len(st) else -1)
                if ep_pred is not None:
                    snapshots["E_pred"].append(_normalize_snap_e_pred(ep_pred[i]))
        _restore_rng_state(rts.get("rng"))
        if extension_steps is not None:
            extension_start_step = int(global_step)
            # Allow enough absolute max_steps headroom for the extension.
            need = int(global_step) + int(extension_steps)
            if max_steps is None or int(max_steps) < need:
                max_steps = need
            # Ensure enough epochs for extension (+ small buffer).
            need_epochs = start_epoch + max(1, math.ceil(int(extension_steps) / steps_per_epoch_est)) + 2
            if int(max_epochs) < need_epochs:
                max_epochs = int(need_epochs)
        if verbose:
            print(
                f"[resume] step={global_step} epoch={start_epoch} "
                f"best_step={best_step} best_score={best_score:.6f} "
                f"extension_steps={extension_steps}",
                flush=True,
            )

    # Fixed val probe for reconstruction snapshots (optional)
    snap_I_noisy = None
    snap_E_true = None
    snap_I_clean = None
    if int(snapshot_every) > 0:
        val_ds = val_loader.dataset
        idx = int(snapshot_val_index) % len(val_ds)
        I_clean_s, E_true_s = val_ds[idx]
        I_clean_s = I_clean_s.unsqueeze(0).to(device)
        E_true_s = E_true_s.unsqueeze(0).to(device)
        g_cpu = torch.Generator()
        g_cpu.manual_seed(int(snapshot_noise_seed))
        noise = torch.randn(
            I_clean_s.shape, generator=g_cpu, dtype=I_clean_s.dtype, device="cpu"
        ).to(device)
        pwr = I_clean_s.pow(2).mean().clamp_min(1e-12)
        snr_lin = 10.0 ** (float(snapshot_snr_db) / 10.0)
        sigma = torch.sqrt(pwr / snr_lin)
        snap_I_noisy = I_clean_s + sigma * noise
        snap_E_true = E_true_s
        snap_I_clean = I_clean_s

    scale = max(float(trace_scale), 1e-8)
    t_train0 = _now()
    train_snr_values_list = (
        [float(x) for x in np.asarray(train_snr_db_values, dtype=float).ravel()]
        if train_snr_db_values is not None
        else None
    )
    val_snr_values_list = (
        [float(x) for x in np.asarray(val_snr_db_values, dtype=float).ravel()]
        if val_snr_db_values is not None
        else None
    )

    # Cache one batch for end-of-epoch grad-norm probe
    probe_I, probe_E = next(iter(train_loader))
    probe_I = probe_I.to(device)
    probe_E = probe_E.to(device)

    for epoch in range(start_epoch, max_epochs):
        model.train()
        epoch_timing = EpochTiming()
        sum_raw = sum_amb = sum_tr = 0.0
        n_seen = 0

        for I_clean, E_true in train_loader:
            batch_t = EpochTiming(n_batches=1)

            _sync(device)
            t0 = _now()
            I_clean = I_clean.to(device)
            E_true = E_true.to(device)
            snr = _sample_train_snr_db(train_snr_db_range, train_snr_values_list)
            I_noisy = add_trace_noise_awgn(I_clean, snr)
            I_trace_ref = _trace_ref_for_batch(I_clean, I_noisy, trace_loss_ref)
            _sync(device)
            batch_t.data_prep_sec = _now() - t0

            _sync(device)
            t1 = _now()
            E_pred = extract_pulse_prediction(model(I_noisy.unsqueeze(1)))
            l_data = _pulse_loss_for_train(
                E_pred,
                E_true,
                pulse_loss_mode,
                ambiguity_backend=ambiguity_backend,
            )
            _sync(device)
            batch_t.loss_data_fwd_sec = _now() - t1

            _sync(device)
            t2 = _now()
            l_reg = trace_l1_sum_batch_torch(frog(E_pred), I_trace_ref) / scale
            if float(lam) > 0.0:
                loss = l_data + float(lam) * l_reg
            else:
                loss = l_data
            _sync(device)
            batch_t.loss_reg_fwd_sec = _now() - t2

            _sync(device)
            t3 = _now()
            loss.backward()
            _sync(device)
            batch_t.total_backward_sec = _now() - t3

            _sync(device)
            t4 = _now()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            _sync(device)
            batch_t.optimizer_step_sec = _now() - t4

            epoch_timing.add(batch_t)
            global_step += 1

            # Logging metrics (outside the five timing buckets; cheap detach)
            with torch.no_grad():
                b = I_clean.shape[0]
                sum_raw += float(pulse_packed_l1_loss_torch(E_pred, E_true).item()) * b
                sum_amb += _batch_mean_best_l1_ambiguity(E_pred, E_true) * b
                sum_tr += float(trace_l1_sum_batch_torch(frog(E_pred), I_clean).item()) * b
                n_seen += b

            if max_steps is not None and global_step >= int(max_steps):
                hit_max_steps = True
                break
            if (
                extension_steps is not None
                and (int(global_step) - int(extension_start_step)) >= int(extension_steps)
            ):
                hit_extension_budget = True
                break

        denom = max(n_seen, 1)
        history.train_pulse_l1_raw.append(sum_raw / denom)
        history.train_pulse_l1_amb.append(sum_amb / denom)
        history.train_trace_l1.append(sum_tr / denom)
        history.timing_per_epoch.append(epoch_timing.mean_per_batch())

        # Gradient norms — after timing, one probe batch
        g_data, g_reg, g_total = _measure_grad_norms(
            model,
            frog,
            probe_I,
            probe_E,
            lam=float(lam),
            trace_scale=scale,
            pulse_loss_mode=pulse_loss_mode,
            snr_db=_sample_train_snr_db(train_snr_db_range, train_snr_values_list),
            ambiguity_backend=ambiguity_backend,
            trace_loss_ref=trace_loss_ref,
        )
        history.grad_norm_data.append(g_data)
        history.grad_norm_reg.append(g_reg)
        history.grad_norm_total.append(g_total)

        val_m = _eval_epoch_metrics(
            model,
            val_loader,
            frog,
            device=device,
            snr_db_range=val_snr_db_range,
            snr_db_values=val_snr_values_list,
        )
        history.val_pulse_l1_raw.append(val_m["pulse_l1_raw"])
        history.val_pulse_l1_amb.append(val_m["pulse_l1_amb"])
        history.val_trace_l1.append(val_m["trace_l1"])

        score = val_m["pulse_l1_amb"]
        if score < best_score:
            best_score = score
            best_epoch = epoch + 1
            best_step = int(global_step)
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Reconstruction snapshots every snapshot_every epochs
        ep1 = epoch + 1
        take_snap = (
            int(snapshot_every) > 0
            and snap_I_noisy is not None
            and (ep1 % int(snapshot_every) == 0)
        )
        if take_snap:
            model.eval()
            with torch.no_grad():
                E_s = extract_pulse_prediction(model(snap_I_noisy.unsqueeze(1)))
            model.train()
            snapshots["epoch"].append(ep1)
            snapshots["step"].append(int(global_step))
            snapshots["E_pred"].append(_normalize_snap_e_pred(E_s.detach().cpu().numpy()))

        if verbose:
            tm = history.timing_per_epoch[-1]
            print(
                f"[{pulse_loss_mode}|{ambiguity_backend}] epoch {ep1:03d}/{max_epochs}  "
                f"step={global_step}  "
                f"tr_raw={history.train_pulse_l1_raw[-1]:.4f}  "
                f"tr_amb={history.train_pulse_l1_amb[-1]:.4f}  "
                f"va_raw={history.val_pulse_l1_raw[-1]:.4f}  "
                f"va_amb={history.val_pulse_l1_amb[-1]:.4f}  "
                f"tr_trace={history.train_trace_l1[-1]:.2f}  "
                f"va_trace={history.val_trace_l1[-1]:.2f}  "
                f"|g_data|={g_data:.3e}  |g_reg|={g_reg:.3e}  |g_tot|={g_total:.3e}  "
                f"t_prep={tm['data_prep_sec']*1e3:.2f}ms  "
                f"t_Ld={tm['loss_data_fwd_sec']*1e3:.2f}ms  "
                f"t_Lr={tm['loss_reg_fwd_sec']*1e3:.2f}ms  "
                f"t_bwd={tm['total_backward_sec']*1e3:.2f}ms  "
                f"t_opt={tm['optimizer_step_sec']*1e3:.2f}ms",
                flush=True,
            )

        stop_early = False
        if extension_steps is None:
            if early_stop_mode == "epochs":
                stop_early = epochs_no_improve >= patience
            elif early_stop_mode == "steps":
                # Fixed patience in optimizer steps (checked after each epoch val).
                if best_step > 0:
                    stop_early = (int(global_step) - int(best_step)) >= int(patience)
            else:
                # frac_best_step: (t - t_best) >= max(frac * t_best, K)
                if best_step > 0:
                    patience_steps = max(
                        float(early_stop_frac) * float(best_step),
                        float(steps_per_epoch_est),
                    )
                    stop_early = (float(global_step) - float(best_step)) >= patience_steps

        if stop_early:
            stopped_epoch = ep1
            if verbose:
                if early_stop_mode == "epochs":
                    print(
                        f"[{pulse_loss_mode}|{ambiguity_backend}] early stop at epoch {stopped_epoch}, "
                        f"best epoch {best_epoch}",
                        flush=True,
                    )
                elif early_stop_mode == "steps":
                    print(
                        f"[{pulse_loss_mode}|{ambiguity_backend}] early stop at epoch {stopped_epoch} "
                        f"(step={global_step}), best epoch {best_epoch} (best_step={best_step}), "
                        f"rule=steps(patience={int(patience)})",
                        flush=True,
                    )
                else:
                    print(
                        f"[{pulse_loss_mode}|{ambiguity_backend}] early stop at epoch {stopped_epoch} "
                        f"(step={global_step}), best epoch {best_epoch} (best_step={best_step}), "
                        f"rule=frac_best_step({early_stop_frac:g})",
                        flush=True,
                    )
            break
        if hit_extension_budget:
            stopped_epoch = ep1
            if verbose:
                print(
                    f"[{pulse_loss_mode}|{ambiguity_backend}] extension done at epoch {stopped_epoch} "
                    f"(step={global_step}, +{int(global_step) - int(extension_start_step)} steps), "
                    f"best epoch {best_epoch} (best_step={best_step})",
                    flush=True,
                )
            break
        if hit_max_steps:
            stopped_epoch = ep1
            if verbose:
                print(
                    f"[{pulse_loss_mode}|{ambiguity_backend}] hit max_steps={max_steps} "
                    f"at epoch {stopped_epoch}, best epoch {best_epoch}",
                    flush=True,
                )
            break
    else:
        stopped_epoch = max_epochs

    # Capture stop-point weights/optimizer/RNG BEFORE swapping model to best.
    last_model_state = copy.deepcopy(model.state_dict())
    last_optimizer_state = copy.deepcopy(optimizer.state_dict())
    rng_at_stop = _capture_rng_state()

    # Ensure snapshot at best / stop epochs if missing
    if int(snapshot_every) > 0 and snap_I_noisy is not None and best_state is not None:
        need_epochs = {int(best_epoch), int(stopped_epoch)}
        have = set(int(e) for e in snapshots["epoch"])
        model.load_state_dict(best_state)
        # For stop epoch that is not best, need current weights — reload chronologically:
        # After loop, model is last-epoch weights; best_state is best.
        # Save best snapshot under best_epoch if missing using best weights.
        if int(best_epoch) not in have:
            model.eval()
            with torch.no_grad():
                E_s = extract_pulse_prediction(model(snap_I_noisy.unsqueeze(1)))
            snapshots["epoch"].append(int(best_epoch))
            snapshots["step"].append(-1)
            snapshots["E_pred"].append(_normalize_snap_e_pred(E_s.detach().cpu().numpy()))
            have.add(int(best_epoch))
        if int(stopped_epoch) not in have and int(stopped_epoch) != int(best_epoch):
            # Last-epoch weights may have been overwritten only if we loaded best above.
            # Re-run forward with best is wrong for stop; skip if not available.
            pass

    if best_state is not None:
        model.load_state_dict(best_state)

    wall_time_train_sec = _now() - t_train0

    snap_payload = None
    if int(snapshot_every) > 0 and snap_I_noisy is not None:
        # Sort by epoch
        order = np.argsort(np.asarray(snapshots["epoch"], dtype=int))
        snap_payload = {
            "epoch": np.asarray(snapshots["epoch"], dtype=np.int32)[order],
            "step": np.asarray(snapshots["step"], dtype=np.int64)[order],
            "E_pred": _stack_snap_e_pred([snapshots["E_pred"][i] for i in order]),
            "I_noisy": snap_I_noisy.detach().cpu().numpy().astype(np.float32),
            "I_clean": snap_I_clean.detach().cpu().numpy().astype(np.float32),
            "E_true": snap_E_true.detach().cpu().numpy().astype(np.float32),
            "snr_db": float(snapshot_snr_db),
            "val_index": int(snapshot_val_index),
            "noise_seed": int(snapshot_noise_seed),
        }

    train_state = None
    if save_full_train_state:
        train_state = {
            "model_last": _cpu_state_dict(last_model_state),
            "model_best": None if best_state is None else _cpu_state_dict(best_state),
            "optimizer": last_optimizer_state,
            "rng": rng_at_stop,
            "global_step": int(global_step),
            "stopped_epoch": int(stopped_epoch),
            "best_score": float(best_score),
            "best_step": int(best_step),
            "best_epoch": int(best_epoch),
            "epochs_no_improve": int(epochs_no_improve),
            "history": history_to_npz_dict(history),
            "snapshots": snap_payload,
            "lam": float(lam),
            "trace_scale": float(scale),
            "lr": float(lr),
            "patience": int(patience),
            "early_stop_mode": str(early_stop_mode),
            "extension_steps": None if extension_steps is None else int(extension_steps),
            "extension_start_step": int(extension_start_step) if extension_steps is not None else None,
        }

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "stopped_epoch": stopped_epoch,
        "lam": float(lam),
        "trace_scale": scale,
        "pulse_loss_mode": pulse_loss_mode,
        "ambiguity_backend": ambiguity_backend,
        "trace_loss_ref": trace_loss_ref,
        "device": str(device),
        "train_snr_db_range": tuple(map(float, train_snr_db_range)),
        "train_snr_db_values": train_snr_values_list,
        "val_snr_db_range": tuple(map(float, val_snr_db_range)),
        "val_snr_db_values": val_snr_values_list,
        "n_train": n_train,
        "n_val": n_val,
        "seed": seed,
        "loader_builder": loader_builder,
        "canonicalize_mode": canonicalize_mode,
        "bundle": bundle,
        "lr": float(lr),
        "batch_size": int(batch_size),
        "max_steps": None if max_steps is None else int(max_steps),
        "global_step": int(global_step),
        "best_step": int(best_step),
        "early_stop_mode": str(early_stop_mode),
        "early_stop_frac": float(early_stop_frac),
        "patience": int(patience),
        "wall_time_data_sec": float(wall_time_data_sec),
        "wall_time_train_sec": float(wall_time_train_sec),
        "snapshots": snap_payload,
        "train_state": train_state,
        "extension_steps": None if extension_steps is None else int(extension_steps),
        "hit_extension_budget": bool(hit_extension_budget),
    }


def history_to_npz_dict(history: DiagnosticHistory) -> dict[str, np.ndarray]:
    timing_keys = [
        "data_prep_sec",
        "loss_data_fwd_sec",
        "loss_reg_fwd_sec",
        "total_backward_sec",
        "optimizer_step_sec",
    ]
    out: dict[str, np.ndarray] = {
        "train_pulse_l1_raw": np.asarray(history.train_pulse_l1_raw, dtype=np.float64),
        "train_pulse_l1_amb": np.asarray(history.train_pulse_l1_amb, dtype=np.float64),
        "val_pulse_l1_raw": np.asarray(history.val_pulse_l1_raw, dtype=np.float64),
        "val_pulse_l1_amb": np.asarray(history.val_pulse_l1_amb, dtype=np.float64),
        "train_trace_l1": np.asarray(history.train_trace_l1, dtype=np.float64),
        "val_trace_l1": np.asarray(history.val_trace_l1, dtype=np.float64),
        "grad_norm_data": np.asarray(history.grad_norm_data, dtype=np.float64),
        "grad_norm_reg": np.asarray(history.grad_norm_reg, dtype=np.float64),
        "grad_norm_total": np.asarray(history.grad_norm_total, dtype=np.float64),
    }
    for k in timing_keys:
        out[f"timing_{k}"] = np.asarray(
            [ep[k] for ep in history.timing_per_epoch], dtype=np.float64
        )
    return out


def save_run_artifacts(result: dict, out_dir: Path, tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = history_to_npz_dict(result["history"])
    np.savez_compressed(out_dir / f"{tag}_history.npz", **hist)
    torch.save(result["model"].state_dict(), out_dir / f"{tag}_model.pt")
    snap = result.get("snapshots")
    if snap is not None:
        np.savez_compressed(out_dir / f"{tag}_snapshots.npz", **snap)
    train_state = result.get("train_state")
    if train_state is not None:
        torch.save(train_state, out_dir / f"{tag}_train_state.pt")
    meta = {
        k: result[k]
        for k in (
            "best_epoch",
            "best_score",
            "stopped_epoch",
            "lam",
            "trace_scale",
            "pulse_loss_mode",
            "ambiguity_backend",
            "trace_loss_ref",
            "device",
            "train_snr_db_range",
            "train_snr_db_values",
            "val_snr_db_range",
            "val_snr_db_values",
            "n_train",
            "n_val",
            "seed",
            "loader_builder",
            "canonicalize_mode",
            "lr",
            "batch_size",
            "max_steps",
            "global_step",
            "best_step",
            "early_stop_mode",
            "early_stop_frac",
            "patience",
            "wall_time_data_sec",
            "wall_time_train_sec",
            "extension_steps",
            "hit_extension_budget",
        )
        if k in result
    }
    meta["has_train_state"] = train_state is not None
    meta.setdefault("ambiguity_backend", "legacy")
    meta.setdefault("trace_loss_ref", "clean")
    meta.setdefault("train_snr_db_values", None)
    meta.setdefault("val_snr_db_values", None)
    meta.setdefault("loader_builder", "data_c")
    meta.setdefault("canonicalize_mode", "t0")
    (out_dir / f"{tag}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def model_a_noisy_trace_lambda_tag(lam: float) -> str:
    lam_s = f"{float(lam):g}".replace(".", "p")
    return f"model_a_raw_pulse_loss_noisy_trace_lam{lam_s}"


def run_model_a_noisy_trace_lambda_sweep(
    lambda_values: list[float],
    out_dir: Path,
    *,
    force: bool = False,
    n_train: int = 2048,
    n_val: int = 200,
    n_test: int = 512,
    batch_size: int = 64,
    seed: int = 0,
    max_epochs: int = 200,
    patience: int = 25,
    lr: float = 1e-3,
    train_snr_db_range: tuple[float, float] = (0.0, 30.0),
    val_snr_db_range: tuple[float, float] = (-10.0, 30.0),
    device: torch.device | None = None,
    verbose: bool = True,
    winner_tag: str = "model_a_raw_pulse_loss_noisy_trace",
) -> dict:
    """
  Train Model A (raw pulse loss) with noisy trace in the trace-loss term.

  Sweeps ``lambda_values``; winner = lowest validation pulse L1 after best ambiguity.
  Saves per-λ artifacts plus a copy of the winner under ``winner_tag``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_lam: dict[float, dict] = {}
    best_lam: float | None = None
    best_score = float("inf")
    best_result: dict | None = None

    for lam in lambda_values:
        lam_f = float(lam)
        tag = model_a_noisy_trace_lambda_tag(lam_f)
        hist_path = out_dir / f"{tag}_history.npz"
        if not force and hist_path.exists():
            print(f"skip λ={lam_f:g} (exists): {tag}", flush=True)
            meta = json.loads((out_dir / f"{tag}_meta.json").read_text(encoding="utf-8"))
            res_stub = {
                "best_epoch": meta["best_epoch"],
                "best_score": meta["best_score"],
                "stopped_epoch": meta["stopped_epoch"],
                "lam": lam_f,
                "trace_loss_ref": meta.get("trace_loss_ref", "noisy"),
                "history": None,
            }
            per_lam[lam_f] = res_stub
            score = float(meta["best_score"])
            if score < best_score:
                best_score = score
                best_lam = lam_f
                best_result = res_stub
            continue

        print(f"Training Model A noisy-trace λ={lam_f:g} ...", flush=True)
        res = train_data_c_amb_diagnostics(
            pulse_loss_mode="raw",
            lam=lam_f,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            batch_size=batch_size,
            seed=seed,
            max_epochs=max_epochs,
            patience=patience,
            lr=lr,
            train_snr_db_range=train_snr_db_range,
            val_snr_db_range=val_snr_db_range,
            device=device,
            verbose=verbose,
            trace_loss_ref="noisy",
        )
        save_run_artifacts(res, out_dir, tag)
        per_lam[lam_f] = res
        if float(res["best_score"]) < best_score:
            best_score = float(res["best_score"])
            best_lam = lam_f
            best_result = res

    if best_lam is None or best_result is None:
        raise RuntimeError("lambda sweep produced no results")

    # Copy winner artifacts
    winner_src_tag = model_a_noisy_trace_lambda_tag(best_lam)
    if best_result.get("history") is not None:
        save_run_artifacts(best_result, out_dir, winner_tag)
    else:
        for suffix in ("_history.npz", "_model.pt", "_meta.json"):
            src = out_dir / f"{winner_src_tag}{suffix}"
            dst = out_dir / f"{winner_tag}{suffix}"
            if src.exists():
                dst.write_bytes(src.read_bytes())

    summary = {
        "lambda_values": [float(x) for x in lambda_values],
        "best_lam": float(best_lam),
        "best_score": float(best_score),
        "winner_tag": winner_tag,
        "per_lam_best_score": {
            float(lam): float(per_lam[float(lam)]["best_score"])
            for lam in lambda_values
        },
        "trace_loss_ref": "noisy",
        "selection_metric": "val_pulse_l1_amb",
    }
    (out_dir / "model_a_noisy_trace_lambda_sweep_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"λ*={best_lam:g}  best val L1-amb={best_score:.4f}  winner={winner_tag}",
        flush=True,
    )
    return {
        "per_lam": per_lam,
        "best_lam": float(best_lam),
        "best_score": float(best_score),
        "winner_tag": winner_tag,
        "summary": summary,
    }


def load_history(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def build_data_c_test_loader(
    *,
    n_train: int = 2048,
    n_val: int = 200,
    n_test: int = 512,
    batch_size: int = 64,
    seed: int = 0,
    device: torch.device | None = None,
):
    """Held-out Data C test loader (seed+2 pulses; never used for early stop)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_stochastic_frog_dataloaders(
        n_train=n_train,
        n_val=max(n_val, 64),
        n_test=n_test,
        batch_size=batch_size,
        seed=seed,
        device=device,
        grid=stochastic_pulse_config_data_c(n=64),
        canonicalize_mode="t0",
    )
    return subset_loader(bundle.test_loader, n_test)


def load_trained_multires(checkpoint_pt: Path, *, device: torch.device) -> torch.nn.Module:
    model = build_model(n=64, device=device, model_name="multires")
    state = torch.load(checkpoint_pt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def run_and_save_test_snr_sweep(
    checkpoint_pt: Path,
    sweep_out: Path,
    *,
    test_loader,
    snr_sweep_db: np.ndarray,
    device: torch.device,
    experiment_name: str,
    verbose: bool = True,
):
    """Evaluate a saved Multires checkpoint on the held-out test SNR sweep."""
    from evaluate_cnn import run_cnn_snr_sweep, save_cnn_sweep

    model = load_trained_multires(checkpoint_pt, device=device)
    result = run_cnn_snr_sweep(
        model,
        test_loader,
        np.asarray(snr_sweep_db, dtype=float),
        experiment_name=experiment_name,
        verbose=verbose,
    )
    save_cnn_sweep(sweep_out, result)
    return result


@torch.no_grad()
def _forward_loader_predictions(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    snr_db_range: tuple[float, float] | None = None,
    fixed_snr_db: float | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
    """Run inference; return per-batch packed ``E_pred`` / ``E_true`` on device."""
    batches_pred: list[torch.Tensor] = []
    batches_true: list[torch.Tensor] = []
    n_seen = 0
    for I_clean, E_true in loader:
        I_clean = I_clean.to(device)
        E_true = E_true.to(device)
        if fixed_snr_db is not None:
            I_in = add_trace_noise_awgn(I_clean, float(fixed_snr_db))
        elif snr_db_range is not None:
            snr = float(np.random.uniform(float(snr_db_range[0]), float(snr_db_range[1])))
            I_in = add_trace_noise_awgn(I_clean, snr)
        else:
            I_in = I_clean
        E_pred = extract_pulse_prediction(model(I_in.unsqueeze(1)))
        batches_pred.append(E_pred)
        batches_true.append(E_true)
        n_seen += int(E_pred.shape[0])
    return batches_pred, batches_true, n_seen


def _time_callable_sec(
    fn,
    *,
    device: torch.device,
    n_warmup: int = 2,
    n_repeat: int = 5,
) -> dict[str, float]:
    for _ in range(n_warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(n_repeat):
        _sync(device)
        t0 = _now()
        fn()
        _sync(device)
        times.append(_now() - t0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_sec": float(arr.mean()),
        "std_sec": float(arr.std()),
        "min_sec": float(arr.min()),
        "max_sec": float(arr.max()),
    }


def _bench_ambiguity_backend(
    batches_pred: list[torch.Tensor],
    batches_true: list[torch.Tensor],
    *,
    backend: AmbiguityBackend,
    device: torch.device,
    n_warmup: int = 2,
    n_repeat: int = 5,
) -> dict[str, float]:
    def _run() -> None:
        for ep, et in zip(batches_pred, batches_true):
            align_pred_best_l1_ambiguity_torch(ep, et, backend=backend)

    return _time_callable_sec(_run, device=device, n_warmup=n_warmup, n_repeat=n_repeat)


def _bench_ambiguity_legacy_numpy(
    batches_pred: list[torch.Tensor],
    batches_true: list[torch.Tensor],
    *,
    device: torch.device,
    n_warmup: int = 2,
    n_repeat: int = 5,
) -> dict[str, float]:
    """Same per-sample NumPy path used in ``evaluate_cnn`` SNR sweeps."""

    def _run() -> None:
        for ep, et in zip(batches_pred, batches_true):
            ep_np = ep.detach().cpu().numpy()
            et_np = et.detach().cpu().numpy()
            for i in range(ep_np.shape[0]):
                best_l1_ambiguity(
                    unpack_packed_field(ep_np[i]),
                    unpack_packed_field(et_np[i]),
                )

    return _time_callable_sec(_run, device=device, n_warmup=n_warmup, n_repeat=n_repeat)


def _verify_ambiguity_backends(
    batches_pred: list[torch.Tensor],
    batches_true: list[torch.Tensor],
    *,
    max_samples: int = 64,
) -> dict[str, float]:
    """Check legacy NumPy vs fast NumPy L1 on cached predictions."""
    from pulse_metrics import best_l1_ambiguity_fast

    n_check = 0
    max_abs_diff = 0.0
    for ep, et in zip(batches_pred, batches_true):
        ep_np = ep.detach().cpu().numpy()
        et_np = et.detach().cpu().numpy()
        for i in range(ep_np.shape[0]):
            e_r = unpack_packed_field(ep_np[i])
            e_t = unpack_packed_field(et_np[i])
            l_legacy = best_l1_ambiguity(e_r, e_t)
            l_fast = best_l1_ambiguity_fast(e_r, e_t)
            max_abs_diff = max(max_abs_diff, abs(l_legacy - l_fast))
            n_check += 1
            if n_check >= max_samples:
                return {
                    "n_checked": n_check,
                    "max_abs_l1_diff_legacy_vs_fast": max_abs_diff,
                }
    return {
        "n_checked": n_check,
        "max_abs_l1_diff_legacy_vs_fast": max_abs_diff,
    }


def benchmark_ambiguity_inference(
    checkpoint_pt: Path,
    *,
    val_loader,
    test_loader,
    device: torch.device | None = None,
    val_snr_db_range: tuple[float, float] = (-10.0, 30.0),
    test_snr_sweep_db: np.ndarray | None = None,
    n_warmup: int = 2,
    n_repeat: int = 5,
    seed: int = 0,
) -> dict:
    """
    Benchmark best-ambiguity **inference only** on cached Model A predictions.

    Protocol: one forward pass per split (or per SNR point), then time ambiguity
    backends on the same ``E_pred`` tensors (no retraining).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if test_snr_sweep_db is None:
        test_snr_sweep_db = np.arange(-10.0, 31.0, 5.0)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    model = load_trained_multires(checkpoint_pt, device=device)

    # --- validation (random SNR each batch, same as val during training) ---
    val_pred, val_true, n_val = _forward_loader_predictions(
        model, val_loader, device=device, snr_db_range=val_snr_db_range
    )

    def _time_forward_val() -> None:
        _forward_loader_predictions(
            model, val_loader, device=device, snr_db_range=val_snr_db_range
        )

    val_forward = _time_callable_sec(
        _time_forward_val, device=device, n_warmup=n_warmup, n_repeat=n_repeat
    )
    val_legacy_torch = _bench_ambiguity_backend(
        val_pred, val_true, backend="legacy", device=device, n_warmup=n_warmup, n_repeat=n_repeat
    )
    val_fast = _bench_ambiguity_backend(
        val_pred, val_true, backend="fast", device=device, n_warmup=n_warmup, n_repeat=n_repeat
    )
    val_legacy_numpy = _bench_ambiguity_legacy_numpy(
        val_pred, val_true, device=device, n_warmup=n_warmup, n_repeat=n_repeat
    )
    val_verify = _verify_ambiguity_backends(val_pred, val_true)

    # --- test SNR sweep (fixed SNR per point, full test loader) ---
    per_snr: list[dict] = []
    test_forward_total = 0.0
    test_legacy_torch_total = 0.0
    test_fast_total = 0.0
    test_legacy_numpy_total = 0.0
    n_test_total = 0

    for snr_db in np.asarray(test_snr_sweep_db, dtype=float):
        pred, true, n_test = _forward_loader_predictions(
            model, test_loader, device=device, fixed_snr_db=float(snr_db)
        )
        n_test_total += n_test
        fwd = _time_callable_sec(
            lambda p=pred, t=true, s=float(snr_db): _forward_loader_predictions(
                model, test_loader, device=device, fixed_snr_db=s
            ),
            device=device,
            n_warmup=1,
            n_repeat=3,
        )
        leg = _bench_ambiguity_backend(
            pred, true, backend="legacy", device=device, n_warmup=1, n_repeat=3
        )
        fast = _bench_ambiguity_backend(
            pred, true, backend="fast", device=device, n_warmup=1, n_repeat=3
        )
        leg_np = _bench_ambiguity_legacy_numpy(
            pred, true, device=device, n_warmup=1, n_repeat=3
        )
        per_snr.append(
            {
                "snr_db": float(snr_db),
                "n_samples": n_test,
                "forward_mean_sec": fwd["mean_sec"],
                "legacy_torch_mean_sec": leg["mean_sec"],
                "fast_torch_mean_sec": fast["mean_sec"],
                "legacy_numpy_mean_sec": leg_np["mean_sec"],
            }
        )
        test_forward_total += fwd["mean_sec"]
        test_legacy_torch_total += leg["mean_sec"]
        test_fast_total += fast["mean_sec"]
        test_legacy_numpy_total += leg_np["mean_sec"]

    def _ms_per_sample(total_sec: float, n: int) -> float:
        return float(total_sec / max(n, 1) * 1e3)

    out = {
        "checkpoint": str(checkpoint_pt),
        "device": str(device),
        "n_warmup": n_warmup,
        "n_repeat": n_repeat,
        "validation": {
            "n_samples": n_val,
            "snr_db_range": tuple(map(float, val_snr_db_range)),
            "verify": val_verify,
            "forward": val_forward,
            "legacy_torch": val_legacy_torch,
            "fast_torch": val_fast,
            "legacy_numpy": val_legacy_numpy,
            "forward_ms_per_sample": _ms_per_sample(val_forward["mean_sec"], n_val),
            "legacy_torch_ms_per_sample": _ms_per_sample(
                val_legacy_torch["mean_sec"], n_val
            ),
            "fast_torch_ms_per_sample": _ms_per_sample(val_fast["mean_sec"], n_val),
            "legacy_numpy_ms_per_sample": _ms_per_sample(
                val_legacy_numpy["mean_sec"], n_val
            ),
            "speedup_legacy_torch_vs_fast": val_legacy_torch["mean_sec"]
            / max(val_fast["mean_sec"], 1e-12),
            "speedup_legacy_numpy_vs_fast": val_legacy_numpy["mean_sec"]
            / max(val_fast["mean_sec"], 1e-12),
        },
        "test_snr_sweep": {
            "snr_points": [float(x) for x in test_snr_sweep_db],
            "n_samples_per_point": n_test,
            "n_points": len(per_snr),
            "per_snr": per_snr,
            "total_forward_sec": test_forward_total,
            "total_legacy_torch_sec": test_legacy_torch_total,
            "total_fast_torch_sec": test_fast_total,
            "total_legacy_numpy_sec": test_legacy_numpy_total,
            "total_samples": n_test_total,
            "legacy_torch_ms_per_sample": _ms_per_sample(
                test_legacy_torch_total, n_test_total
            ),
            "fast_torch_ms_per_sample": _ms_per_sample(test_fast_total, n_test_total),
            "legacy_numpy_ms_per_sample": _ms_per_sample(
                test_legacy_numpy_total, n_test_total
            ),
            "speedup_legacy_torch_vs_fast": test_legacy_torch_total
            / max(test_fast_total, 1e-12),
            "speedup_legacy_numpy_vs_fast": test_legacy_numpy_total
            / max(test_fast_total, 1e-12),
        },
    }
    return out


def save_ambiguity_inference_benchmark(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
