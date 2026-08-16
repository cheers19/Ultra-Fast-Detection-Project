# Data-Efficiency Experiment — Training & Evaluation Protocol (VERSION 2)

> **Status:** PROTOCOL **v2** (supersedes v1 rules for new runs).  
> Keep [`PROTOCOL.txt`](PROTOCOL.txt) / [`PROTOCOL.md`](PROTOCOL.md) as the historical **v1** reference.  
> Text mirror: [`PROTOCOL_v2.txt`](PROTOCOL_v2.txt).

This document is the single source of truth for **v2** training and evaluation of **plain** (\(\lambda=0\)) and **physics** (\(\lambda>0\)) Multires models across training-set sizes.

Code in this folder may import existing modules from the parent `src` directory. From notebooks or scripts here:

```python
import setup_src_path  # adds parent src/ to sys.path
```

**Logging / plots / SNR sweep / test @ 10 dB** should match the spirit of:

- `data efficiency experiment/plain_multires_n2498_NB.ipynb`
- `filtered_c1_multires_2k_diagnostics_NB.ipynb`

---

## 1. Experiment scope

For **each** training size \(n_{\mathrm{train}}\) below, train and evaluate:

1. **Plain network:** \(\lambda = 0\)
2. **Physics network:** \(\lambda > 0\) (TRACE loss), with **`trace_scale = 8` always**

### Training-set sizes

\[
n_{\mathrm{train}} \in \{300,\ 866,\ 2498,\ 7207,\ 20794,\ 60000\}
\]

Process sizes from small to large. For each \(n\):

**plain** (fixed LR, one train) → **physics** (\(\lambda\) tune per §5 band rules → official physics checkpoint) → shared test.

### Data family (all models)

Spectrally **filtered C1** only (same as `filtered_c1_multires_2k_diagnostics_NB.ipynb`). Do **not** mix other generators.

---

## 2. Shared fixed settings (all models, all \(n\))

| Item | Value |
|---|---|
| Architecture | Multires |
| Pulse / TRACE data | Filtered C1 only |
| Batch size \(B\) | **64** |
| `drop_last` | `False` |
| Optimizer | Adam |
| Learning rate (plain & physics) | **\(10^{-3}\)** fixed (**no LR tuning**) |
| \(n_{\mathrm{val}}\) | **200** — identical for every model / \(n\) |
| \(n_{\mathrm{test}}\) | Fixed (e.g. 512) — identical for every model / \(n\) |
| Train SNR | Uniform in \([0,\ 30]\) dB |
| **Val SNR** | **Discrete** \(\{-10,\ 0,\ 30\}\) dB (**not** \(U[0,30]\)) |
| Selection metric | Val pulse L1 after best ambiguity |
| \(K\) | \(K=\lceil n_{\mathrm{train}}/B\rceil\) (informational) |
| Early stopping | Validate every epoch; stop when \((t-t_{\mathrm{best}})\ge patience\_steps\) (`early_stop_mode="steps"`); patience from §2.1 (plain) or §5 (physics screens) |
| Physics TRACE scale | **`trace_scale = 8`** always |
| Canonicalization | `t0` |
| Seeds | Val/test fixed once; train may depend on \(n\) |

### 2.1 Step budgets and patience (plain; physics defaults)

**Plain** always uses this table. **Physics** uses it unless §5 overrides **screen** `patience_steps` for Band B/C.

| Condition | `max_steps` | `patience_steps` |
|---|---:|---:|
| \(n_{\mathrm{train}} \le 3000\) | **6400** | **800** |
| \(n_{\mathrm{train}} > 3000\) | **187500** | **23437** |

**Physics \(\lambda\)-screen patience overrides:**

| Band | \(n_{\mathrm{train}}\) | Screen `patience_steps` |
|---|---|---:|
| A | \(< 6000\) | (no override; use table above) |
| B | \(6000 \le n < 45000\) | **4700** |
| C | \(\ge 45000\) | **3000** |

(\(n=45000\) uses Band C. `max_steps` during screens still follows the table unless overridden in meta.)

**Budget table** (\(B=64\)):

| \(N\) | \(K\) | `max_steps` | plain `patience` | phys screen `patience` | §5 band |
|------:|------:|------------:|-----------------:|-----------------------:|---|
| 300 | 5 | 6400 | 800 | 800 | A |
| 866 | 14 | 6400 | 800 | 800 | A |
| 2498 | 40 | 6400 | 800 | 800 | A |
| 7207 | 113 | 187500 | 23437 | **4700** | B |
| 20794 | 325 | 187500 | 23437 | **4700** | B |
| 60000 | 938 | 187500 | 23437 | **3000** | C |

---

## 3. Logging during training (plain and physics)

Every epoch, record:

1. **Pulse L1 (train & val):** raw and best-ambiguity  
2. **TRACE L1 (train & val)**  
3. **Gradient norms:** \(\|\nabla L_{\mathrm{data}}\|\), \(\|\nabla L_{\mathrm{reg}}\|\), \(\|\nabla L_{\mathrm{total}}\|\)  
4. **Per-batch timings**  
5. **Wall-clock**  

### Reconstruction evolution snapshots

Every **2 epochs** (and at best/stop if needed): fixed val sample; fixed noisy TRACE @ **SNR = 10 dB**; save `I_noisy`, `E_true`, `E_pred` (+ epoch/step).

### \(\lambda\) screens

For **every** \(\lambda\) candidate:

- Same histories + per-run sanity plots as above  
- **Bands B/C:** also save a **full train-state** at stop (§5.0) for exact resume  

Required before declaring \(\lambda^\*\): coarse overlay + all-screened overlay.

---

## 4. Plain network (\(\lambda = 0\))

**No LR tuning.** Budgets from §2.1 only (not §5 overrides).

- Train once with full §3 logging + snapshots  
- Save checkpoint, history, meta  

---

## 5. Physics network (\(\lambda > 0\)) — by \(n_{\mathrm{train}}\) band

\[
L = L_{\mathrm{data}} + \lambda\, L_{\mathrm{reg}},\quad \mathrm{trace\_scale}=8,\quad \mathrm{LR}=10^{-3}
\]

| Band | Condition |
|---|---|
| **A** | \(n_{\mathrm{train}} < 6000\) |
| **B** | \(6000 \le n_{\mathrm{train}} < 45000\) |
| **C** | \(n_{\mathrm{train}} \ge 45000\) |

### 5.0 Full train-state (required for B/C)

Save/load enough to resume **exactly** from the screen stop point:

- `model.state_dict()`, `optimizer.state_dict()` (Adam moments)  
- RNG: `torch` (CPU/CUDA), `numpy`, Python `random`  
- `global_step`, epoch / stopped epoch  
- `best_score`, `best_step`, `best_epoch`, best model weights  
- early-stop bookkeeping + history so far  
- hyperparams / seeds  

Optional: DataLoader / noise generator seeds for bit-exact order.

Disk (Multires): weights ~**40 MB**; Adam ~**80 MB**; full resume ~**120 MB** per \(\lambda\).

### 5.A Band A — \(n < 6000\) (legacy; unchanged)

- Budgets: §2.1 (same as plain)  
- Coarse: \(\lambda \in \{0.75,\ 1.5,\ 3,\ 6,\ 12\}\) (+ optional \(15\))  
- Fine: geometric midpoints vs coarse neighbors; optionally further bisection  
- Official model = **best checkpoint of winning screen** (no scratch retrain, **no** post-selection extension)

### 5.B Band B — \(6000 \le n < 45000\) (e.g. 7207, 20794)

Screen patience: **4700** steps.

1. **Stage 1:** screen only \(\lambda \in \{0.6,\ 1.8,\ 4.3\}\) (full train-state at each stop).  
2. **Stage 2:** if winner is an **edge** (\(0.6\) or \(4.3\)) → **no** extra \(\lambda\).  
   If winner is **1.8** → one extra  
   \(\lambda=\sqrt{1.8\cdot\lambda_{\mathrm{side}}}\) where \(\lambda_{\mathrm{side}}\) is the **worse** of \(\{0.6,\ 4.3\}\) (higher Stage-1 val score).  
3. **Stage 3:** \(\lambda^\*\) = best among Stage-1 three, or among Stage-1+one fine if Stage-2 ran.  
4. **Stage 4:** exact-resume the winning run from its stop state; remember screening `best_*`; train **+2000** more steps; official checkpoint = **best(screen + extension)** (record `best_step` / `best_epoch`).

### 5.C Band C — \(n \ge 45000\) (e.g. 60000)

Screen patience: **3000** steps.

1. **Stage 1:** \(\lambda \in \{0.6,\ 1.8,\ 3.5\}\)  
2. **Stage 2:** edge winner → no fine; if **1.8** wins →  
   \(\lambda=\sqrt{1.8\cdot\lambda_{\mathrm{side}}}\) with \(\lambda_{\mathrm{side}}=\) **worse** of \(\{0.6,\ 3.5\}\) (higher Stage-1 val score)  
3. **Stage 3:** same declaration rule as Band B  
4. **Stage 4:** exact-resume winner; **+1000** more steps; official = **best(screen + extension)**

---

## 6. Evaluation (official plain + physics)

Same held-out test set:

1. **SNR sweep** (e.g. \([-10,\ 30]\) dB step 5)  
2. **Test example @ SNR = 10 dB**  
3. **Summary row** including screen patience / extension steps when used  

---

## 7. Per-\(n\) checklist

- [ ] Shared val (200) with SNR \(\in\{-10,0,30\}\); shared test; train size \(n\)  
- [ ] Plain budget band + physics \(\lambda\) band (A / B / C)  
- [ ] Plain: \(\mathrm{LR}=10^{-3}\); one train + §3  
- [ ] Physics A: legacy coarse/fine; winner = official  
- [ ] Physics B: pat=4700; \(\{0.6,1.8,4.3\}\); optional one geo-mean if 1.8; resume **+2000**  
- [ ] Physics C: pat=3000; \(\{0.6,1.8,3.5\}\); optional one geo-mean if 1.8; resume **+1000**  
- [ ] Screens: curves, overlays; B/C full train-state  
- [ ] Test SNR sweep + example @ 10 dB  
- [ ] Archive meta / timings  

---

## 8. Fairness notes

- Same \(B=64\), val/test, val SNR \(\{-10,0,30\}\), architecture  
- Physics always `trace_scale=8`; LR fixed at \(10^{-3}\)  
- Band A: official = screen winner  
- Bands B/C: official = best(screen + exact-resume extension); no scratch retrain  
- Plain patience = §2.1; physics screen patience = §5 in B/C  
- Further budget overrides only with explicit decision + meta  

---

## 9. Diff vs Protocol v1

| Topic | v1 | v2 |
|---|---|---|
| \(B\) | 300 | **64** |
| Plain LR | grid + edge expand | **fixed \(10^{-3}\)** |
| Budgets | \(100\cdot K\) / \(200\cdot K\) | **bands by \(n\le3000\) vs \(>3000\)** |
| Val SNR | \(U[0,30]\) | **\(\{-10,0,30\}\)** |
| Physics \(n<6000\) | retrain final | **screen winner (legacy grid/fine)** |
| Physics \(n\ge6000\) | (v1 final retrain) | **compact 3-point \(\lambda\) + optional one geo-mean; exact-resume short extension** |
| Logging / sweep / test@10 | required | **same** |

---

## 10. Suggested artifact layout

```text
data efficiency experiment/
  PROTOCOL.txt / PROTOCOL.md      # v1
  PROTOCOL_v2.txt / PROTOCOL_v2.md
  checkpoints/v2/
    n{n}_plain_lr1e-3/...
    n{n}_phys_lam{lam}/...        # include *_train_state.pt for B/C
```

Tags should include `n`, `lam`, `lr`, `protocol=v2`, band (`A`/`B`/`C`), and seed.
