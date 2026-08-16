# Data-Efficiency Experiment — Training & Evaluation Protocol

> **Readable copies (outside the IDE):** [`PROTOCOL.docx`](PROTOCOL.docx) (Word) and [`PROTOCOL.txt`](PROTOCOL.txt) (plain text).  
> This Markdown file is a mirror for git/diff; regenerate the `.docx` with `_write_protocol_docx.py` if you edit the protocol.

This document is the single source of truth for training and evaluating **plain** (\(\lambda=0\)) and **physics** (\(\lambda>0\)) Multires models across training-set sizes. Follow it as written; do not re-negotiate steps ad hoc.

Code in this folder may import existing modules from the parent `src` directory (e.g. `data_generation`, `dataset_utils`, `data_c_amb_loss_diagnostics`, `train`, `frognet`). From notebooks or scripts here, run:

```python
import setup_src_path  # adds parent src/ to sys.path
```

or keep the working directory as `src` and use relative imports/paths accordingly. Prefer writing artifacts under this folder (e.g. `checkpoints/`, `runs/`) using paths anchored to `Path(__file__).resolve().parent` or the notebook’s directory.

---

## 1. Experiment scope

For **each** training size \(n_{\mathrm{train}}\) below, train and evaluate:

1. **Plain network:** \(\lambda = 0\)
2. **Physics network:** \(\lambda > 0\) (TRACE loss term), with **`trace_scale = 8` always**

### Training-set sizes

\[
n_{\mathrm{train}} \in \{300,\ 866,\ 2498,\ 7207,\ 20794,\ 60000\}
\]

Process sizes from small to large. For each \(n\): plain (LR tune → final) → physics (\(\lambda\) tune → final) → shared test.

### Data family (all models)

Every plain and physics model — for every \(n_{\mathrm{train}}\) — is trained and evaluated on the **same** pulse/TRACE family: **spectrally filtered C1**, matching `filtered_c1_multires_2k_diagnostics_NB.ipynb` (`FilteredC1PulseConfig` / `generate_pulses_filtered_c1` + physical `FROGNet`). Do **not** mix Data C / unfiltered C1 / other generators into this campaign.

---

## 2. Shared fixed settings (all models, all \(n\))

| Item | Value |
|---|---|
| Architecture | Multires |
| Pulse / TRACE data | Filtered C1 **only** (same as `filtered_c1_multires_2k_diagnostics_NB.ipynb`); train, val, and test for all models |
| Batch size \(B\) | **300** |
| `drop_last` | `False` |
| Optimizer | Adam |
| \(n_{\mathrm{val}}\) | **200** — **identical val set for every model and every \(n\)** |
| \(n_{\mathrm{test}}\) | Fixed (e.g. 512) — **identical test set for every model and every \(n\)** |
| Train / val SNR | Uniform in \([0,\ 30]\) dB (resampled each use, as in the diagnostics notebook) |
| Selection metric | Validation pulse L1 after **best ambiguity** |
| \(K\) (steps/epoch) | \(K = \lceil n_{\mathrm{train}} / B \rceil\) |
| Early stopping | Validate every epoch (curves vs epoch). Stop when \((t - t_{\mathrm{best}}) \ge patience\_steps\) with \(patience\_steps = 25\cdot K\) (= 25 epochs). Same rule for screens and finals. |
| Step budget | Anchor in epochs, convert via \(K\) (preserves data-passes vs Multires 2K diagnostics at \(B=64\), patience 25 epochs): **screens** \(max\_steps = 100\cdot K\); **finals** \(max\_steps = 200\cdot K\). May set \(max\_epochs = \lceil max\_steps / K \rceil\). |
| Physics TRACE scale | **`trace_scale = 8`** for all physics runs (do not use a data-dependent scale) |
| Canonicalization | Same as diagnostics notebook (`t0`) |
| Seeds | Train pulses may depend on \(n\); **val and test pulses/seeds fixed once and reused for all models** |

**Budget table** (\(B=300\)), all campaign \(n_{\mathrm{train}}\):

| \(N\) | \(K\) | patience (steps) | patience (epochs) | max screen (steps) | screen (epochs) | max final (steps) | final (epochs) |
|------:|------:|-----------------:|------------------:|-----------------:|----------------:|------------------:|---------------:|
| 300 | 1 | 25 | 25 | 100 | 100 | 200 | 200 |
| 866 | 3 | 75 | 25 | 300 | 100 | 600 | 200 |
| 2498 | 9 | 225 | 25 | 900 | 100 | 1800 | 200 |
| 7207 | 25 | 625 | 25 | 2500 | 100 | 5000 | 200 |
| 20794 | 70 | 1750 | 25 | 7000 | 100 | 14000 | 200 |
| 60000 | 200 | 5000 | 25 | 20000 | 100 | 40000 | 200 |

---

## 3. Logging during training (plain and physics)

Every epoch, record (same spirit as Multires 2K plain/physics in `filtered_c1_multires_2k_diagnostics_NB.ipynb`):

1. **Pulse L1 (train & val):** raw and best-ambiguity  
2. **TRACE L1 (train & val)**  
3. **Gradient norms:** \(\|\nabla L_{\mathrm{data}}\|\), \(\|\nabla L_{\mathrm{reg}}\|\), \(\|\nabla L_{\mathrm{total}}\|\)  
   - For \(\lambda=0\): total loss is \(L_{\mathrm{data}}\) only; still compute \(\|\nabla L_{\mathrm{reg}}\|\) for diagnostics and label it as not entering the loss  
4. **Per-batch timings (ms/batch mean ± over epochs):**  
   `data_prep_sec`, `loss_data_fwd_sec`, `loss_reg_fwd_sec`, `total_backward_sec`, `optimizer_step_sec`  
5. **Wall-clock:** dataset generation time; total train time; time to best checkpoint  

### Reconstruction evolution snapshots

Every **2 epochs** (and also at best / stop if those epochs are odd):

- Fix **one validation example** for the whole experiment (same sample for all models).  
- Apply a **fixed** noisy TRACE at **SNR = 10 dB** (same noise realization across epochs).  
- Save `I_noisy`, `E_true`, `E_pred` (+ epoch / step).  

Do **not** store full-set predictions each epoch.

### Hyperparameter-tuning runs (LR / λ screens)

For every screening run, save the same per-epoch train/val histories as finals and **plot** at least:

- train & val pulse L1 (raw + best-amb)  
- train & val TRACE L1  

These plots are required for sanity checks before declaring LR\* / \(\lambda^\*\).  
Optional: overlay val best-amb curves for all LRs (or \(\lambda\)s) on one figure.

---

## 4. Plain network (\(\lambda = 0\)) — LR tuning and final train

**Goal:** choose learning rate, then produce the official plain checkpoint for this \(n\).

### 4.1 LR grid (initial)

Scan exactly:

\[
\mathrm{LR} \in \{10^{-4},\ 10^{-3},\ 10^{-2}\}
\]

### 4.2 If LR\* is at a grid edge — expand

Do **not** accept an edge winner without at least one expansion toward that edge.

- Lower edge (\(\mathrm{LR}^*=10^{-4}\)): add \(\{3\cdot10^{-5},\ 10^{-5}\}\), re-screen, update LR\*.
- Upper edge (\(\mathrm{LR}^*=10^{-2}\)): add \(\{3\cdot10^{-2},\ 10^{-1}\}\), re-screen, update LR\*.

Repeat at most 1–2 expansions until LR\* is interior (both neighbors worse) or the expanded edge remains best after the extra points.

Optional fine point: geometric midpoint between LR\* and its better neighbor (e.g. between \(10^{-4}\) and \(10^{-3}\) try \(\sim 3\cdot10^{-4}\)).

### 4.3 Procedure

1. For each LR in the current grid, train with §2 settings and §3 logging; early stop = fixed step patience (§2).  
2. Select **LR\*** by best validation pulse L1 (best-amb). If at an edge, apply §4.2 before declaring LR\*.  
3. **Final plain training** with LR\*, same early-stop rule, `max_steps`, full logging and snapshots.  
4. Save: best checkpoint, history (curves + grads + timings), snapshots, meta (`n`, LR\*, \(\lambda=0\), seeds, best/stop step).

There is **no** TRACE weight in the optimized loss (\(\lambda=0\)).

---

## 5. Physics network (\(\lambda > 0\)) — \(\lambda\) tuning and final train

**Goal:** choose \(\lambda\) with the plain network’s LR\*; produce the official physics checkpoint for this \(n\).

### 5.1 Fixed for all physics runs

- `trace_scale = 8`  
- **LR = LR\*** from the plain model at the **same** \(n\)  
- **Do not** retune LR after finding \(\lambda^\*\)

Loss form (conceptually):

\[
L = L_{\mathrm{data}} + \lambda\, L_{\mathrm{reg}},\quad L_{\mathrm{reg}}\ \text{uses TRACE scale } 8.
\]

### 5.2 Coarse \(\lambda\) search (short training)

Log-spaced candidates over \([0.75,\ 15]\), e.g.:

\[
\lambda \in \{0.75,\ 1.5,\ 3,\ 6,\ 12\}
\]

(optional extra point \(15\)).

**Prior for small data:** if \(n_{\mathrm{train}} \lesssim 2000\) (here: \(300\), \(866\); treat \(2498\) conservatively the same if desired), do **not** favor \(\lambda < 0.75\); start the coarse grid at **\(\lambda \ge 3\)** (e.g. \(\{3,\ 6,\ 12,\ 15\}\)).

Use the same fixed step-patience early-stop rule as finals (§2); optional explicit shorter screen rule only if labeled. Rank by validation pulse L1 (best-amb).

### 5.3 Fine \(\lambda\) search

Around the coarse winner \(\lambda^\*\), local **geometric bisection** (log-\(\lambda\)):

- Evaluate geometric midpoints between \(\lambda^\*\) and its coarse neighbors (e.g. if \(6\) wins between \(3\) and \(12\), try \(\sqrt{3\cdot6}\) and \(\sqrt{6\cdot12}\)).  
- Optionally one more bisection (2–4 fine points total).  
- Keep screening short; pick the best \(\lambda\) on val.

### 5.4 Final physics training

One full run with \((\lambda^\*,\ \mathrm{LR}^*)\), `trace_scale=8`, early-stop fixed step patience (§2), `max_steps`, full logging and snapshots. Save checkpoint + history + meta.

---

## 6. Evaluation (after each final checkpoint)

Use the **same held-out test set** for every model:

1. **SNR sweep** (same grid as the diagnostics notebook, e.g. \([-10,\ 30]\) dB step 5): report best-amb pulse L1 and similarity (and PCGPA if included in the campaign).  
2. **Single test example @ SNR = 10 dB:** clean TRACE, noisy TRACE, \(|E(t)|\), phase after best-amb — for plain and physics.  
3. **Summary row per model:** \(n_{\mathrm{train}}\), plain/physics, LR\*, \(\lambda\) (0 or \(\lambda^\*\)), `trace_scale` (8 if physics), best/stop step, key val/test metrics, wall-clock and mean batch timings.

---

## 7. Per-\(n\) checklist

For a given \(n_{\mathrm{train}}\):

- [ ] Build / load shared val (200) and test; build train of size \(n\)  
- [ ] Plain: LR grid \(\{10^{-4},\ 10^{-3},\ 10^{-2}\}\); expand if LR\* at edge (§4.2); then final plain train + logs + snapshots
- [ ] HP screens: plot train/val pulse L1 (raw+amb) and TRACE L1 before declaring LR\* / \(\lambda^\*\) (§3)
- [ ] Physics: LR\* fixed, `trace_scale=8`; coarse \(\lambda\) → fine \(\lambda\) → final physics train + logs + snapshots
- [ ] Test SNR sweep + example @ 10 dB on both finals  
- [ ] Archive meta and timings for cross-\(n\) comparison  

---

## 8. Fairness notes (do not violate)

- Same \(B\), val set, test set, early-stop rule (fixed step patience), `max_steps` ceiling, SNR policy, and architecture across models.  
- Different \(K\) (steps/epoch) across \(n\) is expected; do **not** force equal step counts to convergence.  
- Tuning compute (LR/\(\lambda\) screens) should be logged separately from final-train wall time when comparing efficiency.  
- Physics always uses **`trace_scale = 8`**.  
- No LR retuning for physics after \(\lambda^\*\) is chosen.  
- Do **not** declare LR\* at a grid edge without the expansion in §4.2.

---

## 9. Suggested artifact layout (under this folder)

```text
data efficiency experiment/
  PROTOCOL.md                 # this file
  setup_src_path.py           # import helper for parent src/
  checkpoints/
    n{n}_plain_lr{lr}/...
    n{n}_phys_lam{lam}_lr{lr}/...
  sweeps/                     # optional LR / lambda screen summaries
  figures/                    # optional plots
```

Use consistent tags including `n`, `lam`, `lr`, and seed in filenames.
