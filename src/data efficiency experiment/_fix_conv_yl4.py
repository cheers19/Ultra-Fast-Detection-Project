import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import math

NB = Path(
    r"d:\Haim\Haim 3rd paper\Ultra-Fast project\src\data efficiency experiment"
    r"\v2_plain_vs_physics_data_efficiency_NB.ipynb"
)
nb = json.loads(NB.read_text(encoding="utf-8"))
cell = next(c for c in nb["cells"] if c.get("id") == "a094f0e8")
src = "".join(cell["source"])

old = '''plt.tight_layout()
fig.canvas.draw()
_in_bbox = ax_in.get_window_extent()
_h_pts = float(_in_bbox.height) * 72.0 / float(fig.dpi)
_yl_fs = min(float(leg_fs), max(8.0, 0.88 * _h_pts / (len("Convergence") * 0.62)))
_yl.set_fontsize(_yl_fs)'''

new = '''plt.tight_layout()
# Shrink inset ylabel until its drawn height fits inside the inset frame.
_renderer = fig.canvas.get_renderer()
fig.canvas.draw()
_ib = ax_in.get_window_extent(renderer=_renderer)
_yl_fs = float(leg_fs)
for _fs in np.linspace(leg_fs, 8.0, 17):
    _yl.set_fontsize(float(_fs))
    fig.canvas.draw()
    _tb = _yl.get_window_extent(renderer=fig.canvas.get_renderer())
    if float(_tb.height) <= 0.92 * float(_ib.height):
        _yl_fs = float(_fs)
        break
_yl.set_fontsize(_yl_fs)'''

if old not in src:
    raise SystemExit("fit block not found")

src2 = src.replace(old, new)
compile(src2, "c", "exec")
text = src2.replace("\r\n", "\n").rstrip("\n") + "\n"
cell["source"] = [ln + "\n" for ln in text.split("\n")[:-1]]
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

OUT_FIG = Path(
    r"d:\Haim\Haim 3rd paper\Ultra-Fast project\src\data efficiency experiment"
    r"\figures\v2_comparisons"
)
DIAG = Path(
    r"d:\Haim\Haim 3rd paper\Ultra-Fast project\src\checkpoints\benchmark"
    r"\filtered_c1_multires_2k_diagnostics"
)
CONV_CACHE = OUT_FIG / "filtered_c1_conv_sim_leq0p05.npz"
LEGEND_FS = 16
BATCH_SIZE = 64

zconv = np.load(CONV_CACHE, allow_pickle=True)
snr_emp = np.asarray(zconv["snr_sweep_db"], dtype=float)
frac_phys_emp = np.asarray(zconv["phys_2k"], dtype=float)
frac_60k_emp = np.asarray(zconv["plain_60k"], dtype=float)
meta_phys_2k = json.loads(
    (DIAG / "filtered_c1_multires_lam4p5_meta.json").read_text(encoding="utf-8")
)
meta_60k = json.loads(
    (DIAG / "filtered_c1_multires_60k_lam0_meta.json").read_text(encoding="utf-8")
)


def _best_step_from_meta(meta, *, batch_size=BATCH_SIZE):
    if meta.get("best_step") is not None:
        return int(meta["best_step"])
    n_tr = int(meta["n_train"])
    spe = int(math.ceil(float(n_tr) / float(batch_size)))
    return int(meta["best_epoch"]) * spe


best_step_phys = _best_step_from_meta(meta_phys_2k)
best_step_60k = _best_step_from_meta(meta_60k)

plot_src = "".join(cell["source"])
block = plot_src[
    plot_src.index("y_phys_pct = 100.0") : plot_src.index('print("wrote", fig_emp)')
    + len('print("wrote", fig_emp)')
].replace("plt.show()", "plt.close()")
# debug print
block = block.replace(
    "_yl.set_fontsize(_yl_fs)",
    "_yl.set_fontsize(_yl_fs)\nprint('fitted ylabel fs=', _yl_fs)",
)
exec(
    block,
    {
        "plt": plt,
        "np": np,
        "LEGEND_FS": LEGEND_FS,
        "OUT_FIG": OUT_FIG,
        "snr_emp": snr_emp,
        "frac_phys_emp": frac_phys_emp,
        "frac_60k_emp": frac_60k_emp,
        "best_step_phys": best_step_phys,
        "best_step_60k": best_step_60k,
    },
)
