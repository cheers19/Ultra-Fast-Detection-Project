"""Build and run plain n2498 driver from the notebook."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
nb = json.loads((HERE / "plain_multires_n2498_NB.ipynb").read_text(encoding="utf-8"))
parts = [
    "import os",
    'os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")',
    'os.environ.setdefault("MPLBACKEND", "Agg")',
    "import matplotlib",
    'matplotlib.use("Agg")',
    "import matplotlib.pyplot as plt",
    "plt.show = lambda *a, **k: None",
    'print("DRIVER_START", flush=True)',
]
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    parts.append("\n# --- cell ---\n")
    parts.append("".join(c["source"]))
driver = HERE / "_plain_n2498_driver.py"
driver.write_text("\n".join(parts) + "\n", encoding="utf-8")
print("wrote", driver)
