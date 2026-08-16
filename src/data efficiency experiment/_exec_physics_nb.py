"""Execute physics_multires_n2498_NB.ipynb with embedded figures."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.pop("MPLBACKEND", None)

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent
path = HERE / "physics_multires_n2498_NB.ipynb"
print("Executing", path, flush=True)
nb = nbformat.read(path, as_version=4)
ep = ExecutePreprocessor(timeout=None, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": str(HERE)}})
nbformat.write(nb, path)
print("NOTEBOOK_OK", flush=True)
