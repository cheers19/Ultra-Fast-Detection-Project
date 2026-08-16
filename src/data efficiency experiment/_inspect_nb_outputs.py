import json
from pathlib import Path

nb = json.loads(Path("plain_multires_n2498_NB.ipynb").read_text(encoding="utf-8"))
print("nbformat", nb.get("nbformat"), "cells", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    outs = c.get("outputs") or []
    kinds = {}
    for o in outs:
        ot = o.get("output_type")
        kinds[ot] = kinds.get(ot, 0) + 1
        if ot == "display_data":
            data = o.get("data") or {}
            if "image/png" in data:
                png = data["image/png"]
                n = len(png) if isinstance(png, str) else -1
                print(f"  cell {i}: PNG chars={n}")
        elif ot == "error":
            print(f"  cell {i}: ERROR", o.get("ename"), o.get("evalue"))
    src = "".join(c.get("source") or [])[:55].replace("\n", " ")
    print(f"cell {i}: exec={c.get('execution_count')} n_out={len(outs)} kinds={kinds}")
    print(f"   src: {src}")
    for o in outs:
        if o.get("output_type") == "stream":
            t = "".join(o.get("text") or [])[:150].replace("\n", " | ")
            print(f"   stream: {t}")
            break
