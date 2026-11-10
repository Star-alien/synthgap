"""Copy pools downloaded from Kaggle into data/pools and rewrite absolute /kaggle paths."""
import json, shutil, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path("data/pools")
for m in src.rglob("manifest.json"):
    rel = m.parent.relative_to(src)
    out = dst / rel
    if out.exists() and (out / "manifest.json").exists():
        continue
    shutil.copytree(m.parent, out, dirs_exist_ok=True)
    d = json.loads((out / "manifest.json").read_text())
    root = d["root"]
    d["root"] = str(out.resolve())
    d["paths"] = [str(out.resolve() / Path(p).relative_to(root)) for p in d["paths"]]
    (out / "manifest.json").write_text(json.dumps(d))
    print("localized", rel, len(d["paths"]))
