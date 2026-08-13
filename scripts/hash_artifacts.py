#!/usr/bin/env python
import argparse, hashlib
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("path",nargs="?",default="artifacts"); p.add_argument("--output",default="SHA256SUMS"); a=p.parse_args(); root=Path(a.path); rows=[]
for f in sorted(root.rglob("*")) if root.exists() else []:
 if f.is_file(): rows.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.as_posix()}")
Path(a.output).write_text("\n".join(rows)+("\n" if rows else ""),encoding="utf-8"); print(f"Hashed {len(rows)} files")
