#!/usr/bin/env python
import argparse, json, html
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--metrics",default="artifacts/research/metrics.json"); p.add_argument("--output",default="docs/research-findings.html"); a=p.parse_args(); src=Path(a.metrics)
if not src.exists(): raise SystemExit("Metrics file not found. Run evaluation first; findings are never fabricated.")
m=json.loads(src.read_text(encoding="utf-8")); rows="".join(f"<tr><td>+{html.escape(h)}h</td><td>{v['mae']:.3f}</td><td>{v['rmse']:.3f}</td><td>{v['mape']:.2f}%</td><td>{v['r2']:.3f}</td></tr>" for h,v in m['horizons'].items()); doc=f'''<!doctype html><meta charset="utf-8"><title>Research Findings — İstanbul GraphTraffic AI</title><link rel="stylesheet" href="assets/site.css"><main class="wrap"><a href="index.html">← Project</a><h1>Research Findings</h1><p class="notice">Generated from <code>{html.escape(a.metrics)}</code>. Dataset provenance must be checked before interpretation.</p><table><thead><tr><th>Horizon</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>R²</th></tr></thead><tbody>{rows}</tbody></table></main>'''; Path(a.output).write_text(doc,encoding="utf-8"); print(a.output)
