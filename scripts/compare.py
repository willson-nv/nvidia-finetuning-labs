#!/usr/bin/env python3
"""Fills in the Demo B table from the two run_stats.json files."""
import json, pathlib, sys
rows = []
for d in sys.argv[1:] or ["../checkpoints/demo-a", "../checkpoints/demo-b"]:
    f = pathlib.Path(d) / "run_stats.json"
    if f.exists():
        rows.append(json.loads(f.read_text()))
    else:
        print(f"(missing {f})")
if rows:
    print(f"\n{'':22s}{'peak GPU':>12s}{'minutes':>10s}")
    for r in rows:
        print(f"{r['mode']:22s}{r['peak_gpu_gb']:>10.1f} GB{r['minutes']:>10.1f}")
    if len(rows) == 2 and rows[0]['peak_gpu_gb']:
        saved = 100 * (1 - rows[1]['peak_gpu_gb'] / rows[0]['peak_gpu_gb'])
        print(f"\n  QLoRA used {saved:.0f}% less memory, and took "
              f"{rows[1]['minutes'] - rows[0]['minutes']:+.1f} min longer.\n")
