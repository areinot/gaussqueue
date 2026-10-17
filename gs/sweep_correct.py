"""Sweep correction for a whole capture, by search: move each sweep onto the rest, one at a time.

  python sweep_correct.py <data> --out <dir> [--train-args "..."] [--search-args "..."]

<data> is a dataset train.py reads (images/, sparse/0, and optionally masks/, sensor_data/,
box.json). The photos are split into sweeps (train.find_sweeps: capture order, broken at jumps,
sharp turns and every 15 photos); the largest sweep is the anchor and never moves. Then, for
every other sweep, largest first:

  1. train a model on all the other photos (their cameras as corrected so far), this sweep held out
  2. search for the rigid move of this sweep's cameras that best matches that model (sweep_search.py)
  3. keep the moved cameras for the next round

Each sweep is matched against everything already corrected, so the capture is pulled together
step by step. Writes <out>/corrected/sparse/0 (the model with every sweep moved) and
<out>/sweeps.json (each sweep's photos, move and scores). Work files go to <out>/work.
"""
import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import gsdata
import sweep_search
import train as T


def link_dataset(src, dst, model):
    """A dataset folder sharing <src>'s photos, masks and depth, with its own sparse/0."""
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("images", "masks", "sensor_data", "patch_density"):
        s, d = src / name, dst / name
        if s.is_dir() and not d.exists():
            if sys.platform == "win32":
                subprocess.run(["cmd", "/c", "mklink", "/J", str(d), str(s)], check=True, capture_output=True)
            else:
                d.symlink_to(s.resolve())
    for name in ("box.json",):
        if (src / name).exists():
            shutil.copyfile(src / name, dst / name)
    m = dst / "sparse" / "0"
    if m.exists():
        shutil.rmtree(m)
    shutil.copytree(model, m)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data")
    p.add_argument("--out", required=True)
    p.add_argument("--train-args", default="--steps 10000 --lr-decay-steps 5000 --refine-stop 4000 "
                                           "--max-res 1600 --strategy mcmc --max-splats 400000",
                   help="train.py options for each held-out model, one quoted string")
    p.add_argument("--search-args", default="--shift-range 0.024 --turn-range 2.0",
                   help="sweep_search.py options, one quoted string")
    p.add_argument("--sweep-jump", type=float, default=0.0, help="see train.find_sweeps (0 = automatic)")
    p.add_argument("--sweep-max", type=int, default=0, help="split sweeps longer than this (0 = never)")
    p.add_argument("--sweep-min", type=int, default=3, help="fold sweeps shorter than this into a neighbour")
    a = p.parse_args(argv)
    data, out = Path(a.data), Path(a.out)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    model = gsdata.find_model(data)
    images, _, _ = gsdata.read_model(model)
    names = [im["name"] for im in images]
    vm = torch.tensor(np.stack([im["viewmat"] for im in images]), dtype=torch.float32)
    sid = T.find_sweeps(names, torch.linalg.inv(vm), a.sweep_jump, longest=a.sweep_max, min_size=a.sweep_min)
    groups = {}
    for n, s in zip(names, sid):
        groups.setdefault(s, []).append(n)
    anchor = max(groups, key=lambda g: len(groups[g]))
    order = sorted((g for g in groups if g != anchor), key=lambda g: -len(groups[g]))
    print(f"{len(names)} photos in {len(groups)} sweeps; anchor = sweep {anchor} ({len(groups[anchor])} photos)",
          flush=True)

    report = {"anchor": anchor, "sweeps": {str(g): {"photos": sorted(groups[g])} for g in groups}}
    cur = model
    for k, g in enumerate(order):
        tag = f"{k:02d}_sweep{g}"
        ds = work / f"data_{tag}"
        link_dataset(data, ds, cur)
        hold = work / f"{tag}.txt"
        hold.write_text("\n".join(groups[g]) + "\n")
        print(f"=== sweep {g} ({len(groups[g])} photos, {sorted(groups[g])[0]} - {sorted(groups[g])[-1]}): "
              f"training without it", flush=True)
        tdir = work / f"train_{tag}"
        targs = [str(ds), "--holdout-names", str(hold), "--out", str(tdir)] + shlex.split(a.train_args)
        if (ds / "sensor_data").exists():
            targs += ["--depth-dir", str(ds / "sensor_data")]
        if (ds / "box.json").exists():
            targs += ["--box", str(ds / "box.json")]
        r = subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("train.py"))] + targs,
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
            raise SystemExit(f"training without sweep {g} failed")
        ply = sorted(tdir.glob("export_*.ply"), key=lambda f: int(f.stem.split("_")[1]))[-1]
        new = work / f"model_{tag}"
        sweep_search.main([str(ply), str(ds), "--names", str(hold), "--out", str(work / f"search_{tag}.json"),
                           "--write-model", str(new)] + shlex.split(a.search_args))
        report["sweeps"][str(g)]["search"] = json.loads((work / f"search_{tag}.json").read_text())
        cur = new
        (out / "sweeps.json").write_text(json.dumps(report, indent=1))

    final = out / "corrected" / "sparse" / "0"
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(cur, final)
    report["seconds"] = round(time.time() - t0, 1)
    (out / "sweeps.json").write_text(json.dumps(report, indent=1))
    print(f"all sweeps done in {report['seconds']:.0f} s -> {final}", flush=True)


if __name__ == "__main__":
    main()
