"""Render a trained .ply at a dataset's cameras and score each view against its photo.

  python render.py <ply> <data> --out <dir> [--names file] [--save name ...] [--sh-degree 3]

Scores every photo in <data> (or those listed in --names): PSNR/SSIM, over the box mask for
box datasets. Writes <out>/views.json; --save also writes those renders as PNGs ("all" for every one).
"""
import argparse
import json
from pathlib import Path

import torch

import gsdata
import train as T


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ply")
    p.add_argument("data")
    p.add_argument("--out", required=True)
    p.add_argument("--names")
    p.add_argument("--save", nargs="*", default=[])
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--max-res", type=int, default=1920)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    images, _, _ = gsdata.read_model(gsdata.find_model(a.data))
    if a.names:
        want = {l.strip() for l in Path(a.names).read_text().splitlines() if l.strip()}
        images = [im for im in images if im["name"] in want]
    params = T.make_params(gsdata.load_ply(a.ply), "cuda")
    deg = min(a.sh_degree, gsdata.sh_degree_of(params))
    res = []
    for k in range(0, len(images), 64):   # a batch of photos at a time keeps memory small
        v = gsdata.Views(a.data, images[k:k + 64], a.max_res, "cuda")
        with torch.no_grad():
            for i in range(len(v)):
                rgb, *_ = T.render(params, v, i, deg)
                rgb, gt = rgb.clamp(0, 1), v.image(i)
                m = v.masks[i] == 255 if v.masks[i] is not None else None
                res.append({"name": v.names[i], "psnr": gsdata.psnr(rgb, gt, m),
                            "ssim": float(gsdata.ssim(rgb, gt, m))})
                if "all" in a.save or Path(v.names[i]).stem in a.save or v.names[i] in a.save:
                    gsdata.save_png(out / (Path(v.names[i]).stem + ".png"), rgb)
        del v
        torch.cuda.empty_cache()
    (out / "views.json").write_text(json.dumps(res, indent=1))
    print(f"{len(res)} views, mean PSNR {sum(r['psnr'] for r in res) / max(1, len(res)):.2f}")


if __name__ == "__main__":
    main()
