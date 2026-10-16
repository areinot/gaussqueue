"""Carve a trained splat file once, with no training: splats centred in space the LiDAR saw as
empty are removed, and long ones poking into it are shortened (train.py's --carve, applied one time).

  python carve_ply.py <in.ply> <space.npz> <out.ply> [--band 0.03]

Nothing repairs what the cut takes (a glass rim LiDAR saw through stays cut); for that, carve
during a short fine-tune instead (train.py --init-ply ... --strategy none --carve).
"""
import argparse

import torch

import gsdata
import train as T


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ply")
    p.add_argument("space")
    p.add_argument("out")
    p.add_argument("--band", type=float, default=0.0, help="only within this many metres of a LiDAR surface")
    a = p.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    params = {k: v.to(dev) for k, v in gsdata.load_ply(a.ply).items()}
    space = T.Space(a.space, dev)
    removed, shortened = T.carve(params, space, band=a.band)
    keep = torch.sigmoid(params["opacities"]) > 0.005
    gsdata.save_ply(a.out, {k: v[keep] for k, v in params.items()})
    print(f"{removed:,} splats removed, {shortened:,} shortened; {int(keep.sum()):,} kept -> {a.out}")


if __name__ == "__main__":
    main()
