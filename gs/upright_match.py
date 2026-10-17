"""Feature matching on gravity-uprighted photos: ALIKED + LightGlue into a COLMAP database.

  python upright_match.py --colmap <colmap> --images <dir> --model <sparse/0> --work <dir>
                          --models <dir with aliked-n16rot.onnx, aliked-lightglue.onnx>
                          [--max-image-size 960] [--min-score 0.2] [--lightglue-min-score 0.1]

LightGlue matches poorly across a large in-plane rotation: photos taken with the phone held one
way barely match photos taken with it turned 90 degrees, and the few matches it makes are
mostly wrong. Every photo is first turned by a multiple of 90 degrees so that the world's up (from
the poses, ARKit's +y) points up in the image; features are extracted and matched on these
copies; then the keypoints are mapped back into each original photo's frame, so the database
reads like any other (time_correct.py, match_sweeps.py, ...). Writes <work>/database.db.
"""
import argparse
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

import gsdata


def quarter_turns(R, up=(0.0, 1.0, 0.0)):
    """Counter-clockwise quarter turns that bring the world's up closest to image up, for a
    camera-to-world rotation R."""
    u = (R.T @ np.asarray(up))[:2]
    best, k, v = -2.0, 0, u / max(np.linalg.norm(u), 1e-12)
    for j in range(4):
        if -v[1] > best:
            best, k = -v[1], j
        v = np.array([v[1], -v[0]])   # one CCW image turn (y down)
    return k


def unturn(xy, k, w, h):
    """Keypoints found in a photo turned k quarter turns CCW, back in the original w x h photo."""
    x, y = xy[:, 0].copy(), xy[:, 1].copy()
    if k == 1:
        x, y = w - y, x
    elif k == 2:
        x, y = w - x, h - y
    elif k == 3:
        x, y = y, h - x
    out = xy.copy()
    out[:, 0], out[:, 1] = x, y
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--colmap", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--model", required=True, help="poses (COLMAP text model), for each photo's up direction")
    p.add_argument("--work", required=True)
    p.add_argument("--models", required=True, help="folder with aliked-n16rot.onnx and aliked-lightglue.onnx")
    p.add_argument("--max-image-size", type=int, default=960)
    p.add_argument("--max-features", type=int, default=8192)
    p.add_argument("--min-score", type=float, default=0.2)
    p.add_argument("--lightglue-min-score", type=float, default=0.1)
    p.add_argument("--max-error", type=float, default=4.0, help="px, COLMAP's two-view verification")
    p.add_argument("--cuda-path", help="file whose contents are prepended to PATH (CUDA libraries for onnxruntime)")
    a = p.parse_args(argv)
    if a.cuda_path:
        os.environ["PATH"] = Path(a.cuda_path).read_text().strip() + os.pathsep + os.environ["PATH"]
    src, work = Path(a.images), Path(a.work)
    turned = work / "upright_images"
    if work.exists():
        shutil.rmtree(work)
    turned.mkdir(parents=True)
    ks = {}
    for im in gsdata.read_model(Path(a.model))[0]:
        f = src / im["name"]
        if not f.exists():
            continue
        k = quarter_turns(np.linalg.inv(im["viewmat"])[:3, :3])
        ks[im["name"]] = k
        img = Image.open(f)
        (img.rotate(90 * k, expand=True) if k else img).save(turned / im["name"], quality=96)
    print(f"{len(ks)} photos; quarter turns: " + ", ".join(f"{k}: {sum(v == k for v in ks.values())}" for k in range(4)),
          flush=True)
    db = work / "database.db"
    log = open(work / "colmap.log", "w")

    def run(args):
        r = subprocess.run([a.colmap] + args + ["--log_target", "stderr"], stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            raise SystemExit(f"colmap {args[0]} failed (exit {r.returncode}); see {work / 'colmap.log'}")
    m = Path(a.models).as_posix()
    run(["feature_extractor", "--database_path", str(db), "--image_path", str(turned),
         "--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera_per_image", "1",
         "--FeatureExtraction.use_gpu", "1", "--FeatureExtraction.max_image_size", str(a.max_image_size),
         "--FeatureExtraction.type", "ALIKED_N16ROT", "--AlikedExtraction.n16rot_model_path", f"{m}/aliked-n16rot.onnx",
         "--AlikedExtraction.max_num_features", str(a.max_features), "--AlikedExtraction.min_score", str(a.min_score)])
    print("features extracted", flush=True)
    run(["exhaustive_matcher", "--database_path", str(db), "--FeatureMatching.use_gpu", "1",
         "--FeatureMatching.type", "ALIKED_LIGHTGLUE", "--AlikedMatching.lightglue_model_path", f"{m}/aliked-lightglue.onnx",
         "--AlikedMatching.lightglue_min_score", str(a.lightglue_min_score), "--TwoViewGeometry.max_error", str(a.max_error)])
    print("matched", flush=True)
    con = sqlite3.connect(db)
    names = {i: n for i, n in con.execute("select image_id, name from images")}
    for i, r, c, d in list(con.execute("select image_id, rows, cols, data from keypoints")):
        k = ks.get(names[i], 0)
        if k == 0 or r == 0:
            continue
        kp = np.frombuffer(d, np.float32).reshape(r, c).copy()
        w, h = Image.open(src / names[i]).size
        kp[:, :2] = unturn(kp[:, :2], k, w, h)
        con.execute("update keypoints set data = ? where image_id = ?", (kp.tobytes(), i))
    con.commit()
    con.close()
    shutil.rmtree(turned, ignore_errors=True)
    print(f"keypoints back in the original frames -> {db}", flush=True)


if __name__ == "__main__":
    main()
