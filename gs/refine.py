"""Refine a SplatKing capture's ARKit/LiDAR camera poses with COLMAP. Standard library only, so
it runs with the Mac's Python or on a worker; GaussQueue runs it on whichever machine refines.

  python refine.py --colmap <colmap> --images <dir> --model <sparse/0 of the capture>
                   --work <scratch dir> --out <dir> [--gpu]

SIFT features -> matching (exhaustive for small captures, else spatial, using the known
positions as priors) -> triangulate with the known poses -> bundle-adjust poses and points
(intrinsics fixed) -> similarity-align back onto the original camera centres, so the capture's
LiDAR points and "up" still line up.

Writes to --out: images.txt (refined poses, in the capture's order and camera id),
cameras.txt (unchanged intrinsics), points3D_features.txt (COLMAP's triangulated feature
points) and stats.json. Prints "PHASE: ..." lines for progress. Exits non-zero, with the
reason on the last line, if refinement fails or looks worse than the original poses.
"""
import argparse
import json
import math
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import time
from pathlib import Path


def phase(text):
    print(f"PHASE: {text}", flush=True)


def quat_to_R(w, x, y, z):
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def read_poses(images_txt):
    """{name: (image line fields, quaternion, camera centre)} from a COLMAP images.txt."""
    out = {}
    for line in open(images_txt):
        p = line.split()
        if line.startswith("#") or len(p) < 10 or not p[0].isdigit():
            continue
        q = list(map(float, p[1:5]))
        t = list(map(float, p[5:8]))
        R = quat_to_R(*q)
        out[p[9]] = (p, q, [-(R[0][i] * t[0] + R[1][i] * t[1] + R[2][i] * t[2]) for i in range(3)])
    return out


def run(cmd, log, timeout=4 * 3600):
    with open(log, "a") as lf:
        lf.write("$ " + " ".join(map(str, cmd)) + "\n")
        lf.flush()
        t = time.time()
        r = subprocess.run([str(c) for c in cmd], stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
        lf.write(f"# {time.time() - t:.0f} s\n")
    if r.returncode:
        raise RuntimeError(f"colmap {cmd[1]} failed (exit {r.returncode}); see {log}")


def mean_reprojection(colmap, model):
    r = subprocess.run([str(colmap), "model_analyzer", "--path", str(model), "--log_target", "stderr"],
                       capture_output=True, text=True)
    m = re.search(r"Mean reprojection error: ([\d.]+)", r.stderr + r.stdout)
    return float(m.group(1)) if m else None


def refine(colmap, images, src, work, out, gpu=False):
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    out.mkdir(parents=True, exist_ok=True)
    log, db = work / "colmap.log", work / "database.db"
    use_gpu = 1 if gpu else 0
    cams = [l.split() for l in open(src / "cameras.txt") if l.strip() and not l.startswith("#")]
    if len(cams) != 1:
        raise RuntimeError("pose refinement supports single-camera captures only")
    model, params = cams[0][1], ",".join(cams[0][4:])
    orig = read_poses(src / "images.txt")
    n = len(orig)
    t0 = time.time()

    phase(f"features ({n} photos)")
    run([colmap, "feature_extractor", "--database_path", db, "--image_path", images,
         "--ImageReader.camera_model", model, "--ImageReader.single_camera", 1,
         "--ImageReader.camera_params", params, "--FeatureExtraction.use_gpu", use_gpu,
         "--FeatureExtraction.max_image_size", 1600, "--log_target", "stderr"], log)

    con = sqlite3.connect(db)
    ids = {nm: (iid, cid) for iid, nm, cid in con.execute("select image_id, name, camera_id from images")}
    frames = {d: (f, r) for f, r, d in con.execute(
        "select f.frame_id, f.rig_id, d.data_id from frames f join frame_data d on f.frame_id = d.frame_id")}
    phase("matching")
    if n <= 120:
        con.close()
        run([colmap, "exhaustive_matcher", "--database_path", db,
             "--FeatureMatching.use_gpu", use_gpu, "--log_target", "stderr"], log)
    else:  # match each photo with its spatial neighbours, using the known positions
        cov = struct.pack("<9d", 0.04, 0, 0, 0, 0.04, 0, 0, 0, 0.04)
        for nm, (_, _, c) in orig.items():
            iid, cid = ids[nm]
            con.execute("insert into pose_priors (corr_data_id, corr_sensor_id, corr_sensor_type, "
                        "position, position_covariance, gravity, coordinate_system) "
                        "values (?,?,0,?,?,NULL,1)", (iid, cid, struct.pack("<3d", *c), cov))
        con.commit()
        con.close()
        run([colmap, "spatial_matcher", "--database_path", db, "--FeatureMatching.use_gpu", use_gpu,
             "--SpatialMatching.ignore_z", 0, "--SpatialMatching.max_num_neighbors", 30,
             "--SpatialMatching.max_distance", 10, "--log_target", "stderr"], log)

    # Known-pose model with the database's ids (COLMAP 4 also wants rigs + frames).
    known = work / "known"
    known.mkdir()
    cid = next(iter(ids.values()))[1]
    (known / "cameras.txt").write_text(" ".join([str(cid)] + cams[0][1:]) + "\n")
    (known / "rigs.txt").write_text(f"1 1 CAMERA {cid}\n")
    img_lines, frame_lines = [], []
    for nm, (p, _, _) in orig.items():
        iid, c = ids[nm]
        fid, rig = frames[iid]
        img_lines += [" ".join([str(iid)] + p[1:8] + [str(c), nm]), ""]
        frame_lines.append(" ".join([str(fid), str(rig)] + p[1:8] + ["1", "CAMERA", str(c), str(iid)]))
    (known / "images.txt").write_text("\n".join(img_lines) + "\n")
    (known / "frames.txt").write_text("\n".join(frame_lines) + "\n")
    (known / "points3D.txt").write_text("")

    phase("bundle adjustment")
    tri, ba, aligned, txt = (work / d for d in ("tri", "ba", "aligned", "txt"))
    for d in (tri, ba, aligned, txt):
        d.mkdir()
    run([colmap, "point_triangulator", "--database_path", db, "--image_path", images,
         "--input_path", known, "--output_path", tri, "--refine_intrinsics", 0,
         "--log_target", "stderr"], log)
    before = mean_reprojection(colmap, tri)
    run([colmap, "bundle_adjuster", "--input_path", tri, "--output_path", ba,
         "--BundleAdjustment.refine_focal_length", 0,
         "--BundleAdjustment.refine_principal_point", 0,
         "--BundleAdjustment.refine_extra_params", 0,
         "--BundleAdjustment.refine_sensor_from_rig", 0, "--log_target", "stderr"], log)
    after = mean_reprojection(colmap, ba)
    ref = work / "ref_centres.txt"
    ref.write_text("".join(f"{nm} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n" for nm, (_, _, c) in orig.items()))
    run([colmap, "model_aligner", "--input_path", ba, "--output_path", aligned,
         "--ref_images_path", ref, "--ref_is_gps", 0, "--alignment_type", "custom",
         "--alignment_max_error", 0.25, "--log_target", "stderr"], log)
    run([colmap, "model_converter", "--input_path", aligned, "--output_path", txt,
         "--output_type", "TXT", "--log_target", "stderr"], log)

    new = read_poses(txt / "images.txt")
    common = [nm for nm in orig if nm in new]
    if len(common) < 0.9 * n:
        raise RuntimeError(f"only {len(common)}/{n} photos survived refinement")
    shift = sorted(math.dist(orig[nm][2], new[nm][2]) * 100 for nm in common)
    turn = sorted(math.degrees(2 * math.acos(min(1, abs(sum(a * b for a, b in zip(orig[nm][1], new[nm][1]))))))
                  for nm in common)
    stats = {"photos": n, "reprojection_px_before": before, "reprojection_px_after": after,
             "median_shift_cm": round(shift[len(shift) // 2], 1), "max_shift_cm": round(shift[-1], 1),
             "median_turn_deg": round(turn[len(turn) // 2], 2), "max_turn_deg": round(turn[-1], 2),
             "seconds": round(time.time() - t0), "gpu": bool(gpu)}
    if after is None or (before is not None and after > before * 1.05) or shift[-1] > 200:
        raise RuntimeError(f"refinement looked worse than the original poses: {stats}")

    lines = []
    for nm in orig:   # the capture's order and camera id, poses from the refined model
        p = new[nm][0] if nm in new else orig[nm][0]
        lines += [" ".join(p[:8] + [cams[0][0], nm]), ""]
    (out / "images.txt").write_text("\n".join(lines) + "\n")
    (out / "cameras.txt").write_text(" ".join(cams[0]) + "\n")   # intrinsics unchanged
    shutil.copyfile(txt / "points3D.txt", out / "points3D_features.txt")
    (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    phase("done")
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--colmap", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--model", required=True, help="the capture's sparse/0 (cameras.txt, images.txt)")
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", action="store_true", help="SIFT extraction and matching on the GPU (CUDA)")
    a = p.parse_args(argv)
    try:
        stats = refine(a.colmap, Path(a.images), Path(a.model), Path(a.work), Path(a.out), a.gpu)
        print(json.dumps(stats), flush=True)
    except Exception as e:
        print(f"error: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
