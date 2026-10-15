"""Refine a SplatKing capture's ARKit/LiDAR camera poses with COLMAP. Standard library only, so
it runs with the Mac's Python or on a worker; GaussQueue runs it on whichever machine refines.

  python refine.py --colmap <colmap> --images <dir> --model <sparse/0 of the capture>
                   --work <scratch dir> --out <dir> [--gpu]

SIFT features -> matching (exhaustive for small captures, else spatial, using the known
positions as priors) -> triangulate with the known poses -> bundle-adjust poses and points
(intrinsics fixed) -> similarity-align back onto the original camera centres, so the capture's
LiDAR points and "up" still line up.

The model may have one camera for all photos or one per photo (PINHOLE, e.g. each photo's
own ARKit intrinsics: autofocus changes the focal length by several percent).

With --lens, the lens is calibrated after the poses: one radial distortion (k1, k2) and focal
length for the whole capture, fitted on a shared camera. Each photo keeps its own starting
focal length (ARKit's, which tracks autofocus) scaled so their median matches the fitted one;
fitting each photo's focal freely is unreliable (a photo of mostly flat surfaces can trade
focal length for distance, the "dolly zoom"). Poses then settle again with these cameras.
cameras.txt holds RADIAL cameras (one per photo); undistort the photos with them (colmap
image_undistorter) before training. Without --lens the
intrinsics are kept as given.

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


def read_txt_model(d):
    """(camera lines, image header lines with their 2D-point lines, rigs lines, frames lines)."""
    keep = lambda f: [l.rstrip("\n") for l in open(d / f) if l.strip() and not l.startswith("#")] if (d / f).exists() else []
    lines = [l.rstrip("\n") for l in open(d / "images.txt") if not l.startswith("#")]
    while lines and not lines[0].strip():
        lines.pop(0)
    pairs = [(lines[i], lines[i + 1] if i + 1 < len(lines) else "") for i in range(0, len(lines), 2)]
    return keep("cameras.txt"), pairs, keep("rigs.txt"), keep("frames.txt")


def write_txt_model(d, cams, pairs, rigs, frames, points_from):
    d.mkdir(parents=True, exist_ok=True)
    (d / "cameras.txt").write_text("\n".join(cams) + "\n")
    (d / "images.txt").write_text("".join(f"{h}\n{p}\n" for h, p in pairs))
    (d / "rigs.txt").write_text("\n".join(rigs) + "\n")
    (d / "frames.txt").write_text("\n".join(frames) + "\n")
    shutil.copyfile(points_from / "points3D.txt", d / "points3D.txt")


def intrinsics(c):
    """(fx, fy, cx, cy) of a camera line's fields (PINHOLE, SIMPLE_PINHOLE, OPENCV...)."""
    v = list(map(float, c[4:]))
    return (v[0], v[0], v[1], v[2]) if c[1].startswith("SIMPLE") else tuple(v[:4])


def calibrate_lens(colmap, ba, work, log, focal_blend=1.0):
    """Fit the capture's lens on the bundle-adjusted model (no new features or matches):
    shared OPENCV distortion first, then each photo's focal with it fixed. Returns
    (model folder, distortion [k1 k2 p1 p2], reprojection before, after)."""
    txt = work / "lens_in"
    txt.mkdir()
    run([colmap, "model_converter", "--input_path", ba, "--output_path", txt, "--output_type", "TXT",
         "--log_target", "stderr"], log)
    cams, pairs, rigs, frames = read_txt_model(txt)
    params = {c.split()[0]: c.split() for c in cams}
    intr = {cid: intrinsics(c) for cid, c in params.items()}
    first = next(iter(params.values()))
    w, h = first[2], first[3]
    fxm, fym, cxm, cym = (sorted(v[i] for v in intr.values())[len(intr) // 2] for i in range(4))
    fm = (fxm + fym) / 2
    # 1. one shared OPENCV camera: every image, frame and rig points at camera 1
    shared = work / "lens_shared"
    one_pairs = [(" ".join(h_.split()[:8] + ["1"] + h_.split()[9:]), p) for h_, p in pairs]
    one_frames = []
    for f in frames:
        t = f.split()
        n = int(t[9])
        data = []
        for k in range(n):
            st_, _sid, did = t[10 + 3 * k: 13 + 3 * k]
            data += [st_, "1", did]
        one_frames.append(" ".join([t[0], "1"] + t[2:10] + data))
    write_txt_model(shared, [f"1 RADIAL {w} {h} {fm} {cxm} {cym} 0 0"], one_pairs,
                    ["1 1 CAMERA 1"], one_frames, txt)
    shared_out = work / "lens_shared_ba"
    shared_out.mkdir()
    run([colmap, "bundle_adjuster", "--input_path", shared, "--output_path", shared_out,
         "--BundleAdjustment.refine_focal_length", 1, "--BundleAdjustment.refine_principal_point", 0,
         "--BundleAdjustment.refine_extra_params", 1, "--BundleAdjustment.refine_sensor_from_rig", 0,
         "--log_target", "stderr"], log)
    sh = work / "lens_shared_txt"
    sh.mkdir()
    run([colmap, "model_converter", "--input_path", shared_out, "--output_path", sh, "--output_type", "TXT",
         "--log_target", "stderr"], log)
    c = read_txt_model(sh)[0][0].split()
    dist = c[7:9]   # RADIAL: f cx cy k1 k2
    # 2. each photo's own focal (scaled to the fitted median) with the shared distortion;
    #    poses and points settle again, intrinsics fixed
    f_fit = float(c[4])
    scale = f_fit / fm
    last = work / "lens_final"
    per_cams = []
    for cid, c0 in params.items():
        fx_, fy_, cx_, cy_ = intr[cid]
        f_own = (fx_ + fy_) / 2 * scale   # this photo's ARKit focal, scaled to the fit
        f = f_fit + focal_blend * (f_own - f_fit)   # 0 = one shared focal, 1 = each photo's own
        per_cams.append(" ".join([cid, "RADIAL", c0[2], c0[3], repr(f), repr(cx_), repr(cy_)] + dist))
    write_txt_model(last, per_cams, pairs, rigs, frames, txt)
    clamped = 0
    out_ = work / "lens_ba"
    out_.mkdir()
    run([colmap, "bundle_adjuster", "--input_path", last, "--output_path", out_,
         "--BundleAdjustment.refine_focal_length", 0, "--BundleAdjustment.refine_principal_point", 0,
         "--BundleAdjustment.refine_extra_params", 0, "--BundleAdjustment.refine_sensor_from_rig", 0,
         "--log_target", "stderr"], log)
    return out_, [float(x) for x in dist], clamped, mean_reprojection(colmap, ba), mean_reprojection(colmap, out_)


def refine(colmap, images, src, work, out, gpu=False, lens=False, focal_blend=1.0):
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    out.mkdir(parents=True, exist_ok=True)
    log, db = work / "colmap.log", work / "database.db"
    use_gpu = 1 if gpu else 0
    cams = [l.split() for l in open(src / "cameras.txt") if l.strip() and not l.startswith("#")]
    per_image = len(cams) > 1
    if per_image and any(c[1] != "PINHOLE" for c in cams):
        raise RuntimeError("per-photo cameras must be PINHOLE")
    cam_by_id = {c[0]: c for c in cams}
    model, params = cams[0][1], ",".join(cams[0][4:])
    orig = read_poses(src / "images.txt")
    n = len(orig)
    t0 = time.time()

    phase(f"features ({n} photos)")
    cam_args = (["--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera_per_image", 1]
                if per_image else
                ["--ImageReader.camera_model", model, "--ImageReader.single_camera", 1,
                 "--ImageReader.camera_params", params])
    run([colmap, "feature_extractor", "--database_path", db, "--image_path", images, *cam_args,
         "--FeatureExtraction.use_gpu", use_gpu,
         "--FeatureExtraction.max_image_size", 1600, "--log_target", "stderr"], log)

    con = sqlite3.connect(db)
    ids = {nm: (iid, cid) for iid, nm, cid in con.execute("select image_id, name, camera_id from images")}
    own = {}   # db camera id -> the photo's camera line (per-photo intrinsics)
    if per_image:   # give each photo's database camera its real intrinsics (used in matching too)
        for nm, (iid, cid) in ids.items():
            if nm not in orig:
                continue
            c = cam_by_id[orig[nm][0][8]]
            own[cid] = c
            con.execute("update cameras set model = 1, params = ?, prior_focal_length = 1 where camera_id = ?",
                        (struct.pack("<4d", *map(float, c[4:8])), cid))
        con.commit()
    rigs = {cid: rid for rid, cid in con.execute("select rig_id, ref_sensor_id from rigs where ref_sensor_type = 0")}
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
    if per_image:
        (known / "cameras.txt").write_text("".join(" ".join([str(cid)] + c[1:]) + "\n" for cid, c in own.items()))
        (known / "rigs.txt").write_text("".join(f"{rigs[cid]} 1 CAMERA {cid}\n" for cid in own))
    else:
        cid = next(iter(ids.values()))[1]
        (known / "cameras.txt").write_text(" ".join([str(cid)] + cams[0][1:]) + "\n")
        (known / "rigs.txt").write_text(f"{rigs.get(cid, 1)} 1 CAMERA {cid}\n")
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
    lens_stats = None
    if lens:
        phase("lens calibration")
        try:
            lba, dist, clamped, lb, la = calibrate_lens(colmap, ba, work, log, focal_blend)
            if la is not None and lb is not None and la <= lb * 1.02:
                ba = lba
                fs = sorted(float(l.split()[4]) for l in read_txt_model(work / "lens_final")[0])
                f = fs[len(fs) // 2]
                edge = max(abs((dist[0] * r * r + dist[1] * r ** 4) * r * f)
                           for r in (x / f for x in (400, 600, 800, 960, 1200)))
                lens_stats = {"distortion_k1_k2": [round(x, 6) for x in dist], "reprojection_px": round(la, 4),
                              "max_shift_px": round(edge, 1), "focal_px": [round(fs[0]), round(f), round(fs[-1])]}
                after = la
            else:
                lens_stats = {"skipped": f"no better ({lb} -> {la} px)"}
        except Exception as e:   # keep the fixed-intrinsics result
            lens_stats = {"skipped": str(e)[:160]}
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
    if lens_stats:
        stats["lens"] = lens_stats
    if after is None or (before is not None and after > before * 1.05) or shift[-1] > 200:
        raise RuntimeError(f"refinement looked worse than the original poses: {stats}")

    lines = []
    for nm in orig:   # the capture's order and camera ids, poses from the refined model
        p = new[nm][0] if nm in new else orig[nm][0]
        lines += [" ".join(p[:8] + [orig[nm][0][8], nm]), ""]
    (out / "images.txt").write_text("\n".join(lines) + "\n")
    if lens_stats and "distortion_k1_k2" in lens_stats:   # calibrated cameras, under the capture's ids
        fitted = {c.split()[0]: c.split() for c in read_txt_model(txt)[0]}
        db_id = {nm: str(new[nm][0][8]) for nm in new}
        out_cams = {}
        for nm in orig:
            cid = orig[nm][0][8]
            if cid not in out_cams:
                c = fitted.get(db_id.get(nm, ""))
                out_cams[cid] = ([cid] + c[1:]) if c else cam_by_id[cid]
        (out / "cameras.txt").write_text("".join(" ".join(c) + "\n" for c in out_cams.values()))
    else:
        (out / "cameras.txt").write_text("".join(" ".join(c) + "\n" for c in cams))   # intrinsics unchanged
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
    p.add_argument("--lens", action="store_true", help="also calibrate the lens (distortion + per-photo focal)")
    p.add_argument("--focal-blend", type=float, default=1.0,
                   help="with --lens: how much of each photo's own focal to keep (0 = one shared focal, 1 = own)")
    a = p.parse_args(argv)
    try:
        stats = refine(a.colmap, Path(a.images), Path(a.model), Path(a.work), Path(a.out), a.gpu, a.lens, a.focal_blend)
        print(json.dumps(stats), flush=True)
    except Exception as e:
        print(f"error: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
