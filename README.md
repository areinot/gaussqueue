# Note for Humans
This is some AI vibe-coded foolishness so uh, clone at your own risk.

# GaussQueue

A dashboard and training queue for Gaussian splats on a Mac.

Drop a capture into an inbox folder (an iCloud Drive folder works well, so you
can send captures from your phone), and GaussQueue copies it to local disk, trains
it with [Brush](https://github.com/ArthurBrussee/brush) on the Mac's GPU, and puts
the finished `.ply` in an outbox folder. You can follow along on a localhost dashboard, in a
`STATUS.txt` file, or by iMessage.

![GaussQueue dashboard](docs/dashboard.png)

It was built for [SplatKing](https://radiancefields.com/platforms/splatking) LiDAR
captures, which ship with camera poses as a COLMAP model, so there is no
structure-from-motion step. Any capture that contains a COLMAP model
(`sparse/0/cameras.{txt,bin}` next to an `images/` folder) should work.

## Features

- **Inbox → work → outbox.** Captures can be zips or folders. Each item has to stay unchanged
  for 30 s before it's picked up, so half-synced iCloud uploads are left alone.
  Training happens in a local `work_dir`, and only results go to `outbox_dir`.
- **Instant thumbnails.** When a new item arrives, the middle frame is read straight out
  of the zip or folder (no unpacking) and rotated upright using the capture's
  orientation metadata.
- **Capture ratings.** SplatKing rates every photo at capture time ("stable AR tracking",
  "fast movement", "large pose jump", …) in `splatpack.json`. GaussQueue reads these as soon
  as a capture arrives and shows them as a bar on the inbox card, the training panel and the
  history. It flags ⚠ *low overlap* when most photos are "large pose jumps" (the phone moved
  too far between shots, the usual cause of blurry results). The ratings also go into the 📥
  text and `meta.json`.
- **Dashboard** at `http://127.0.0.1:8765`. It shows live progress, the checkpoint
  previews (renders from a camera held out of training), the queue and recent results.
- **Per-capture quality settings.** Preview / Fast / Standard / High presets, or set the steps,
  resolution, SH degree, splat cap and frame subsampling yourself. Captures can also be
  held or moved to the front. Each queued capture shows an estimated duration.
- **Compressed copies.** Each result also gets an `.spz` (SPZ v3 by default). It's typically
  10–15× smaller than the `.ply` and much easier to open on a phone.
- **Sharper poses (optional).** SplatKing's LiDAR export uses ARKit's camera poses as
  they are, with no bundle adjustment, so small drift blurs the result. With `refine_poses`
  (also a per-capture checkbox), COLMAP refines them first. It finds SIFT features, matches
  every pair of photos (or, for big captures, spatial neighbours using the known positions),
  triangulates, bundle-adjusts poses and points with the intrinsics fixed, and then aligns
  the result back onto the original frame so the LiDAR points and "up" still line up. On a
  68-photo capture this took about 1.5 minutes on CPU and halved the mean reprojection error
  (1.26 → 0.56 px). `meta.json` records the before/after numbers. If refinement fails or makes
  things worse, the capture trains on the original poses and the problem is reported.
  Refined poses are cached in `$GAUSSQUEUE_HOME/pose_cache/`, keyed by the capture's original
  poses, the method version and the COLMAP version. A quick Preview run followed by a High
  run of the same capture only pays for COLMAP once.
- **Right way up.** SplatKing/COLMAP captures use +Y as up, but most splat viewers treat
  −Y as up, so results look upside down. With `flip_up` (on in the example config; it can
  also be set per capture, even mid-training), results are rotated 180° about X before
  compression. The rotation covers the view-dependent colour data too. `meta.json` records
  what was applied.
- **Rerun at higher quality.** Each finished or failed capture in *Recent* has a
  **Requeue** button. It moves the original capture back into the inbox, on hold, so you
  can choose new settings and then release the hold. When the rerun finishes, the earlier
  result is moved to `<outbox>/<name>/previous/` instead of being overwritten.
- **Test runs.** The *Preview* preset (720 px, 3k steps, SH 1, ≤1M splats) takes a few
  minutes. It shows whether a capture holds together (no ghosting, right way up, decent
  coverage) before you commit to hours of training. It can't predict fine detail. The
  resolution choices only go up to the capture's own photo size, since photos are never upscaled.
- **Mid-training snapshots.** *Publish snapshot* on the running capture turns its latest
  checkpoint into a normal result, `<name>-step<N>`: flipped `.ply`, `.spz`,
  preview and `meta.json`, plus the phone copies. Training keeps going. The checkpoint is
  hard-linked aside instantly. Processing waits until at least `snapshot_min_free_pct`
  (default 25 %) of memory is free, or until training finishes, so it can't squeeze the run.
- **Pause and tweak.** Pause the running capture: Brush is frozen in place
  (`SIGSTOP`) and the Mac is allowed to sleep. Resume to carry on where it stopped,
  or change its settings while it's paused and resume to restart it from step 0 with
  them. The capture stays on local disk, so nothing is copied again.
- **Notifications (optional).** iMessages when a capture starts, at 25/50/75 % with a
  preview image, and when it finishes or fails.
- **Unattended.** Runs as a launchd agent that starts at login and restarts if it
  crashes. It keeps the Mac awake only while training. When `gaussqueue.py` changes,
  it reloads itself the next time it's idle.

## Requirements

- macOS on Apple Silicon (the pinned Brush build is `aarch64-apple-darwin`).
  It uses the built-in macOS tools `brctl`, `sips`, `caffeinate`, `osascript` and launchd.
- Python 3.9+. The system `python3` from the Xcode Command Line Tools is fine. There are
  no third-party packages.
- Optional: [COLMAP](https://colmap.github.io) 4.x for pose refinement (`brew install colmap`).
- Optional: CMake + a C++17 compiler (`xcode-select --install`, `brew install cmake`) to
  build **gq_spz** (`tools/spz-native`), a small native tool built on
  [Niantic's spz library](https://github.com/nianticlabs/spz). When it's installed,
  GaussQueue uses it for `.spz` output and for flips. SplatTransform's SPZ encoder is
  WebAssembly with a 4 GB memory ceiling, and it aborts on large (~10M-splat) captures.
- Optional: [Node.js](https://nodejs.org) (`brew install node`) for SplatTransform, the
  fallback `.spz` encoder when gq_spz isn't built. With neither, results are `.ply` only.
- For iMessage notifications: the Messages app signed in to iMessage on the Mac.

## Install

```bash
git clone https://github.com/areinot/gaussqueue.git ~/code/gaussqueue
cd ~/code/gaussqueue
scripts/install.sh --inbox "~/Library/Mobile Documents/com~apple~CloudDocs/GaussQueue/inbox"
```

`install.sh`:

1. downloads Brush v0.3.0 and verifies its SHA-256, then, if Node.js is present,
   installs SplatTransform 3.10.1 into a private folder (not a global npm install),
2. writes `~/Library/Application Support/GaussQueue/config.json` from
   [`config.example.json`](config.example.json). By default the outbox and status
   folders go next to the inbox. Pass `--outbox`, `--status` or `--work` to put them elsewhere.
3. runs `gaussqueue.py --check`, then installs and starts the launchd agent.

Re-running is safe; it never overwrites an existing config. To update, run
`git pull` and then re-run `install.sh`. To remove it, run `scripts/uninstall.sh`,
which leaves your data in place.

## Remote GPU workers (optional)

A Windows PC with an NVIDIA GPU can do the training while the Mac keeps coordinating.
The Mac still watches the inbox, refines poses, compresses, publishes and texts. For each
capture it streams the photos and poses to the PC over SSH, runs Brush there, pulls the
progress renders back for the dashboard, and downloads the finished `.ply`. Two captures
can train at once, one on each machine. Captures set to *Auto* go to an idle remote worker
first, then to the Mac. Each capture can also be pinned with **Run on**. The dashboard's
*Machines* panel shows each machine's job, GPU load and memory.

Setup on the PC (Windows 10/11, PowerShell as Administrator):

1. Add *OpenSSH Server* (Settings → System → Optional features), then
   `Start-Service sshd; Set-Service sshd -StartupType Automatic`, and allow TCP 22 in the firewall.
2. Authorize a key that exists only for this purpose (`ssh-keygen -t ed25519 -f ~/.ssh/gaussqueue_worker`
   on the Mac). For an administrator account, append the **public** key to
   `%ProgramData%\ssh\administrators_authorized_keys` and restrict that file with
   `icacls ... /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F"`.
3. Download the Brush Windows release, check its SHA-256, and unzip it, e.g. to `D:\gaussqueue\brush`.

On the Mac, add an alias to `~/.ssh/config` (it keeps the address and key out of GaussQueue's config):

```
Host gpu-pc
    HostName <pc-address>
    User <windows-user>
    IdentityFile ~/.ssh/gaussqueue_worker
    IdentitiesOnly yes
```

Then list the machine in `config.json`:

```json
"workers": [{"name": "pc", "label": "PC (RTX 4080)", "ssh": "gpu-pc",
             "brush": "D:/gaussqueue/brush/brush_app.exe", "work_dir": "D:/gaussqueue/jobs"}]
```

Set `"local_training": false` to train only on remote workers. Limitations: *Pause* isn't
available on remote machines yet. If the SSH connection drops mid-run, the capture fails and
can be requeued.

## gsplat trainer and pruning (CUDA workers, optional)

A remote worker with an NVIDIA GPU can also run [gsplat](https://github.com/nerfstudio-project/gsplat),
through the small trainer and pruner in `gs/`. The Mac copies `gs/*.py` to the worker
before every job, so the worker always runs the code in this repo.

- **Trainer: gsplat** (per capture, or `"trainer": "gsplat"` for everything). Adds what Brush
  doesn't have:
  - **LiDAR depth supervision.** SplatKing saves a depth map and a confidence map with every
    photo. Rendered depth is pulled toward the LiDAR depth only where ARKit is confident (near
    surfaces, up to about 5 m), so sky and distant scenery are left to the photos.
  - **Flatten.** Penalises each splat's thinnest axis so splats lie flat on surfaces.
  - **Densify:** *Default* (classic 3DGS growth), *Abs-grad* (grows more in fine detail) or
    *MCMC* (a fixed budget: it always ends at the splat cap and moves weak splats to where
    they're needed).
  - Held-out photos (every `eval_split_every`th) are scored after training: PSNR, SSIM and
    the median depth error against the LiDAR. These land in `meta.json`.
- **Prune** (per capture, or `prune_pct`). After training, every splat is scored by how much
  it matters to the photos, the lowest `prune_pct` % are removed, and the rest are fine-tuned
  for `prune.finetune_steps` so they cover the gaps. Two scores are available:
  `lightgaussian` (default; LightGaussian: each splat's total contribution to every pixel) and
  `pup` (PUP-3DGS: how sensitive the photos are to each splat's position and size, estimated
  here with random probes; it kept less fidelity than `lightgaussian` in our tests). Works on
  results from either trainer. Pruning runs on the CUDA worker; Mac-trained results are
  uploaded for it. The unpruned result stays in the work folder, and `meta.json` records the
  fidelity before and after.
- **Feature points** (`init_feature_points`, on by default; needs `refine_poses`). Training
  starts from COLMAP's triangulated feature points as well as the LiDAR points. LiDAR only
  reaches about 5 m. Without the feature points, distant trees, buildings and sky have no
  starting splats, and training covers them with large blurry splats near the camera. This
  applies to Brush too.

Set up the worker once (Windows, PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup-gsplat-worker.ps1 -Root D:\gaussqueue
```

It installs a checksum-verified uv, Python 3.10, torch 2.4.1 (CUDA 12.4) and gsplat 1.5.3's
prebuilt wheel, all under the root folder. It doesn't need a compiler or touch the system
Python. Then add the line it prints to the worker's entry:

```json
"workers": [{"name": "pc", "label": "PC (RTX 4080)", "ssh": "gpu-pc",
             "brush": "D:/gaussqueue/brush/brush_app.exe", "work_dir": "D:/gaussqueue/jobs",
             "gsplat": {"python": "D:/gaussqueue/gsplat/venv/Scripts/python.exe",
                        "scripts": "D:/gaussqueue/gsplat/gs"}}]
```

Captures set to gsplat wait for a machine that has it. If no machine does, or the capture
is pinned to the Mac with *Run on*, it trains with Brush and says so in the history.
The scripts also run by hand on any CUDA machine:

```bash
python gs/train.py <capture> --out out --depth-dir <capture>/sensor_data --depth-weight 0.3 --flat-weight 10
python gs/prune.py out/export_30000.ply <capture> --out pruned --keep 0.3 --finetune 3000
```

## Folders

| Setting | Default | What goes there |
|---|---|---|
| `inbox_dir` | *(required)* | Captures to train. Processed items move to `inbox/done/` or `inbox/failed/`. |
| `work_dir` | `~/Library/Application Support/GaussQueue/work` | Unpacked captures and training output. **Keep this out of iCloud.** Note that `~/Documents` *is* in iCloud if "Desktop & Documents Folders" is enabled. Cleaned up after `keep_work_days`. |
| `outbox_dir` | next to the inbox | One folder per capture: `<name>.ply`, `<name>.spz`, `preview.jpg`, `meta.json`, `train.log`. Nothing else is written here, so it is safe to share. |
| `share_dir` | *(off)* | Optional. Also copies each result's small, phone-friendly files (`share_files`, default `.spz`, `preview.jpg`, `meta.json`) here, e.g. to iCloud, while the full results stay local in `outbox_dir`. |
| `status_dir` | next to the inbox | `STATUS.txt` and preview JPGs. Put it in iCloud to check progress from your phone. |

App state (`config.json`, `state.json`, `overrides.json`, `gaussqueue.log`, the Brush
binary) lives in `$GAUSSQUEUE_HOME`, which defaults to `~/Library/Application Support/GaussQueue`.

## Configuration

`config.json` is re-read on every poll, so most changes apply without a restart.
The exception is `dashboard_port`, which needs a restart.

| Key | Default | |
|---|---|---|
| `standard` | 15k steps, 1440 px, SH 3, ≤10M splats | The default preset. *Fast* and *High* are fixed. |
| `spz_version` | `3` | SPZ format version. Version 4, SplatTransform's default, isn't gzip-wrapped, and many apps load it as empty. Version 3 is the classic format that most apps support. |
| `compress` | `["spz"]` | Compressed formats written next to the `.ply`. Use `[]` to turn this off. If compression fails, the capture still counts as done, and the problem is noted in the history. |
| `refine_poses` | `false` (example config: `true`) | Refine camera poses with COLMAP before training. A few minutes of CPU per capture. |
| `init_feature_points` | `true` | Also start training from COLMAP's triangulated feature points (needs `refine_poses`). See [gsplat trainer and pruning](#gsplat-trainer-and-pruning-cuda-workers-optional). |
| `trainer` | `"brush"` | `"gsplat"` trains on CUDA workers that have it. Per-capture override on the dashboard. |
| `gsplat` | `{"densify": "mcmc", "depth_weight": 0.3, "flat_weight": 10, "max_splats": 6000000, "sky_points": 50000, "mode": "3dgs"}` | gsplat options. `max_splats` caps gsplat runs whatever the preset says (MCMC always fills its budget; GPU memory is the limit). `mode` can be `"2dgs"` (flat disks). |
| `depth_supervision` / `flatten` | `true` / `true` | Defaults for the per-capture gsplat checkboxes. |
| `prune_pct` | `0` (off) | Remove this % of splats after training. Per-capture override. |
| `prune` | `{"score": "lightgaussian", "finetune_steps": 3000}` | Pruning score (`lightgaussian` or `pup`) and fine-tuning length. |
| `flip_up` | `false` (example config: `true`) | Rotate results 180° about X so they're the right way up in SuperSplat and similar viewers. |
| `keep_previous_results` | `true` | When a requeued capture finishes, keep the earlier result in `previous/`. If `false`, it's overwritten. |
| `imessage_to` | `""` (off) | Phone number or Apple ID email to text. |
| `notify_checkpoints` | `[0.25, 0.5, 0.75]` | Fractions of training that trigger a progress text. |
| `settle_seconds` | `30` | How long an inbox item must stay unchanged before it's trained. |
| `checkpoint_every` / `eval_every` | `2500` / `500` | How often Brush saves a checkpoint / renders an evaluation image. Evaluation images drive the progress display. |
| `extra_args` | `[]` | Extra Brush CLI flags, e.g. `["--ssim-weight", "0.3"]`. |
| `brush` | `$GAUSSQUEUE_HOME/bin/brush_app` | Path to a Brush binary you built yourself. |

### iMessage notes

Set `imessage_to`, then run `scripts/notify-test.sh`. macOS will ask once for
permission to control Messages. Image attachments are staged in
`~/Pictures/GaussQueue` (`attach_dir`) because Messages refuses to attach files from
some other locations. If you text the same Apple ID that the Mac is signed in with, the
phone may receive the message silently. Texting a different handle, such as your phone
number, makes sure it alerts.

### Fixing results you already have

```bash
python3 gaussqueue.py --flip "<outbox>/<capture>" ["<outbox>/<capture2>" …]
```

This rotates the `.ply` in place, rebuilds the `.spz` from it, and toggles the
rotation recorded in `meta.json`. Running it twice restores the original orientation.

## Dashboard API

The dashboard listens on `127.0.0.1` only.

- `GET /api/status` returns a JSON snapshot: the current job, the queue with settings
  and estimates, history and presets.
- `POST /api/item` changes a queued item. Example bodies:
  `{"name": "...", "action": "preset", "value": "fast"}`,
  `{"name": "...", "action": "set", "key": "max_resolution", "value": 1920}`, and the
  actions `run_next`, `hold` (with `"value": true|false`) and `reset`. For a capture in
  *Recent*, `requeue` moves it back into the inbox, on hold. For the running
  capture, `pause` and `resume` are available, and settings can only be changed while it's paused.
  Requests must come from the dashboard's own origin and be JSON. That stops other
  websites open in your browser from changing your queue.

## Limitations

- Video or photo-only captures (no COLMAP model) are not trained yet. They would need
  frame extraction and COLMAP. Video zips get a thumbnail only if `ffmpeg` is installed.
- If the Mac restarts, the job that was running starts again from the beginning.
- Brush prints nothing in this mode, so progress comes from its evaluation renders
  (every `eval_every` steps).

## Development

```bash
python3 -m unittest discover -s tests   # dashboard API, presets, queue ordering
GAUSSQUEUE_HOME=/tmp/gq python3 gaussqueue.py --check
```

`gaussqueue.py` is a single file that uses only the standard library, and
`dashboard.html` is plain HTML/JS. The service reads the page from disk on every
request, so edits show up on reload.

## License

MIT. Brush is a separate project, licensed under Apache-2.0. GaussQueue downloads
it at install time and does not include it.
