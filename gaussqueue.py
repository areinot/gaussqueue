#!/usr/bin/env python3
"""GaussQueue: a folder-watching training queue for Gaussian splats on macOS.

Drop a capture (a zip or folder containing a COLMAP model, e.g. a SplatKing LiDAR
export) into the inbox folder. GaussQueue copies it to a local work folder,
trains it with Brush (https://github.com/ArthurBrussee/brush), and writes the
.ply plus a preview into the outbox folder. Progress is reported through a
localhost dashboard, a STATUS.txt file and (optionally) iMessage.

State and config live in $GAUSSQUEUE_HOME (default
~/Library/Application Support/GaussQueue). config.json is re-read on every poll,
so most edits apply without a restart. See README.md.
"""
import base64
import datetime as dt
import glob
import http.server
import math
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import zipfile
from pathlib import Path

APP_HOME = Path(os.environ.get("GAUSSQUEUE_HOME",
                               Path.home() / "Library/Application Support/GaussQueue")).expanduser()
CONFIG_PATH = APP_HOME / "config.json"
STATE_PATH = APP_HOME / "state.json"
OVERRIDES_PATH = APP_HOME / "overrides.json"
DASHBOARD_HTML = Path(__file__).resolve().parent / "dashboard.html"
GS_DIR = Path(__file__).resolve().parent / "gs"   # gsplat trainer/pruner, copied to CUDA workers
SF_DATALESS = 0x40000000   # st_flags bit for iCloud placeholder ("dataless") files
IMAGE_EXT = (".jpg", ".jpeg", ".png")
VIDEO_EXT = (".mov", ".mp4", ".m4v")

DEFAULTS = {
    "inbox_dir": None,                       # required: where captures arrive
    "outbox_dir": str(APP_HOME / "outbox"),  # finished splats only (safe to share/sync)
    "work_dir": str(APP_HOME / "work"),      # local scratch; keep it out of iCloud
    "status_dir": str(APP_HOME / "status"),  # STATUS.txt + preview JPGs
    # Optional: also copy the small, phone-friendly files of each result here
    # (e.g. an iCloud folder) while the full results stay in outbox_dir.
    "share_dir": None,
    "share_files": ["spz", "preview.jpg", "meta.json"],
    "attach_dir": "~/Pictures/GaussQueue",   # staging for iMessage image attachments
    "brush": str(APP_HOME / "bin/brush_app"),
    # Compressed copies written next to the .ply ([] = off)
    "compress": ["spz"],
    # SPZ versions written for each result (per-capture checkboxes), as <name>.v3.spz and
    # <name>.v4.spz. v3 is gzip-wrapped and opens nearly everywhere; v4 (zstd) is newer and
    # faster to load, but only recent readers support it (Niantic tools, PlayCanvas, Postshot).
    "spz_versions": [3],
    # Native PLY->SPZ/flip tool built from tools/spz-native (Niantic's spz library). Used for
    # .spz and for the flip when present: no WebAssembly 4 GB ceiling, much less memory.
    "spz_native": str(APP_HOME / "bin/gq_spz"),
    # Rotate results 180° about X so "up" is up in SuperSplat & co. (SplatKing/COLMAP
    # captures are +Y-up, most splat viewers expect -Y-up). Per-capture override on the dashboard.
    "flip_up": False,
    # Start view: move and turn each result so it opens where the first capture photo was
    # taken. The origin goes to that camera's position and the scene is turned about the
    # vertical axis (yaw only; pitch and roll stay as captured) so the photo's heading points
    # along start_view_forward, the viewer's default forward ("-z": OpenGL-style viewers, which
    # look down -Z from the origin; "+z" for viewers that look down +Z). Output-only, per capture.
    "start_view": True,
    "start_view_forward": "-z",
    # Refine the capture's camera poses with COLMAP (features, matching, bundle
    # adjustment) before training. Slower, usually sharper. Needs `colmap` on PATH.
    "refine_poses": False,
    "colmap": "colmap",
    # Also start training from the feature points COLMAP triangulates while refining poses,
    # not just the LiDAR points. LiDAR only reaches ~5 m, so without them distant trees,
    # buildings and sky have nothing to grow from and end up as blur near the camera.
    "init_feature_points": True,
    # Trainer: "brush" (runs anywhere) or "gsplat" (CUDA; only on remote workers that have
    # a "gsplat" entry, see workers below). Per-capture override on the dashboard.
    "trainer": "brush",
    # gsplat options. depth_supervision and flatten are per-capture toggles; the weights tune
    # them. densify: "mcmc" (fixed budget: always ends at the splat cap; best in our tests),
    # "default" (3DGS growth) or "absgrad" (grows more in fine detail). max_splats caps the
    # splat count for gsplat runs whatever the preset says (MCMC always fills its budget, and
    # GPU memory is the limit). sky_points: extra starting splats on a distant sphere for sky.
    # mode: "3dgs" or "2dgs" (flat disks). depth_fusion (experimental, off): before training,
    # merge every photo's LiDAR depth into one 3D model (closer, more confident readings count
    # more) and render it back into each photo for denoised, wider depth coverage. In tests it
    # made held-out depth worse in views facing foliage; training already fuses the per-photo
    # depth implicitly.
    # carve: remove splats in space the LiDAR saw through (and shorten ones poking into it);
    # removes floaters that show when you move off the capture path. far_flat_weight: make
    # splats beyond the scanned region camera-facing disks. Both use a map of the LiDAR's
    # space built on the worker and cached with the pose refinement.
    "gsplat": {"densify": "mcmc", "depth_weight": 0.3, "flat_weight": 10, "max_splats": 6_000_000,
               "sky_points": 50000, "mode": "3dgs", "depth_fusion": False,
               "carve": True, "far_flat_weight": 1.0},
    "depth_supervision": True,   # gsplat: match LiDAR depth where ARKit is confident
    "flatten": True,             # gsplat: push splats flat onto surfaces
    # Post-training pruning (on a CUDA worker): remove this percentage of splats, ranked by
    # how much they matter to the photos, then fine-tune the rest. 0 = off. Per-capture.
    "prune_pct": 0,
    "prune": {"score": "lightgaussian", "finetune_steps": 3000},
    "splat_transform": str(APP_HOME / "tools/splat-transform/node_modules/.bin/splat-transform"),
    "imessage_to": "",                       # phone number or Apple ID email; empty = off
    "notify_checkpoints": [0.25, 0.5, 0.75], # fractions of training to text a preview at
    "poll_seconds": 5,
    "settle_seconds": 30,         # inbox item must be unchanged this long before training
    "dashboard_port": 8765,
    "standard": {                 # the default quality preset
        "total_steps": 15000, "max_resolution": 1440, "sh_degree": 3,
        "max_splats": 10_000_000, "subsample_frames": 1,
    },
    "checkpoint_every": 2500,
    "eval_every": 500,
    "eval_split_every": 16,
    "extra_args": [],
    "keep_work_days": 7,
    # When a capture is requeued and rerun, move the earlier result into
    # <outbox>/<name>/previous/ instead of overwriting it.
    "keep_previous_results": True,
    # "Publish snapshot" turns the latest checkpoint of a running capture into a result.
    # While something is training, snapshots are only processed when at least this much
    # memory is free (they wait for the run to finish otherwise).
    "snapshot_min_free_pct": 25,
    # Training machines. The Mac trains locally unless local_training is false; remote
    # workers are Windows/Linux PCs reachable over SSH (key auth; host = an ~/.ssh/config
    # alias), e.g. {"name": "pc", "label": "PC (RTX 4080)", "ssh": "gpu-pc",
    #  "brush": "D:/gaussqueue/brush/brush_app.exe", "work_dir": "D:/gaussqueue/jobs",
    #  "gsplat": {"python": "D:/gaussqueue/gsplat/venv/Scripts/python.exe",
    #             "scripts": "D:/gaussqueue/gsplat/gs"}}   <- optional, enables gsplat + pruning
    # "Auto" captures go to an idle remote worker first, then the Mac.
    "local_training": True,
    "workers": [],
}


# Per-item processing settings, editable from the dashboard. "standard" comes
# from config.json, so editing config still changes the default.
TRAIN_KEYS = ("total_steps", "max_resolution", "sh_degree", "max_splats", "subsample_frames")
GSPLAT_KEYS = ("trainer", "densify", "depth_supervision", "flatten")
RESTART_KEYS = TRAIN_KEYS + ("refine_poses",) + GSPLAT_KEYS   # changing these restarts training
OUTPUT_KEYS = ("flip_up", "start_view", "prune_pct", "spz_v3", "spz_v4")   # only affect the output files
SETTING_KEYS = RESTART_KEYS + OUTPUT_KEYS
SETTING_LIMITS = {"total_steps": (500, 100_000), "max_resolution": (256, 4096),
                  "sh_degree": (0, 3), "max_splats": (50_000, 20_000_000),
                  "subsample_frames": (1, 20), "flip_up": (0, 1), "refine_poses": (0, 1),
                  "depth_supervision": (0, 1), "flatten": (0, 1), "prune_pct": (0, 95),
                  "start_view": (0, 1), "spz_v3": (0, 1), "spz_v4": (0, 1)}
SETTING_CHOICES = {"trainer": ("brush", "gsplat"), "densify": ("mcmc", "default", "absgrad")}
PRESETS = {
    # Quick look: is the scene coherent, oriented right, covered? Not for judging detail.
    "preview": {"total_steps": 3000, "max_resolution": 720, "sh_degree": 1,
                "max_splats": 1_000_000, "subsample_frames": 1},
    "fast": {"total_steps": 7000, "max_resolution": 1080, "sh_degree": 2,
             "max_splats": 1_500_000, "subsample_frames": 1},
    "standard": None,  # from config["standard"]
    "high": {"total_steps": 30000, "max_resolution": 1920, "sh_degree": 3,
             "max_splats": 10_000_000, "subsample_frames": 1},
}
def standard_settings(cfg):
    st = dict(DEFAULTS["standard"])
    st.update(cfg.get("standard") or {})
    return {k: int(st[k]) for k in TRAIN_KEYS}


def effective_settings(cfg, ov):
    """Settings a job will run with: preset (default standard) + field overrides."""
    preset = ov.get("preset") or "standard"
    base = PRESETS.get(preset) or standard_settings(cfg)
    out = dict(base)
    out["flip_up"] = int(bool(cfg.get("flip_up")))
    out["start_view"] = int(bool(cfg.get("start_view", True)))
    for v in (3, 4):
        out[f"spz_v{v}"] = int(v in (cfg.get("spz_versions") or []))
    out["refine_poses"] = int(bool(cfg.get("refine_poses")))
    out["trainer"] = cfg.get("trainer") if cfg.get("trainer") in SETTING_CHOICES["trainer"] else "brush"
    densify = sub_config(cfg, "gsplat")["densify"]
    out["densify"] = densify if densify in SETTING_CHOICES["densify"] else "mcmc"
    out["depth_supervision"] = int(bool(cfg.get("depth_supervision", True)))
    out["flatten"] = int(bool(cfg.get("flatten", True)))
    out["prune_pct"] = int(cfg.get("prune_pct") or 0)
    out.update({k: ov[k] for k in SETTING_KEYS if k in ov})
    out["base"] = preset
    out["preset"] = preset if not any(k in ov for k in TRAIN_KEYS) else "custom"
    return out


def sub_config(cfg, key):
    """A nested config section with defaults filled in (config.json may set only some keys)."""
    return {**DEFAULTS[key], **(cfg.get(key) or {})}


def settings_summary(st):
    gs = st.get("trainer") == "gsplat"
    return (f"{st['preset']}: {st['total_steps'] / 1000:g}k steps @ {st['max_resolution']}px, "
            f"SH{st['sh_degree']}, <={st['max_splats'] / 1e6:g}M splats"
            + (f", every {st['subsample_frames']} frames" if st["subsample_frames"] > 1 else "")
            + (", refined poses" if st.get("refine_poses") else "")
            + (f", gsplat {st.get('densify')}" + (" +depth" if st.get("depth_supervision") else "")
               + (" +flat" if st.get("flatten") else "") if gs else "")
            + (f", prune {st['prune_pct']}%" if st.get("prune_pct") else "")
            + (", flipped" if st.get("flip_up") else "")
            + (", starts at first photo" if st.get("start_view") else "")
            + (f", spz {'+'.join(f'v{v}' for v in spz_versions(st)) or 'off'}" if spz_versions(st) != [3] else ""))


class Overrides:
    """Per-item settings + queue controls (hold, run-next), persisted to disk."""

    def __init__(self):
        self.lock = threading.Lock()
        try:
            self.data = json.loads(OVERRIDES_PATH.read_text())
        except Exception:
            self.data = {}

    def get(self, name):
        with self.lock:
            return dict(self.data.get(name, {}))

    def _save(self):
        tmp = OVERRIDES_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(OVERRIDES_PATH)

    def apply(self, name, body):
        """Validate and apply a change from the dashboard. Raises ValueError."""
        with self.lock:
            ov = self.data.setdefault(name, {})
            action = body.get("action")
            if action == "reset":
                for k in ("preset",) + TRAIN_KEYS:
                    ov.pop(k, None)
            elif action == "run_next":
                ov["priority"] = time.time()
                ov.pop("hold", None)
            elif action == "hold":
                ov["hold"] = bool(body.get("value", True))
                if ov["hold"]:
                    ov.pop("priority", None)
            elif action == "preset":
                if body.get("value") not in PRESETS:
                    raise ValueError("unknown preset")
                for k in TRAIN_KEYS:
                    ov.pop(k, None)
                ov["preset"] = body["value"]
            elif action == "run_on":   # validated against known workers by the API
                if body.get("value") in (None, "", "auto"):
                    ov.pop("run_on", None)
                else:
                    ov["run_on"] = str(body["value"])
            elif action == "set" and body.get("key") in SETTING_CHOICES:
                key, val = body["key"], body.get("value")
                if val not in SETTING_CHOICES[key]:
                    raise ValueError(f"{key} must be one of {', '.join(SETTING_CHOICES[key])}")
                ov[key] = val
            elif action == "set":
                key, val = body.get("key"), body.get("value")
                if key not in SETTING_LIMITS:
                    raise ValueError("unknown setting")
                lo, hi = SETTING_LIMITS[key]
                try:
                    val = int(val)
                except (TypeError, ValueError):
                    raise ValueError(f"{key} must be a whole number")
                if not lo <= val <= hi:
                    raise ValueError(f"{key} must be between {lo:,} and {hi:,}")
                ov[key] = val
            else:
                raise ValueError("unknown action")
            if not ov:
                self.data.pop(name, None)
            self._save()

    def clear(self, name):
        with self.lock:
            if self.data.pop(name, None) is not None:
                self._save()


def log(msg):
    print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


class ConfigError(Exception):
    pass


_last_good_config = None


def load_config():
    """Defaults + config.json. A broken edit keeps the last good config running."""
    global _last_good_config
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(CONFIG_PATH.read_text()))
    except FileNotFoundError:
        if _last_good_config is not None:
            log(f"config.json missing, keeping previous config")
            return _last_good_config
        raise ConfigError(f"no config at {CONFIG_PATH}; run scripts/install.sh "
                          "or copy config.example.json there")
    except Exception as e:
        if _last_good_config is None:
            raise ConfigError(f"config.json unreadable: {e}")
        log(f"config.json unreadable, keeping previous config: {e}")
        return _last_good_config
    if not cfg.get("inbox_dir"):
        if _last_good_config is not None:
            log("config.json has no inbox_dir, keeping previous config")
            return _last_good_config
        raise ConfigError("config.json must set inbox_dir")
    for k in ("inbox_dir", "outbox_dir", "work_dir", "status_dir", "attach_dir", "brush",
              "splat_transform", "share_dir", "spz_native"):
        if cfg.get(k):
            cfg[k] = str(Path(cfg[k]).expanduser())
    _last_good_config = cfg
    return cfg


def load_state():
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {"history": []}


STATE_LOCK = threading.RLock()


def save_state(state):
    with STATE_LOCK:
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(STATE_PATH)


def fmt_dur(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def item_name(path):
    name = path.name
    while name.lower().endswith(".zip"):
        name = name[:-4]
    return name


def find_processed(paths, name):
    """The original zip/folder of a finished capture, in inbox/done or inbox/failed."""
    for folder in (paths.done, paths.failed):
        try:
            for p in folder.iterdir():
                if not p.name.startswith(".") and item_name(p) == name:
                    return p
        except FileNotFoundError:
            pass
    return None


class Paths:
    def __init__(self, cfg):
        self.inbox = Path(cfg["inbox_dir"])
        self.done = self.inbox / "done"
        self.failed = self.inbox / "failed"
        self.status = Path(cfg["status_dir"])
        self.previews = self.status / "previews"
        self.outbox = Path(cfg["outbox_dir"])
        self.jobs = Path(cfg["work_dir"])
        self.tmp = APP_HOME / "tmp"
        self.attach = Path(cfg["attach_dir"])  # Messages reliably attaches from ~/Pictures
        for p in (self.inbox, self.done, self.failed, self.status, self.previews,
                  self.outbox, self.jobs, self.tmp, self.attach):
            p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- notifications

def imessage(cfg, text, image=None):
    to = cfg.get("imessage_to", "").strip()
    if not to:
        return
    script = [
        "on run argv",
        'tell application "Messages"',
        "set svc to 1st account whose service type = iMessage",
        "set theTarget to participant (item 2 of argv) of svc",
        "send (item 1 of argv) to theTarget",
        "if (count of argv) > 2 then send (POSIX file (item 3 of argv)) to theTarget",
        "end tell",
        "end run",
    ]
    cmd = ["osascript"]
    for line in script:
        cmd += ["-e", line]
    cmd += [text, to] + ([str(image)] if image else [])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode:
            log(f"iMessage failed: {r.stderr.strip()}")
    except Exception as e:
        log(f"iMessage failed: {e}")


# ---------------------------------------------------------------- iCloud helpers

def is_dataless(p):
    try:
        return bool(os.stat(p).st_flags & SF_DATALESS)
    except FileNotFoundError:
        return True


def request_download(p):
    """Ask iCloud to materialise a file or folder; returns immediately."""
    subprocess.Popen(["brctl", "download", str(p)],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def dataless_files(folder):
    out = []
    for dirpath, _, files in os.walk(folder):
        for f in files:
            if is_dataless(os.path.join(dirpath, f)):
                out.append(f)
    return out


def make_jpeg(src, dst, size, rotate=0):
    cmd = ["sips", "-s", "format", "jpeg", "-s", "formatOptions", "80", "-Z", str(size)]
    if rotate:
        cmd += ["-r", str(rotate)]
    subprocess.run(cmd + [str(src), "--out", str(dst)], capture_output=True, check=True)


# SplatKing stores frames in sensor orientation; photo_series.json records how the
# phone was held as an EXIF orientation. Map it to sips' clockwise rotation.
EXIF_ROTATION = {1: 0, 3: 180, 6: 90, 8: 270}


def rotation_from_series(text):
    m = re.search(r'"cgImagePropertyOrientation"\s*:\s*(\d)', text)
    return EXIF_ROTATION.get(int(m.group(1)), 0) if m else 0


def capture_rotation(root):
    """Rotation for a capture folder (searches for photo_series.json)."""
    for f in Path(root).rglob("photo_series.json"):
        try:
            return rotation_from_series(f.read_text(errors="replace"))
        except OSError:
            return 0
    return 0


# ---------------------------------------------------------------- inbox watcher

def summarize_splatpack(data):
    """SplatKing's per-photo capture ratings (splatpack.json) -> a compact summary:
    median score, and how many photos fell into each rating (e.g. "large pose
    jump" = the phone moved too far between photos, i.e. low overlap)."""
    pairs = data.get("pairs") or []
    if not pairs:
        return None
    counts, scores = {}, []
    for pr in pairs:
        q = pr.get("quality") or {}
        label = q.get("summary") or "unrated"
        band = q.get("band") or "?"
        key = f"{band}|{label}"
        counts[key] = counts.get(key, 0) + 1
        if isinstance(q.get("score"), (int, float)):
            scores.append(q["score"])
    n = len(pairs)
    ratings = [{"band": k.split("|", 1)[0], "label": k.split("|", 1)[1], "count": c,
                "pct": round(100 * c / n)} for k, c in sorted(counts.items(), key=lambda kv: -kv[1])]
    jump = sum(r["count"] for r in ratings if "pose jump" in r["label"])
    scores.sort()
    return {"photos": n, "score": scores[len(scores) // 2] if scores else None,
            "ratings": ratings, "pose_jump_pct": round(100 * jump / n),
            "low_overlap": jump / n >= 0.5}


def quality_text(q):
    if not q:
        return ""
    top = ", ".join(f"{r['pct']}% {r['label']}" for r in q["ratings"][:3])
    return (("⚠ low overlap. " if q["low_overlap"] else "")
            + f"SplatKing score {q['score']}: {top}")


def find_capture_quality(root):
    for f in Path(root).rglob("splatpack.json"):
        try:
            return summarize_splatpack(json.loads(f.read_text()))
        except Exception:
            return None
    return None


class InboxItem:
    def __init__(self, path):
        self.path = path
        self.name = item_name(path)
        self.is_dir = path.is_dir()
        # When it landed in the inbox (survives restarts, so queue order is stable).
        try:
            self.first_seen = min(path.stat().st_birthtime, time.time())
        except (AttributeError, OSError):
            self.first_seen = time.time()
        self.sig = None
        self.sig_since = time.time()
        self.state = "arriving"
        self.kind = None      # lidar | photos | video | unknown
        self.frames = None
        self.size = 0
        self.thumb = None     # filename in previews/
        self.thumb_note = None
        self.thumb_tries = 0
        self.quality = None   # SplatKing capture ratings, if present
        self.photo_px = None  # longest side of the capture's photos (from cameras.txt)
        self.quality_tries = 0

    def signature(self):
        if not self.is_dir:
            st = self.path.stat()
            self.size = st.st_size
            return (st.st_size, st.st_mtime)
        count = total = 0
        newest = 0.0
        for dirpath, _, files in os.walk(self.path):
            for f in files:
                try:
                    st = os.stat(os.path.join(dirpath, f))
                except FileNotFoundError:
                    continue
                count += 1
                total += st.st_size
                newest = max(newest, st.st_mtime)
        self.size = total
        return (count, total, newest)

    def to_json(self):
        return {"name": self.name, "file": self.path.name, "is_dir": self.is_dir,
                "state": self.state, "kind": self.kind, "frames": self.frames,
                "size": self.size, "first_seen": self.first_seen,
                "thumb": self.thumb, "thumb_note": self.thumb_note, "quality": self.quality,
                "photo_px": self.photo_px}


class Inbox:
    """Tracks inbox items, decides when they're settled, and makes thumbnails."""

    def __init__(self, paths):
        self.paths = paths
        self.items = {}   # path name -> InboxItem
        self.lock = threading.Lock()

    def entries(self):
        out = []
        for p in self.paths.inbox.iterdir():
            if p.name.startswith(".") or p in (self.paths.done, self.paths.failed):
                continue
            if p.is_dir() or p.name.lower().endswith(".zip"):
                out.append(p)
        return out

    def tick(self, cfg, busy=()):
        present = {p.name: p for p in self.entries()}
        with self.lock:
            for gone in set(self.items) - set(present):
                del self.items[gone]
            for fname, p in present.items():
                if fname not in self.items:
                    self.items[fname] = InboxItem(p)
                    log(f"inbox: new item {fname}")
        for item in list(self.items.values()):
            if item.name in busy:
                item.state = "processing"
                continue
            try:
                sig = item.signature()
            except FileNotFoundError:
                continue
            if sig != item.sig:
                item.sig, item.sig_since = sig, time.time()
            settled = time.time() - item.sig_since >= cfg["settle_seconds"]
            item.state = "queued" if settled else "arriving"
            if item.thumb is None and item.thumb_tries < 20:
                self.try_thumbnail(item)
            if item.quality is None and item.quality_tries < 20:
                self.read_quality(item)

    def order(self, overrides):
        """Queue order: 'run next' items first (latest press first), then oldest."""
        def key(i):
            return (-overrides.get(i.name).get("priority", 0), i.first_seen)
        return sorted(self.items.values(), key=key)

    def ready(self, overrides):
        """Settled, not-held items in queue order."""
        return [i for i in self.order(overrides)
                if i.state == "queued" and not overrides.get(i.name).get("hold")]

    def snapshot(self, cfg, overrides, rate, busy=(), estimates=None):
        out = []
        for i in self.order(overrides):
            if i.name in busy:
                continue
            ov = overrides.get(i.name)
            d = i.to_json()
            d["settings"] = effective_settings(cfg, ov)
            d["hold"] = bool(ov.get("hold"))
            d["run_next"] = bool(ov.get("priority"))
            d["est_seconds"] = estimate_seconds(d["settings"], d["frames"], rate)
            d["run_on"] = ov.get("run_on", "auto")
            if estimates:
                d["est"] = estimates(d["settings"], d["frames"], ov)
                known = [v for v in d["est"].values() if v]
                if known:
                    d["est_seconds"] = min(known)
            out.append(d)
        return out

    # -- thumbnails: the cheapest possible look at the data
    def try_thumbnail(self, item):
        dst = self.paths.previews / f"{item.name}_thumb.jpg"
        if dst.exists():
            item.thumb = dst.name
            self.describe(item)
            return
        try:
            if item.is_dir:
                ok = self.thumb_from_dir(item, dst)
            else:
                ok = self.thumb_from_zip(item, dst)
            if ok:
                item.thumb, item.thumb_note = dst.name, None
                log(f"thumbnail: {dst.name}")
        except Exception as e:
            item.thumb_tries += 1
            item.thumb_note = f"thumbnail failed: {e}"[:120]

    def read_quality(self, item):
        """Read SplatKing's splatpack.json straight from the zip/folder (small file)."""
        item.quality_tries += 1
        try:
            self.read_photo_size(item)
            if item.is_dir:
                f = next(item.path.rglob("splatpack.json"), None)
                if f is None:
                    item.quality_tries = 99
                    return
                if is_dataless(f):
                    request_download(f)
                    return
                data = json.loads(f.read_text())
            else:
                if is_dataless(item.path):
                    return   # try_thumbnail already asked iCloud for it
                with zipfile.ZipFile(item.path) as z:
                    names = [n for n in z.namelist() if n.endswith("splatpack.json")]
                    if not names:
                        item.quality_tries = 99
                        return
                    data = json.loads(z.read(names[0]))
            item.quality = summarize_splatpack(data)
            if item.quality is None:
                item.quality_tries = 99
        except (zipfile.BadZipFile, OSError, ValueError):
            pass

    @staticmethod
    def photo_size_from(text):
        for line in text.splitlines():
            p = line.split()
            if line.startswith("#") or len(p) < 4 or not p[0].isdigit():
                continue
            return max(int(p[2]), int(p[3]))
        return None

    def read_photo_size(self, item):
        if item.photo_px:
            return
        if item.is_dir:
            f = next(item.path.rglob("sparse/0/cameras.txt"), None)
            if f is None or is_dataless(f):
                if f is not None:
                    request_download(f)
                return
            item.photo_px = self.photo_size_from(f.read_text())
        elif not is_dataless(item.path):
            with zipfile.ZipFile(item.path) as z:
                names = [n for n in z.namelist() if n.endswith("sparse/0/cameras.txt")]
                if names:
                    item.photo_px = self.photo_size_from(z.read(names[0]).decode(errors="replace"))

    def describe(self, item):
        if item.kind is not None:
            return
        try:
            if item.is_dir:
                names = [str(p.relative_to(item.path)) for p in item.path.rglob("*")]
            else:
                if is_dataless(item.path):
                    return
                with zipfile.ZipFile(item.path) as z:
                    names = z.namelist()
            self.classify(item, names)
        except Exception:
            pass

    @staticmethod
    def classify(item, names):
        imgs = [n for n in names if n.lower().endswith(IMAGE_EXT)
                and "/images/" in "/" + n and not Path(n).name.startswith(".")]
        if any("sparse/0/" in n for n in names):
            item.kind = "lidar"
        elif imgs:
            item.kind = "photos"
        elif any(n.lower().endswith(VIDEO_EXT) for n in names):
            item.kind = "video"
        else:
            item.kind = "unknown"
        item.frames = len(imgs) or None
        return sorted(imgs)

    def thumb_from_dir(self, item, dst):
        names = [str(p.relative_to(item.path)) for p in item.path.rglob("*")]
        imgs = self.classify(item, names)
        if not imgs:
            vids = [n for n in names if n.lower().endswith(VIDEO_EXT)]
            if vids:
                return self.thumb_from_video_file(item, item.path / vids[0], dst)
            item.thumb_note = "no images found"
            item.thumb_tries = 99
            return False
        src = item.path / imgs[len(imgs) // 2]
        if is_dataless(src):
            request_download(src)
            item.thumb_note = "fetching one frame from iCloud…"
            return False
        series = item.path / "photo_series.json"
        rotate = 0
        if series.exists() and not is_dataless(series):
            rotate = rotation_from_series(series.read_text(errors="replace"))
        make_jpeg(src, dst, 640, rotate)
        return True

    def thumb_from_zip(self, item, dst):
        if is_dataless(item.path):
            request_download(item.path)
            item.thumb_note = "downloading zip from iCloud…"
            return False
        try:
            z = zipfile.ZipFile(item.path)
        except zipfile.BadZipFile:
            item.thumb_note = "zip still arriving…"
            return False
        with z:
            imgs = self.classify(item, z.namelist())
            if imgs:
                series = [n for n in z.namelist() if n.endswith("photo_series.json")]
                rotate = (rotation_from_series(z.read(series[0]).decode(errors="replace"))
                          if series else 0)
                tmp = self.paths.tmp / f"{item.name}_thumbsrc{Path(imgs[0]).suffix}"
                tmp.write_bytes(z.read(imgs[len(imgs) // 2]))
                try:
                    make_jpeg(tmp, dst, 640, rotate)
                finally:
                    tmp.unlink(missing_ok=True)
                return True
            vids = [i for i in z.infolist() if i.filename.lower().endswith(VIDEO_EXT)]
            if vids:
                vid = next((v for v in vids if "wide" in Path(v.filename).name
                            and "ultra" not in v.filename), vids[0])
                return self.thumb_from_zipped_video(item, z, vid, dst)
        item.thumb_note = "no images found"
        item.thumb_tries = 99
        return False

    def thumb_from_video_file(self, item, video, dst):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            item.thumb_note = "video: install ffmpeg for thumbnails"
            item.thumb_tries = 99
            return False
        if is_dataless(video):
            request_download(video)
            item.thumb_note = "fetching video from iCloud…"
            return False
        subprocess.run([ffmpeg, "-v", "error", "-ss", "1", "-i", str(video), "-frames:v", "1",
                        "-vf", "scale=640:-2", "-y", str(dst)], check=True, timeout=60)
        return dst.exists()

    def thumb_from_zipped_video(self, item, z, info, dst):
        """Grab one frame from a video inside a zip without extracting it: if
        the member is stored (uncompressed), ffmpeg can read it in place."""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            item.thumb_note = "video: install ffmpeg for thumbnails"
            item.thumb_tries = 99
            return False
        if info.compress_type != zipfile.ZIP_STORED:
            item.thumb_note = "video is compressed inside the zip; no quick thumbnail"
            item.thumb_tries = 99
            return False
        z.fp.seek(info.header_offset)
        header = z.fp.read(30)
        name_len = int.from_bytes(header[26:28], "little")
        extra_len = int.from_bytes(header[28:30], "little")
        offset = info.header_offset + 30 + name_len + extra_len
        subprocess.run([ffmpeg, "-v", "error", "-skip_initial_bytes", str(offset),
                        "-ss", "1", "-i", str(item.path), "-frames:v", "1",
                        "-vf", "scale=640:-2", "-y", str(dst)], check=True, timeout=60)
        return dst.exists()


# ---------------------------------------------------------------- status

class Status:
    def __init__(self, paths, state, inbox, overrides):
        self.paths, self.state, self.inbox, self.overrides = paths, state, inbox, overrides
        self.jobs = {}          # worker name -> job dict (one job per worker)
        self.workers = {}       # worker name -> Worker, set in main()
        self.quality_cache = {}
        self.snapshots = None   # set in main()
        self.snapshot = {}    # served as JSON by the dashboard

    # Compatibility: "the" current job is the Mac's, else any running one.
    @property
    def current(self):
        return self.jobs.get("mac") or next(iter(list(self.jobs.values())), None)

    @current.setter
    def current(self, job):
        if job is None:
            self.jobs.pop("mac", None)
        else:
            self.jobs["mac"] = job

    def job_for(self, name):
        return next((j for j in list(self.jobs.values()) if j["name"] == name), None)

    def busy_names(self):
        return {j["name"] for j in list(self.jobs.values())}

    def job_view(self, cfg, c):
        v = {k: c.get(k) for k in
             ("name", "phase", "step", "total", "started", "eta", "preview", "kind", "frames",
              "thumb", "settings", "paused", "paused_at", "quality", "photo_px", "worker")}
        w = self.workers.get(c.get("worker"))
        v["worker_label"] = w.label if w else c.get("worker")
        v["remote"] = bool(w and w.remote)
        if self.snapshots:
            v["snapshots"] = self.snapshots.for_job(c["name"])
            for sn in v["snapshots"]:   # published snapshots are results of their own
                sn["result"] = f"{sn['name']}-step{sn['step']}"
                sn["downloads"] = result_downloads(self.paths, sn["result"]) if sn.get("state") == "done" else []
        v["pending_settings"] = effective_settings(cfg, self.overrides.get(c["name"]))
        v["previews"] = sorted(p.name for p in self.paths.previews.glob(f"{c['name']}_step*.jpg"))
        return v

    def write(self, cfg):
        c = self.current
        busy = self.busy_names()
        rate = current_rate(c, self.state)
        workers = list(self.workers.values()) or [LocalWorker()]
        any_gsplat = any(w.gsplat for w in workers)

        def estimates(st, frames, ov):
            """{machine: seconds or None} for the machines that could run this capture:
            all eligible ones on Auto, else just the chosen one."""
            out = {}
            for w in workers:
                if not can_take(cfg, w, ov, any_gsplat):
                    continue
                trainer = "gsplat" if st.get("trainer") == "gsplat" and w.gsplat else "brush"
                r = current_rate(self.jobs.get(w.name), self.state, w.name, trainer)
                out[w.name] = estimate_seconds(st, frames, r)
            return out
        queue = self.inbox.snapshot(cfg, self.overrides, rate, busy, estimates)
        hist, seen = [], set()
        in_inbox = {i.name for i in list(self.inbox.items.values())}
        shown = [h for h in self.state.get("history", []) if not h.get("dismissed")]
        for h in shown[-20:][::-1]:
            h = dict(h)
            h["id"] = history_id(h)
            if "quality" not in h:
                h["quality"] = self.quality_cache.get(h["name"], "?")
                if h["quality"] == "?":
                    h["quality"] = self.quality_cache[h["name"]] = find_capture_quality(
                        self.paths.jobs / h["name"] / "data")
            h["downloads"] = (result_downloads(self.paths, h["name"])
                              if h.get("result") in ("done", "snapshot") else [])
            h["requeueable"] = (h["name"] not in seen and h["name"] not in in_inbox
                                and find_processed(self.paths, h["name"]) is not None)
            seen.add(h["name"])
            hist.append(h)
        order = list(self.workers) or list(self.jobs)
        jobs = [self.job_view(cfg, self.jobs[w]) for w in order if w in self.jobs]
        jobs += [self.job_view(cfg, j) for w, j in list(self.jobs.items()) if w not in order]
        current = jobs[0] if jobs else None
        self.snapshot = {
            "updated": time.time(), "host": socket.gethostname(),
            "current": current, "jobs": jobs, "queue": queue, "history": hist,
            "workers": [w.info(self.jobs.get(n)) for n, w in list(self.workers.items())],
            "settings": standard_settings(cfg),
            "inbox_dir": cfg["inbox_dir"],
            "presets": {k: (v or standard_settings(cfg)) for k, v in PRESETS.items()},
            "limits": SETTING_LIMITS,
            "choices": SETTING_CHOICES,
        }
        self.write_text(jobs, queue, hist)

    def write_text(self, jobs, queue, hist):
        now = dt.datetime.now()
        lines = [
            "GaussQueue status",
            f"Updated: {now:%a %b %d %H:%M:%S} on {socket.gethostname()}",
            "(If this time is old, the Mac service is not running.)",
            "",
        ]
        for c in jobs:
            lines.append(f"NOW on {c.get('worker_label') or 'Mac'}: {c['name']}")
            lines.append(f"  phase:   {c['phase']}")
            if c.get("settings"):
                lines.append(f"  quality: {settings_summary(c['settings'])}")
            if c.get("step") is not None:
                pct = 100 * c["step"] / c["total"]
                lines.append(f"  step:    {c['step']:,} / {c['total']:,} ({pct:.0f}%)")
            if c.get("started"):
                lines.append(f"  elapsed: {fmt_dur(time.time() - c['started'])}")
            if c.get("eta"):
                lines.append(f"  ETA:     ~{dt.datetime.fromtimestamp(c['eta']):%H:%M}"
                             f" ({fmt_dur(c['eta'] - time.time())} left)")
            if c.get("preview"):
                lines.append(f"  preview: previews/{c['preview']}")
        if not jobs:
            lines.append("NOW: idle")
        lines.append("")
        lines.append(f"QUEUE ({len(queue)}):")
        lines += [f"  - {q['file']}  [{'on hold' if q['hold'] else q['state']}]\n"
                  f"      {settings_summary(q['settings'])}"
                  + (f", ~{fmt_dur(q['est_seconds'])}" if q.get("est_seconds") else "")
                  for q in queue] or ["  (empty)"]
        lines.append("")
        lines.append("RECENT:")
        for h in hist[:10]:
            lines.append(f"  {h['finished']}  {h['result']:<6} {h['name']}  ({h['duration']})"
                         + (f"  {h['detail']}" if h.get("detail") else ""))
        if not hist:
            lines.append("  (none)")
        out = self.paths.status / "STATUS.txt"
        tmp = self.paths.status / f".STATUS.{threading.get_ident()}.tmp"  # writers: main + http threads
        tmp.write_text("\n".join(lines) + "\n")
        tmp.replace(out)


# ---------------------------------------------------------------- dashboard server

def requeue(status, overrides, name):
    """Move a finished capture back into the inbox, on hold so its settings can be
    changed before it runs again."""
    paths = status.inbox.paths
    if any(i.name == name for i in list(status.inbox.items.values())):
        raise ValueError("already in the inbox")
    src = find_processed(paths, name)
    if src is None:
        raise ValueError("original capture not found in inbox/done or inbox/failed")
    dst = paths.inbox / src.name
    if dst.exists():
        raise ValueError(f"{src.name} already exists in the inbox")
    overrides.apply(name, {"action": "hold", "value": True})  # before it becomes visible
    shutil.move(str(src), dst)
    log(f"requeued (on hold): {name}")


def start_dashboard(cfg, paths, status, overrides):
    previews = paths.previews
    port = int(cfg["dashboard_port"])
    allowed_origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}

    class Handler(http.server.BaseHTTPRequestHandler):
        def send(self, code, body, ctype, cache="no-store"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path.startswith("/download/"):
                return self.download(path)
            if path in ("/", "/index.html"):
                self.send(200, DASHBOARD_HTML.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/status":
                self.send(200, json.dumps(status.snapshot).encode(), "application/json")
            elif path.startswith("/img/"):
                name = path[len("/img/"):]
                f = previews / name
                if re.fullmatch(r"[\w.\-]+\.jpg", name) and f.is_file():
                    self.send(200, f.read_bytes(), "image/jpeg", "max-age=60")
                else:
                    self.send(404, b"not found", "text/plain")
            else:
                self.send(404, b"not found", "text/plain")

        def download(self, path):
            """Stream a result file as an attachment. Only to our own host name, so a page
            from elsewhere can't read results through DNS rebinding."""
            host = (self.headers.get("Host") or "").split(":")[0]
            if host not in ("127.0.0.1", "localhost"):
                return self.send(403, b"forbidden", "text/plain")
            parts = [urllib.parse.unquote(x) for x in path[len("/download/"):].split("/")]
            f = download_path(paths, *parts) if len(parts) == 2 else None
            if f is None:
                return self.send(404, b"not found", "text/plain")
            size = f.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition", f'attachment; filename="{f.name}"')
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                with open(f, "rb") as fh:
                    shutil.copyfileobj(fh, self.wfile, 1 << 20)
            except (BrokenPipeError, ConnectionResetError):
                pass   # the browser cancelled

        def do_POST(self):
            # Only the dashboard itself may change settings: require our own Origin
            # and a JSON body (cross-site pages can't send either without CORS approval).
            if (self.headers.get("Origin") not in allowed_origins
                    or self.headers.get("Content-Type") != "application/json"):
                return self.send(403, b"forbidden", "text/plain")
            if self.path != "/api/item":
                return self.send(404, b"not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(min(n, 10_000)))
                name = body.get("name", "")
                if body.get("action") == "requeue":
                    requeue(status, overrides, name)
                    status.write(load_config())
                    return self.send(200, b'{"ok":true}', "application/json")
                if body.get("action") == "dismiss":
                    dismiss_history(status.state, str(body.get("id", "")))
                    status.write(load_config())
                    return self.send(200, b'{"ok":true}', "application/json")
                if not any(i.name == name for i in list(status.inbox.items.values())):
                    raise ValueError("no such inbox item")
                cur = status.job_for(name)
                action = body.get("action")
                if action == "run_on":
                    if body.get("value") not in ["auto"] + list(status.workers):
                        raise ValueError("unknown machine")
                    if cur:
                        raise ValueError("already processing")
                    overrides.apply(name, body)
                    status.write(load_config())
                    return self.send(200, b'{"ok":true}', "application/json")
                remote = bool(cur and status.workers.get(cur.get("worker"))
                              and status.workers[cur.get("worker")].remote)
                if cur:
                    if action == "snapshot":
                        if cur.get("phase") not in ("training", "paused"):
                            raise ValueError("snapshots are only possible while training")
                        step = status.snapshots.request(cur)
                        return self.send(200, json.dumps({"ok": True, "step": step}).encode(),
                                         "application/json")
                    if action in ("pause", "resume"):
                        if remote:
                            raise ValueError("pause isn't available on remote machines yet")
                        if cur.get("phase") not in ("training", "paused"):
                            raise ValueError("can only pause while training")
                        if action == "resume" and not cur.get("paused"):
                            raise ValueError("not paused")
                        cur["request"] = action
                        return self.send(200, b'{"ok":true}', "application/json")
                    if action in ("run_next", "hold"):
                        raise ValueError("already processing")
                    output_only = action == "set" and body.get("key") in OUTPUT_KEYS
                    if not cur.get("paused") and not output_only:
                        raise ValueError("pause training before changing its settings")
                elif action in ("pause", "resume", "snapshot"):
                    raise ValueError("only the running capture can be paused or snapshotted")
                overrides.apply(name, body)
                status.write(load_config())
                self.send(200, b'{"ok":true}', "application/json")
            except ValueError as e:
                self.send(400, json.dumps({"error": str(e)}).encode(), "application/json")
            except Exception as e:
                self.send(500, json.dumps({"error": str(e)}).encode(), "application/json")

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log(f"dashboard on http://127.0.0.1:{port}")


# ---------------------------------------------------------------- job steps

def fetch_local(item, job_dir, beat, timeout=3600):
    """Materialise the inbox item from iCloud and copy it to the local job dir."""
    deadline = time.time() + timeout
    if item.is_dir:
        request_download(item.path)
        while dataless_files(item.path):
            if time.time() > deadline:
                raise RuntimeError("timed out waiting for iCloud download")
            beat()
            time.sleep(5)
        dest = job_dir / "data" / item.path.name
        if dest.parent.exists():
            shutil.rmtree(dest.parent)
        shutil.copytree(item.path, dest)
        return None
    if is_dataless(item.path):
        request_download(item.path)
        while is_dataless(item.path):
            if time.time() > deadline:
                raise RuntimeError("timed out waiting for iCloud download")
            beat()
            time.sleep(5)
    local = job_dir / item.path.name
    shutil.copyfile(item.path, local)
    return local


def unpack(local_zip, data_dir):
    if data_dir.exists():
        shutil.rmtree(data_dir)
    with zipfile.ZipFile(local_zip) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"corrupt zip member: {bad}")
        z.extractall(data_dir)  # extractall strips absolute paths and '..'


def find_colmap(data_dir):
    for sparse in sorted(data_dir.rglob("sparse/0")):
        root = sparse.parent.parent
        has_cams = any((sparse / f).exists() for f in ("cameras.txt", "cameras.bin"))
        if has_cams and (root / "images").is_dir():
            return root
    raise RuntimeError("no COLMAP model found (only LiDAR captures are supported for now)")


def latest_step(out_dir, prefix):
    best = None
    for p in out_dir.glob(f"{prefix}_*"):
        m = re.fullmatch(rf"{prefix}_(\d+)(\.ply)?", p.name)
        if m:
            n = int(m.group(1))
            best = n if best is None or n > best else best
    return best


def run_cmd(cmd, beat, log_file, timeout=7200):
    """Run a tool, keeping the dashboard alive; output goes to log_file."""
    with open(log_file, "a") as lf:
        lf.write("$ " + " ".join(map(str, cmd)) + "\n")
        lf.flush()
        proc = subprocess.Popen([str(c) for c in cmd], stdout=lf, stderr=subprocess.STDOUT)
        deadline = time.time() + timeout
        while proc.poll() is None:
            if time.time() > deadline:
                proc.kill()
                raise RuntimeError(f"{Path(str(cmd[0])).name} {cmd[1]} timed out")
            beat()
            time.sleep(1)
    if proc.returncode:
        raise RuntimeError(f"{Path(str(cmd[0])).name} {cmd[1]} failed (exit {proc.returncode})")


def _quat_to_R(qw, qx, qy, qz):
    return [[1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]]


def read_poses(images_txt):
    """{name: (image line fields, quaternion, camera centre)} from a COLMAP images.txt."""
    out = {}
    for line in open(images_txt):
        p = line.split()
        if line.startswith("#") or len(p) < 10 or not p[0].isdigit():
            continue
        q = list(map(float, p[1:5]))
        t = list(map(float, p[5:8]))
        R = _quat_to_R(*q)
        out[p[9]] = (p, q, [-(R[0][i]*t[0] + R[1][i]*t[1] + R[2][i]*t[2]) for i in range(3)])
    return out


def mean_reprojection(colmap, model):
    r = subprocess.run([colmap, "model_analyzer", "--path", str(model), "--log_target", "stderr"],
                       capture_output=True, text=True)
    m = re.search(r"Mean reprojection error: ([\d.]+)", r.stderr + r.stdout)
    return float(m.group(1)) if m else None


REFINE_VERSION = "1"   # bump when the refinement method changes, to invalidate the cache
POSE_CACHE = APP_HOME / "pose_cache"


def pose_cache_dir(cfg, colmap_dir, name):
    """Cache folder for a capture's refined poses. The key covers the capture's
    original poses + intrinsics, the method version and the COLMAP version, so a
    different capture or method never reuses stale results."""
    import hashlib
    src = colmap_dir / "sparse" / "0"
    h = hashlib.sha256(REFINE_VERSION.encode())
    for f in ("cameras.txt", "images.txt"):
        h.update((src / f).read_bytes())
    colmap = shutil.which(cfg.get("colmap") or "colmap") or ""
    if colmap:
        r = subprocess.run([colmap, "-h"], capture_output=True, text=True)
        h.update((r.stdout.splitlines() or [""])[0].encode())
    return POSE_CACHE / f"{name}-{h.hexdigest()[:16]}"


FEATURE_POINTS = "points3D_features.txt"   # written by gs/refine.py
SPACE_VERSION = "space1"   # bump when gs/fuse.py's space map changes, to invalidate cached maps


def build_refined_model(colmap_dir, cache, out):
    """A trainable COLMAP folder: cached refined poses, original photos, LiDAR points
    (+ the cached feature points, when there are any)."""
    src = colmap_dir / "sparse" / "0"
    shutil.rmtree(out, ignore_errors=True)
    (out / "sparse" / "0").mkdir(parents=True)
    (out / "images").symlink_to((colmap_dir / "images").resolve())
    for f in ("images.txt", "cameras.txt"):
        shutil.copyfile(cache / f, out / "sparse" / "0" / f)
    if (cache / FEATURE_POINTS).exists():
        shutil.copyfile(cache / FEATURE_POINTS, out / "sparse" / "0" / FEATURE_POINTS)
    pts = src / "points3D.txt" if (src / "points3D.txt").exists() else cache / "points3D.txt"
    shutil.copyfile(pts, out / "sparse" / "0" / "points3D.txt")
    return out


def save_pose_cache(cache, model, stats):
    tmp = cache.with_name(cache.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for f in ("images.txt", "cameras.txt", FEATURE_POINTS):
        if (model / "sparse" / "0" / f).exists():
            shutil.copyfile(model / "sparse" / "0" / f, tmp / f)
    (tmp / "stats.json").write_text(json.dumps(
        dict(stats, saved=dt.datetime.now().isoformat(timespec="seconds")), indent=2) + "\n")
    shutil.rmtree(cache, ignore_errors=True)
    tmp.rename(cache)


def refined_model(cfg, colmap_dir, work, beat, job, worker=None):
    """Refined poses for a capture: from the pose cache if this exact capture was
    refined before, otherwise computed with COLMAP and saved to the cache."""
    cache = pose_cache_dir(cfg, colmap_dir, job["name"])
    job["pose_cache"] = str(cache)   # other per-capture products (the LiDAR space map) live here too
    want_features = bool(cfg.get("init_feature_points"))
    if ((cache / "images.txt").exists() and (cache / "stats.json").exists()
            and (not want_features or (cache / FEATURE_POINTS).exists())):
        stats = json.loads((cache / "stats.json").read_text())
        log(f"using cached pose refinement for {job['name']} ({cache.name})")
        model, stats = build_refined_model(colmap_dir, cache, work / "model"), dict(stats, cached=True)
    else:
        model, stats = refine_poses(cfg, colmap_dir, work, beat, job, worker)
        try:
            save_pose_cache(cache, model, stats)
        except OSError as e:
            log(f"could not save pose cache for {job['name']}: {e}")
    if want_features:
        n = merge_feature_points(model)
        if n:
            stats = dict(stats, feature_points=n)
    return model, stats


def merge_feature_points(model):
    """Append COLMAP's triangulated feature points to the model's (LiDAR) points3D.txt, so
    training starts with splats on distant geometry too. Returns how many were added."""
    sp = model / "sparse" / "0"
    feats = sp / FEATURE_POINTS
    if not feats.exists():
        return 0
    def rows(f):   # ID X Y Z R G B ERROR [track] -> the 7 values after the ID
        if not f.exists():
            return []
        return [l.split()[1:8] for l in f.read_text().splitlines()
                if l.strip() and not l.startswith("#") and len(l.split()) >= 8]
    lidar, extra = rows(sp / "points3D.txt"), rows(feats)
    (sp / "points3D.txt").write_text("# LiDAR points + COLMAP feature points\n" + "".join(
        f"{i} {' '.join(r)}\n" for i, r in enumerate(lidar + extra, 1)))
    return len(extra)


def refine_poses(cfg, colmap_dir, work, beat, job, worker=None):
    """Refine ARKit/LiDAR poses with COLMAP (gs/refine.py) and return a new COLMAP folder with
    the refined poses, the original photos and LiDAR points, and COLMAP's feature points.
    Runs on a remote worker that has COLMAP when one is given (GPU SIFT; its photos are then
    reused for training), else, or if that fails, on this Mac. Raises on failure; the caller
    falls back to the original poses."""
    src = colmap_dir / "sparse" / "0"
    if not (src / "images.txt").exists():
        raise RuntimeError("pose refinement needs a text COLMAP model (images.txt)")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    out = work / "refined"
    stats = None
    if worker is not None and worker.remote and worker.can_refine():
        try:
            stats = worker.refine(cfg, job, colmap_dir, out, beat)
        except Exception as e:
            log(f"pose refinement on {worker.label} failed, refining on the Mac instead: {e}")
            warn = job.setdefault("warnings", [])
            warn.append(f"refined on the Mac ({worker.label}: {str(e)[:80]})")
    if stats is None:
        stats = refine_locally(cfg, colmap_dir, work, out, beat, job)
    model = work / "model"
    (model / "sparse" / "0").mkdir(parents=True)
    (model / "images").symlink_to((colmap_dir / "images").resolve())
    for f in ("images.txt", "cameras.txt", FEATURE_POINTS):
        shutil.copyfile(out / f, model / "sparse" / "0" / f)
    pts = src / "points3D.txt" if (src / "points3D.txt").exists() else out / FEATURE_POINTS
    shutil.copyfile(pts, model / "sparse" / "0" / "points3D.txt")
    log(f"refined poses for {job['name']}: {stats}")
    return model, stats


def refine_locally(cfg, colmap_dir, work, out, beat, job):
    colmap = shutil.which(cfg.get("colmap") or "colmap") or cfg.get("colmap")
    if not colmap or not os.access(colmap, os.X_OK):
        raise RuntimeError("colmap not found (brew install colmap)")
    cmd = [sys.executable, "-I", "-u", str(GS_DIR / "refine.py"), "--colmap", colmap,
           "--images", str(colmap_dir / "images"), "--model", str(colmap_dir / "sparse" / "0"),
           "--work", str(work / "colmap"), "--out", str(out)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = ""
    for line in proc.stdout:   # PHASE lines drive the dashboard; the last line is the result
        line = line.strip()
        if line.startswith("PHASE: "):
            job["phase"] = f"refining poses: {line[7:]}"
            beat(force=True)
        elif line:
            last = line
    proc.wait()
    if proc.returncode:
        raise RuntimeError(last.removeprefix("error: ")[:200] or f"refine.py exited with {proc.returncode}")
    return json.loads((out / "stats.json").read_text())


RESTART = object()  # train() result: settings changed while paused, run again


def same_settings(a, b):
    return all(a.get(k) == b.get(k) for k in RESTART_KEYS)


def train(cfg, paths, job, colmap_dir, out_dir, beat, pending_settings):
    """Run Brush until done. Honours job["request"] = "pause" | "resume" (set by
    the dashboard): pause SIGSTOPs Brush in place; resume either SIGCONTs it, or,
    if the settings were changed while paused, kills it and returns RESTART."""
    if not os.access(cfg["brush"], os.X_OK):
        raise RuntimeError(f"Brush not found at {cfg['brush']} (run scripts/install.sh)")
    st = job["settings"]
    total = int(st["total_steps"])
    cmd = [cfg["brush"], str(colmap_dir),
           "--total-steps", str(total),
           "--max-resolution", str(st["max_resolution"]),
           "--sh-degree", str(st["sh_degree"]),
           "--max-splats", str(st["max_splats"]),
           "--subsample-frames", str(st["subsample_frames"]),
           "--export-every", str(cfg["checkpoint_every"]),
           "--eval-every", str(cfg["eval_every"]),
           "--eval-split-every", str(cfg["eval_split_every"]),
           "--eval-save-to-disk",
           "--export-path", str(out_dir)] + [str(a) for a in cfg["extra_args"]]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    log("run: " + " ".join(cmd))
    with open(out_dir.parent / "train.log", "w") as tlog:
        # cwd matters: Brush writes a GPU autotune cache relative to it (launchd's cwd is /)
        proc = subprocess.Popen(cmd, stdout=tlog, stderr=subprocess.STDOUT,
                                cwd=out_dir.parent)
    keep_awake = lambda: subprocess.Popen(["caffeinate", "-i", "-m", "-w", str(proc.pid)])
    awake = keep_awake()
    job.update(phase="training", step=0, total=total, started=time.time(),
               train_started=time.time(), eta=None, paused=False, paused_at=None,
               request=None)
    rotate = capture_rotation(out_dir.parent / "data")
    notified = set()
    last_preview_step = None
    last_progress = 0.0
    try:
        while proc.poll() is None:
            time.sleep(1)
            req, job["request"] = job.get("request"), None
            if req == "pause" and not job["paused"]:
                os.kill(proc.pid, signal.SIGSTOP)
                awake.terminate()  # let the Mac sleep while paused
                job.update(paused=True, paused_at=time.time(), phase="paused", eta=None)
                log(f"paused: {job['name']} at step {job['step']}")
                beat(force=True)
            elif req == "resume" and job["paused"]:
                new = pending_settings()
                if not same_settings(new, st):
                    log(f"settings changed while paused; restarting {job['name']}")
                    return RESTART  # finally: kills the stopped process
                os.kill(proc.pid, signal.SIGCONT)
                gap = time.time() - job["paused_at"]
                job["started"] += gap        # keep elapsed/ETA honest
                job["train_started"] += gap
                job.update(paused=False, paused_at=None, phase="training")
                awake = keep_awake()
                log(f"resumed: {job['name']}")
                beat(force=True)
            if job["paused"] or time.time() - last_progress < 5:
                beat()
                continue
            last_progress = time.time()
            step = latest_step(out_dir, "eval")
            if step:
                job["step"] = step
                rate = (time.time() - job["started"]) / step
                job["eta"] = time.time() + rate * (total - step)
                ckpt = int(cfg["checkpoint_every"])
                due = any(f not in notified and step >= f * total
                          for f in cfg["notify_checkpoints"])
                if step != last_preview_step and (step % ckpt == 0 or step == total or due):
                    pngs = sorted((out_dir / f"eval_{step}").glob("*.png"))
                    if pngs:
                        name = f"{job['name']}_step{step:05d}.jpg"
                        try:
                            make_jpeg(pngs[0], paths.previews / name, 1280, rotate)
                            job["preview"] = name
                            last_preview_step = step
                        except subprocess.CalledProcessError:
                            pass  # PNG still being written; retry next poll
                crossed = [f for f in cfg["notify_checkpoints"]
                           if f not in notified and step >= f * total]
                if crossed and last_preview_step == step:
                    notified.update(crossed)  # one text even if several were passed
                    img = paths.attach / job["preview"]
                    shutil.copyfile(paths.previews / job["preview"], img)
                    imessage(cfg, f"🟡 {job['name']}: {100 * step // total}% "
                                  f"({step:,}/{total:,} steps), "
                                  f"ETA ~{dt.datetime.fromtimestamp(job['eta']):%H:%M}", img)
            beat()
    finally:
        if proc.poll() is None:
            proc.kill()   # SIGKILL also works on a SIGSTOPped process
            proc.wait()
        awake.terminate()
        job.update(paused=False, paused_at=None)
    if proc.returncode != 0:
        reason = ""
        tlines = (out_dir.parent / "train.log").read_text(errors="replace").splitlines()
        for i, line in enumerate(tlines):
            if "panicked at" in line and i + 1 < len(tlines):
                reason = ": " + tlines[i + 1].strip()[:100]
                break
        raise RuntimeError(f"brush exited with code {proc.returncode}{reason}")
    job["train_seconds"] = time.time() - job["train_started"]
    final = out_dir / f"export_{total}.ply"
    if not final.exists():
        raise RuntimeError("brush finished but no final .ply was written")
    return final


COMPRESS_FORMATS = ("spz",)
FLIP = "180,0,0"   # degrees about X, Y, Z


def splat_transform(cfg, args, dst, beat=lambda: None, timeout=1800):
    """Run SplatTransform; returns None on success or a short error string."""
    tool = cfg["splat_transform"]
    if not os.access(tool, os.X_OK):
        return f"SplatTransform not found at {tool}"
    proc = subprocess.Popen([tool, "--no-tty", "-q", "-w"] + [str(a) for a in args],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + timeout
    while proc.poll() is None and time.time() < deadline:
        beat()
        time.sleep(1)
    if proc.poll() is None:
        proc.kill()
        return "timed out"
    out = proc.stdout.read().strip()
    if proc.returncode or not Path(dst).exists():
        log(f"splat-transform {' '.join(map(str, args))} failed: {out[-500:]}")
        return (out.splitlines() or ["failed"])[-1][:120]
    return None


def native_spz(cfg):
    tool = cfg.get("spz_native")
    return tool if tool and os.access(tool, os.X_OK) else None


def run_native(cfg, args, dst, beat=lambda: None, timeout=3600):
    """Run gq_spz; returns None on success or a short error string."""
    proc = subprocess.Popen([native_spz(cfg)] + [str(a) for a in args],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + timeout
    while proc.poll() is None and time.time() < deadline:
        beat()
        time.sleep(1)
    if proc.poll() is None:
        proc.kill()
        return "timed out"
    out = proc.stdout.read().strip()
    if proc.returncode or not Path(dst).exists():
        log(f"gq_spz {' '.join(map(str, args))} failed: {out[-500:]}")
        return (out.splitlines() or ["failed"])[-1][:120]
    return None


def spz_versions(st):
    """SPZ versions a capture's settings ask for, e.g. [3] or [3, 4]."""
    return [v for v in (3, 4) if st.get(f"spz_v{v}")]


def compress(cfg, ply, out_dir, name, beat, formats=None, versions=None):
    """Write the compressed copies: <name>.v3.spz / <name>.v4.spz for the SPZ versions asked
    for (natively with gq_spz when installed, otherwise with SplatTransform). An output may
    also be given as ("spz", version) for an old-style <name>.spz.
    Returns ({file suffix: bytes}, [error strings]); failures never fail the job."""
    if formats is None:
        formats = [f for f in cfg.get("compress") or [] if f in COMPRESS_FORMATS]
    if versions is None:
        versions = [int(v) for v in cfg.get("spz_versions") or [3]]
    outputs = [(f"v{v}.spz", v) for v in versions] if "spz" in formats else []
    outputs += [o for o in formats if isinstance(o, tuple)]
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes, errors = {}, []
    for suffix, version in outputs:
        dst = out_dir / f"{name}.{suffix}"
        if native_spz(cfg):   # native: no WebAssembly memory ceiling
            err = run_native(cfg, [ply, dst, "--spz-version", version], dst, beat)
        else:
            # -N drops NaN/degenerate splats, which some viewers choke on
            err = splat_transform(cfg, [ply, "-N", "--spz-version", version, dst], dst, beat)
        if err:
            errors.append(f"{suffix}: {err}")
        else:
            sizes[suffix] = dst.stat().st_size
    return sizes, errors


def start_view_pose(model_dir, flip, forward="-z"):
    """The first capture photo's camera, in the output's coordinates: (photo name, position,
    yaw in degrees that turns its heading onto the viewer's forward axis). None if unknown."""
    images = Path(model_dir) / "sparse" / "0" / "images.txt"
    if not images.exists():
        return None
    poses = read_poses(images)
    if not poses:
        return None
    name = sorted(poses)[0]   # SplatKing names photos by capture time
    fields, q, centre = poses[name]
    R = _quat_to_R(*q)
    fwd = [R[2][i] for i in range(3)]   # camera +Z (viewing direction) in world coordinates
    if flip:   # 180° about X, as gq_spz --flip
        centre = [centre[0], -centre[1], -centre[2]]
        fwd = [fwd[0], -fwd[1], -fwd[2]]
    if math.hypot(fwd[0], fwd[2]) < 1e-6:   # looking straight up or down: no heading
        yaw = 0.0
    else:
        yaw = -math.degrees(math.atan2(fwd[0], fwd[2]))   # heading onto +Z
        if forward == "-z":
            yaw += 180.0
        yaw = (yaw + 180.0) % 360.0 - 180.0
    return name, [round(c, 6) for c in centre], round(yaw, 4)


def trainable_model(job_dir):
    """The COLMAP folder a job trained on: refined poses if there are any, else the capture's."""
    refined = Path(job_dir) / "refine" / "model"
    if (refined / "sparse" / "0" / "images.txt").exists():
        return refined
    try:
        return find_colmap(Path(job_dir) / "data")
    except Exception:
        return None


def orient_ply(cfg, src, dst, flip, view, beat=lambda: None, keep=1.0):
    """Write src to dst flipped and/or moved to the start view (one native pass), optionally
    keeping only the `keep` fraction of most important splats (a quick prune, no GPU). Returns
    None or a short error. Without gq_spz, only the flip is possible (SplatTransform)."""
    if native_spz(cfg):
        args = [src, dst] + (["--flip"] if flip else [])
        if view:
            args += ["--origin", *view[1], "--yaw", view[2]]
        if keep < 1:
            args += ["--keep", round(keep, 4)]
        return run_native(cfg, args, dst, beat)
    if keep < 1:
        return "quick pruning needs gq_spz (scripts/install.sh builds it)"
    if view:
        return "start view needs gq_spz (scripts/install.sh builds it)"
    return splat_transform(cfg, [src, "-r", FLIP, "-N", dst], dst, beat)


def flip_ply(cfg, src, dst, beat=lambda: None):
    """Rotate a splat 180° about X (turns a +Y-up scene -Y-up and vice versa), including the
    spherical harmonics. Natively with gq_spz when installed (fast, low memory), otherwise
    with SplatTransform."""
    if native_spz(cfg):
        return run_native(cfg, [src, dst, "--flip"], dst, beat)
    return splat_transform(cfg, [src, "-r", FLIP, "-N", dst], dst, beat)


def share_result(cfg, result_dir):
    """Copy the phone-friendly files of one result into share_dir/<name>/.
    Entries in share_files are either a format ("spz" -> <name>.v3.spz, <name>.v4.spz and any
    old-style <name>.spz) or a filename."""
    if not cfg.get("share_dir"):
        return []
    name = result_dir.name
    dest = Path(cfg["share_dir"]) / name
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    names = []
    for entry in cfg.get("share_files") or []:
        if entry == "spz":
            names += sorted(f.name for f in result_dir.glob(f"{glob.escape(name)}.v*.spz")) + [f"{name}.spz"]
        else:
            names.append(f"{name}.{entry}" if "." not in entry else entry)
    for fname in names:
        src = result_dir / fname
        if src.is_file():
            tmp = dest / f".{fname}.partial"
            shutil.copyfile(src, tmp)
            tmp.replace(dest / fname)   # never expose a half-copied file to iCloud
            copied.append(fname)
    return copied


def write_meta(dest, job, files, flipped):
    meta = {"name": job["name"], "finished": dt.datetime.now().isoformat(timespec="seconds"),
            "settings": {k: v for k, v in (job.get("settings") or {}).items() if k != "base"},
            "frames": job.get("frames"), "files": files,
            "rotation_xyz_degrees": [180, 0, 0] if flipped else [0, 0, 0]}
    if job.get("refine"):
        meta["pose_refinement"] = job["refine"]
    if job.get("metrics"):
        meta["held_out_photos"] = job["metrics"]
    if job.get("depth_fusion"):
        meta["depth_fusion"] = job["depth_fusion"]
    if job.get("start_view"):
        meta["start_view"] = job["start_view"]
    if job.get("prune"):
        meta["pruning"] = job["prune"]
    if job.get("quality"):
        meta["capture_quality"] = job["quality"]
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


# ---------------------------------------------------------------- training machines

class LocalWorker:
    """This Mac."""
    name, label, remote, gsplat = "mac", "Mac", False, None

    def __init__(self):
        self.stats = {}

    def refresh(self):
        self.stats = {"mem_free_pct": memory_free_pct()}

    def info(self, job):
        return {"name": self.name, "label": self.label, "remote": False, "online": True, "gsplat": False,
                "job": job and {"name": job["name"], "phase": job.get("phase"),
                                "step": job.get("step"), "total": job.get("total")},
                **self.stats}

    def finish(self, name):
        pass


class RemoteWorker:
    """A GPU machine reached over SSH (key auth via an ~/.ssh/config alias). Runs
    PowerShell on Windows; the Mac uploads the trainable capture, starts Brush,
    polls progress renders and downloads the result. Pause isn't supported."""
    remote = True

    def __init__(self, spec):
        self.spec = dict(spec)
        self.name = spec["name"]
        self.label = spec.get("label") or self.name
        self.ssh = spec["ssh"]
        self.brush = spec["brush"].replace("\\", "/")
        g = spec.get("gsplat") or {}
        self.gsplat = ({"python": g["python"].replace("\\", "/"), "scripts": g["scripts"].replace("\\", "/")}
                       if g.get("python") and g.get("scripts") else None)
        self.root = spec["work_dir"].replace("\\", "/").rstrip("/")
        self.online = False
        self.checked = 0.0
        self.stats = {}

    # -- plumbing
    def _ssh(self, *args):
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4", self.ssh, *args]

    def ps(self, script, timeout=120, check=True):
        """Run a PowerShell script remotely (encoded, so no quoting problems)."""
        enc = base64.b64encode(("$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'\n"
                                + script).encode("utf-16-le")).decode()
        r = subprocess.run(self._ssh("powershell", "-NoProfile", "-NonInteractive",
                                     "-EncodedCommand", enc),
                           capture_output=True, text=True, timeout=timeout)
        if check and r.returncode:
            lines = (r.stderr or r.stdout).strip().splitlines()
            raise RuntimeError(f"{self.label}: {lines[-1] if lines else 'ssh failed'}")
        return r.stdout

    def job_dir(self, name):
        return f"{self.root}/{name}"

    def download(self, remote_path, local):
        tmp = Path(str(local) + ".partial")
        r = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{self.ssh}:/{remote_path}", str(tmp)],
                           capture_output=True, text=True, timeout=3600)
        if r.returncode:
            raise RuntimeError(f"download from {self.label} failed: {r.stderr.strip()[:200]}")
        tmp.replace(local)

    def remove(self, remote_path):
        self.ps(f"Remove-Item -Recurse -Force -ErrorAction SilentlyContinue '{remote_path}'", check=False)

    # -- status
    def refresh(self):
        try:
            out = self.ps("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
                          "--format=csv,noheader,nounits", timeout=15)
            util, used, total = [int(x.strip()) for x in out.strip().splitlines()[0].split(",")]
            self.stats = {"gpu_util": util, "vram_used_mb": used, "vram_total_mb": total}
            self.online = True
        except Exception:
            self.online, self.stats = False, {}
        self.checked = time.time()

    def info(self, job):
        return {"name": self.name, "label": self.label, "remote": True, "online": self.online,
                "gsplat": bool(self.gsplat),
                "job": job and {"name": job["name"], "phase": job.get("phase"),
                                "step": job.get("step"), "total": job.get("total")},
                **self.stats}

    # -- checkpoints for snapshots
    def latest_export(self, name):
        out = self.ps(f"Get-ChildItem '{self.job_dir(name)}/out' -Filter export_*.ply -ErrorAction "
                      "SilentlyContinue | ForEach-Object { $_.Name }", check=False)
        steps = [int(m.group(1)) for m in re.finditer(r"export_(\d+)\.ply", out)]
        return max(steps) if steps else None

    def keep_export(self, name, step):
        d = self.job_dir(name)
        self.ps(f"New-Item -ItemType Directory -Force '{d}/snapshots' | Out-Null; "
                f"Copy-Item -Force '{d}/out/export_{step}.ply' '{d}/snapshots/export_{step}.ply'",
                timeout=600)
        return f"{d}/snapshots/export_{step}.ply"

    # -- job plumbing
    def upload_dir(self, parent, names, dest, beat):
        """Stream folders to the worker with tar (-h follows symlinks such as images/)."""
        env = dict(os.environ, COPYFILE_DISABLE="1")   # no macOS ._ metadata files
        tar = subprocess.Popen(["tar", "-C", str(parent), "-chf", "-", *names],
                               stdout=subprocess.PIPE, env=env)
        up = subprocess.Popen(self._ssh(f'tar -xf - -C "{dest}"'), stdin=tar.stdout,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        tar.stdout.close()
        while up.poll() is None:
            beat()
            time.sleep(1)
        tar.wait()
        if up.returncode or tar.returncode:
            raise RuntimeError(f"upload to {self.label} failed: {up.stderr.read().decode()[-200:]}")

    def upload_file(self, local, remote_path):
        r = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(local), f"{self.ssh}:/{remote_path}"],
                           capture_output=True, text=True, timeout=7200)
        if r.returncode:
            raise RuntimeError(f"upload to {self.label} failed: {r.stderr.strip()[:200]}")

    def upload_capture(self, d, colmap_dir, sensor, beat, have_images=False):
        """A fresh <d>/data with the trainable photos + poses (and the LiDAR depth maps).
        have_images: the photos are already there (uploaded for pose refinement); keep them."""
        if have_images:
            self.ps(f"Remove-Item -Recurse -Force -ErrorAction SilentlyContinue '{d}/data/sparse','{d}/out',"
                    f"'{d}/data/sensor_data'; New-Item -ItemType Directory -Force '{d}/out' | Out-Null")
            self.upload_dir(colmap_dir, ["sparse"], f"{d}/data", beat)
        else:
            self.ps(f"Remove-Item -Recurse -Force -ErrorAction SilentlyContinue '{d}'; "
                    f"New-Item -ItemType Directory -Force '{d}/data','{d}/out' | Out-Null")
            self.upload_dir(colmap_dir, ["images", "sparse"], f"{d}/data", beat)
        if sensor:
            self.upload_dir(sensor.parent, [sensor.name], f"{d}/data", beat)

    def can_refine(self):
        """Pose refinement here needs COLMAP and the gs scripts (run with the gsplat Python)."""
        return bool(self.spec.get("colmap") and self.gsplat and self.online)

    def refine(self, cfg, job, colmap_dir, out, beat):
        """Run gs/refine.py here on the capture's photos and original poses; download the
        refined poses into out. The photos stay in <job dir>/data/images for training."""
        name, d = job["name"], self.job_dir(job["name"])
        job["phase"] = f"uploading to {self.label} for pose refinement"
        beat(force=True)
        self.ps(f"Remove-Item -Recurse -Force -ErrorAction SilentlyContinue '{d}'; "
                f"New-Item -ItemType Directory -Force '{d}/data','{d}/orig' | Out-Null")
        self.upload_dir(colmap_dir, ["images"], f"{d}/data", beat)
        self.upload_dir(colmap_dir, ["sparse"], f"{d}/orig", beat)
        job["images_on"] = self.name
        self.sync_gsplat()
        cwd = self.gsplat["scripts"]
        args = [f"{cwd}/refine.py", "--colmap", self.spec["colmap"].replace("\\", "/"),
                "--images", f"{d}/data/images", "--model", f"{d}/orig/sparse/0",
                "--work", f"{d}/refine", "--out", f"{d}/refined"]
        if self.spec.get("colmap_gpu", True):
            args.append("--gpu")
        log(f"run on {self.label}: refine {' '.join(map(str, args))}")
        job["phase"] = f"refining poses on {self.label}"
        beat(force=True)

        def poll():
            out_ = self.ps(f"Get-Content '{d}/refine.log' -Tail 5 -ErrorAction SilentlyContinue", check=False)
            phases = [l[7:].strip() for l in out_.splitlines() if l.startswith("PHASE: ")]
            if phases:
                job["phase"] = f"refining poses on {self.label}: {phases[-1]}"
        code = self.run(self.gsplat["python"], args, cwd, d, "refine", beat, poll)
        if code:
            last = self.ps(f"Get-Content '{d}/refine.log' -Tail 3 -ErrorAction SilentlyContinue", check=False)
            raise RuntimeError(f"exit {code}: {last.strip().splitlines()[-1][:160] if last.strip() else ''}")
        out.mkdir(parents=True, exist_ok=True)
        for f in ("images.txt", "cameras.txt", FEATURE_POINTS, "stats.json"):
            self.download(f"{d}/refined/{f}", out / f)
        return dict(json.loads((out / "stats.json").read_text()), machine=self.label)

    def sync_gsplat(self):
        """Copy this repo's gs/*.py to the worker, so it always runs the matching code."""
        dst = self.gsplat["scripts"]
        self.ps(f"New-Item -ItemType Directory -Force '{dst}' | Out-Null")
        r = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", *sorted(map(str, GS_DIR.glob("*.py"))),
                            f"{self.ssh}:/{dst}/"], capture_output=True, text=True, timeout=300)
        if r.returncode:
            raise RuntimeError(f"copying the gsplat scripts to {self.label} failed: {r.stderr.strip()[:200]}")

    def run(self, exe, args, cwd, d, name, beat, poll=None):
        """Run a program on the worker and wait for it, logging to <d>/<name>.log/.err.
        poll() is called about every 10 s. Stopping (an exception here) kills it."""
        arglist = ", ".join("'" + str(a).replace("'", "''") + "'" for a in args)
        script = (f"$env:PYTHONUNBUFFERED='1'\n"
                  f"$p = Start-Process -FilePath '{exe}' -ArgumentList @({arglist}) "
                  f"-WorkingDirectory '{cwd}' -NoNewWindow -PassThru "
                  f"-RedirectStandardOutput '{d}/{name}.log' -RedirectStandardError '{d}/{name}.err'\n"
                  f"$null = $p.Handle\n$p.Id | Out-File -Encoding ascii '{d}/pid.txt'\n"
                  f"$p.WaitForExit()\nexit $p.ExitCode")
        enc = base64.b64encode(("$ErrorActionPreference='Stop'\n" + script).encode("utf-16-le")).decode()
        proc = subprocess.Popen(self._ssh("powershell", "-NoProfile", "-NonInteractive",
                                          "-EncodedCommand", enc),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        last_poll = time.time()
        try:
            while proc.poll() is None:
                time.sleep(1)
                beat()
                if poll and time.time() - last_poll >= 10:
                    last_poll = time.time()
                    poll()
        except BaseException:
            self.ps(f"$i = Get-Content '{d}/pid.txt' -ErrorAction SilentlyContinue; "
                    "if ($i) { Stop-Process -Id $i -Force -ErrorAction SilentlyContinue }", check=False)
            proc.kill()
            raise
        return proc.returncode

    def failure(self, d, name):
        """The most telling line from a failed run's logs."""
        out = self.ps(f"Get-Content '{d}/{name}.err','{d}/{name}.log' -Tail 8 -ErrorAction SilentlyContinue",
                      check=False)
        return next((l.strip() for l in reversed(out.splitlines()) if "panicked" in l or "rror" in l), "")[:160]

    # -- the job
    def train(self, cfg, paths, job, colmap_dir, out_dir, beat, pending_settings, sensor=None):
        name, d = job["name"], self.job_dir(job["name"])
        st = job["settings"]
        total = int(st["total_steps"])
        gs = st.get("trainer") == "gsplat" and bool(self.gsplat)
        job.update(phase=f"uploading to {self.label}", step=None, eta=None, paused=False)
        beat(force=True)
        self.upload_capture(d, colmap_dir, sensor if gs else None, beat,
                            have_images=job.get("images_on") == self.name)
        if gs:
            self.sync_gsplat()
            g = sub_config(cfg, "gsplat")
            what, exe, cwd = "gsplat", self.gsplat["python"], self.gsplat["scripts"]
            args = [f"{cwd}/train.py", f"{d}/data", "--out", f"{d}/out", "--steps", total,
                    "--max-res", st["max_resolution"], "--sh-degree", st["sh_degree"],
                    "--max-splats", min(int(st["max_splats"]), int(g["max_splats"])),
                    "--subsample", st["subsample_frames"],
                    "--strategy", st.get("densify") or "mcmc", "--mode", g["mode"],
                    "--sky-points", g["sky_points"], "--export-every", cfg["checkpoint_every"],
                    "--eval-every", cfg["eval_every"], "--holdout-every", cfg["eval_split_every"]]
            if sensor:   # depth maps: supervision when on, held-out depth error either way
                raw = f"{d}/data/{sensor.name}"
                depth = raw
                if st.get("depth_supervision") and g.get("depth_fusion"):
                    depth = self.fuse_depth(cfg, job, d, raw, beat)
                args += ["--depth-dir", depth, "--eval-depth-dir", raw]
                if g.get("carve") or g.get("far_flat_weight"):
                    space = self.space_map(cfg, job, d, raw, beat)
                    if space:
                        args += ["--space", space] + (["--carve"] if g.get("carve") else [])
                        if g.get("far_flat_weight"):
                            args += ["--far-flat-weight", g["far_flat_weight"]]
                if st.get("depth_supervision"):
                    args += ["--depth-weight", g["depth_weight"]]
            if st.get("flatten"):
                args += ["--flat-weight", g["flat_weight"]]
        else:
            what, exe, cwd = "brush", self.brush, d
            args = [f"{d}/data", "--total-steps", total, "--max-resolution", st["max_resolution"],
                    "--sh-degree", st["sh_degree"], "--max-splats", st["max_splats"],
                    "--subsample-frames", st["subsample_frames"],
                    "--export-every", cfg["checkpoint_every"], "--eval-every", cfg["eval_every"],
                    "--eval-split-every", cfg["eval_split_every"], "--eval-save-to-disk",
                    "--export-path", f"{d}/out"] + list(cfg["extra_args"])
        log(f"run on {self.label}: {what} {' '.join(map(str, args))}")
        job.update(phase="training", step=0, total=total, started=time.time(),
                   train_started=time.time(), eta=None)
        rotate = capture_rotation(out_dir.parent / "data")
        notified, last_preview = set(), [None]
        tmp = APP_HOME / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)

        def poll():   # progress = the newest eval_<step> folder; fetch its render as the preview
            try:
                listing = self.ps(f"Get-ChildItem '{d}/out' -Directory -Filter eval_* | "
                                  "ForEach-Object { $_.Name }", timeout=30, check=False)
            except subprocess.TimeoutExpired:
                return
            steps = [int(m.group(1)) for m in re.finditer(r"eval_(\d+)", listing)]
            if not steps:
                return
            step = max(steps)
            job["step"] = step
            rate = (time.time() - job["started"]) / step
            job["eta"] = time.time() + rate * (total - step)
            ckpt = int(cfg["checkpoint_every"])
            due = any(f not in notified and step >= f * total for f in cfg["notify_checkpoints"])
            if step != last_preview[0] and (step % ckpt == 0 or step == total or due):
                names = self.ps(f"Get-ChildItem '{d}/out/eval_{step}' -Filter *.png | Sort-Object Name | "
                                "Select-Object -First 1 | ForEach-Object { $_.Name }", check=False).strip()
                if names:
                    try:
                        local_png = tmp / f"{name}_{step}.png"
                        self.download(f"{d}/out/eval_{step}/{names}", local_png)
                        preview = f"{name}_step{step:05d}.jpg"
                        make_jpeg(local_png, paths.previews / preview, 1280, rotate)
                        local_png.unlink(missing_ok=True)
                        job["preview"], last_preview[0] = preview, step
                    except Exception as e:
                        log(f"preview from {self.label} failed: {e}")
            crossed = [f for f in cfg["notify_checkpoints"] if f not in notified and step >= f * total]
            if crossed and last_preview[0] == step:
                notified.update(crossed)
                img = paths.attach / job["preview"]
                shutil.copyfile(paths.previews / job["preview"], img)
                imessage(cfg, f"🟡 {name} on {self.label}: {100 * step // total}% "
                              f"({step:,}/{total:,} steps), "
                              f"ETA ~{dt.datetime.fromtimestamp(job['eta']):%H:%M}", img)

        code = self.run(exe, args, cwd, d, "train", beat, poll)
        if code:
            raise RuntimeError(f"{what} on {self.label} exited with {code} {self.failure(d, 'train')}")
        job["train_seconds"] = time.time() - job["train_started"]
        job["phase"] = f"downloading result from {self.label}"
        beat(force=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        final = out_dir / f"export_{total}.ply"
        self.download(f"{d}/out/export_{total}.ply", final)
        try:
            self.download(f"{d}/train.log", out_dir.parent / "train.log")
        except Exception:
            (out_dir.parent / "train.log").write_text("")
        if gs:
            try:
                self.download(f"{d}/out/metrics.json", out_dir / "metrics.json")
                job["metrics"] = json.loads((out_dir / "metrics.json").read_text()).get("test")
                if job["metrics"]:
                    job["metrics"].pop("views", None)
            except Exception as e:
                log(f"no metrics from {self.label}: {e}")
        # The worker's copy stays until finish(): pruning may still use it.
        return final

    def space_map(self, cfg, job, d, raw, beat):
        """The LiDAR space map for carving: from the pose cache when this capture (same refined
        poses, same held-out split) was mapped before, else built here and cached."""
        cache = (Path(job["pose_cache"]) / f"{SPACE_VERSION}-h{cfg['eval_split_every']}.npz"
                 if job.get("pose_cache") and Path(job["pose_cache"]).is_dir() else None)
        remote = f"{d}/data/space/space.npz"
        if cache and cache.exists():
            try:
                self.ps(f"New-Item -ItemType Directory -Force '{d}/data/space' | Out-Null")
                self.upload_file(cache, remote)
                log(f"using cached LiDAR space map for {job['name']}")
                return remote
            except Exception as e:
                log(f"cached space map upload failed: {e}")
        space = self.fuse_depth(cfg, job, d, raw, beat, grid_only=True)
        if space and cache:
            try:
                self.download(space, cache)
            except Exception as e:
                log(f"could not cache the space map: {e}")
        return space

    def fuse_depth(self, cfg, job, d, raw, beat, grid_only=False):
        """Multi-view LiDAR fusion (gs/fuse.py) on the worker. Returns the fused depth folder
        (raw if fusion fails), or with grid_only the space.npz map of known-empty space (None
        if it fails). Held-out photos are left out of the fusion."""
        job["phase"] = f"{'mapping LiDAR space' if grid_only else 'fusing LiDAR depth'} on {self.label}"
        beat(force=True)
        cwd = self.gsplat["scripts"]
        out_dir = f"{d}/data/{'space' if grid_only else 'fused'}"
        args = [f"{cwd}/fuse.py", f"{d}/data", "--depth-dir", raw, "--out", out_dir,
                "--exclude-every", cfg["eval_split_every"]] + (["--grid-only"] if grid_only else [])
        log(f"run on {self.label}: fuse {' '.join(map(str, args))}")
        try:
            code = self.run(self.gsplat["python"], args, cwd, d, "fuse", beat)
            if code:
                raise RuntimeError(f"exit {code} {self.failure(d, 'fuse')}")
            if grid_only:
                return f"{out_dir}/space.npz"
            out = self.ps(f"Get-Content '{out_dir}/fusion.json' -Raw", check=False)
            job["depth_fusion"] = {k: v for k, v in json.loads(out).items() if k != "frames"}
            return out_dir
        except Exception as e:
            log(f"LiDAR fusion failed on {self.label}: {e}")
            return None if grid_only else raw

    def prune(self, cfg, job, colmap_dir, sensor, ply, pct, out_dir, beat, trained_here):
        """Prune a trained result on this worker. Returns (local pruned .ply, report)."""
        name, st = job["name"], job["settings"]
        pc, g = sub_config(cfg, "prune"), sub_config(cfg, "gsplat")
        if trained_here:   # the capture and the result are still on the worker
            d = self.job_dir(name)
            remote_ply = f"{d}/out/{Path(ply).name}"
            if sensor and not (st.get("trainer") == "gsplat"):   # Brush runs didn't upload depth
                self.upload_dir(sensor.parent, [sensor.name], f"{d}/data", beat)
        else:
            d = self.job_dir(name + "-prune")
            job["phase"] = f"uploading to {self.label} for pruning"
            beat(force=True)
            self.upload_capture(d, colmap_dir, sensor, beat)
            remote_ply = f"{d}/in.ply"
            self.upload_file(ply, remote_ply)
        self.sync_gsplat()
        job["phase"] = f"pruning {pct}% on {self.label}"
        beat(force=True)
        cwd = self.gsplat["scripts"]
        args = [f"{cwd}/prune.py", remote_ply, f"{d}/data", "--out", f"{d}/prune",
                "--score", pc["score"], "--keep", round(1 - pct / 100, 4),
                "--finetune", pc["finetune_steps"], "--eval-res", st["max_resolution"], "--holdout-every", 0]
        if sensor:
            fused = trained_here and job.get("depth_fusion")   # fine-tune on the fused depth too
            args += ["--depth-dir", f"{d}/data/fused" if fused else f"{d}/data/{sensor.name}"]
            if st.get("depth_supervision"):
                args += ["--depth-weight", g["depth_weight"]]
        if st.get("flatten"):
            args += ["--flat-weight", g["flat_weight"]]
        log(f"run on {self.label}: prune {' '.join(map(str, args))}")
        try:
            code = self.run(self.gsplat["python"], args, cwd, d, "prune", beat)
            if code:
                raise RuntimeError(f"pruning on {self.label} exited with {code} {self.failure(d, 'prune')}")
            local = out_dir / f"pruned_{pct}.ply"
            self.download(f"{d}/prune/pruned.ply", local)
            self.download(f"{d}/prune/prune.json", out_dir / "prune.json")
            return local, json.loads((out_dir / "prune.json").read_text())
        finally:
            if not trained_here:
                self.remove(d)

    def finish(self, name):
        """Free the worker's disk once the job is completely done."""
        self.remove(self.job_dir(name))

def make_workers(cfg):
    """Configured machines, in scheduling preference order (remote GPUs first)."""
    ws = [RemoteWorker(spec) for spec in cfg.get("workers") or []]
    if cfg.get("local_training", True):
        ws.append(LocalWorker())
    return {w.name: w for w in ws}


def wait_for_mac_memory(cfg, status, job, beat):
    """Pose refinement uses several GB of RAM; if the Mac is also training, wait for headroom."""
    need = int(cfg.get("snapshot_min_free_pct", 25))
    while True:
        mac = status.jobs.get("mac")
        if not mac or mac.get("phase") not in ("training", "paused"):
            return
        free = memory_free_pct()
        if free >= need:
            return
        job["phase"] = f"waiting for Mac memory to refine poses ({free}% free, needs {need}%)"
        beat()
        time.sleep(30)


def process(cfg, paths, state, status, item, beat, worker=None):
    worker = worker or LocalWorker()
    name = item.name
    job_dir = paths.jobs / name
    job_dir.mkdir(parents=True, exist_ok=True)
    job = {"name": name, "phase": "downloading from iCloud", "step": None,
           "kind": item.kind, "frames": item.frames, "thumb": item.thumb,
           "file": item.path.name, "quality": item.quality, "photo_px": item.photo_px,
           "settings": effective_settings(cfg, status.overrides.get(name)), "worker": worker.name}
    for old in paths.previews.glob(f"{name}_step*.jpg"):  # from an earlier run
        old.unlink()
    status.jobs[worker.name] = job
    beat(force=True)
    t0 = time.time()
    imessage(cfg, f"📥 {name}: received, starting"
                  + (f". {quality_text(item.quality)}" if item.quality else ""))
    try:
        local_zip = fetch_local(item, job_dir, beat)
        if local_zip:
            job["phase"] = "unpacking"
            beat()
            unpack(local_zip, job_dir / "data")
        colmap = find_colmap(job_dir / "data")
        sensor = find_sensor_data(job_dir / "data")
        if not job.get("quality"):
            job["quality"] = find_capture_quality(job_dir / "data")
        out_dir = job_dir / "out"
        job["out_dir"], job["job_dir"] = str(out_dir), str(job_dir)
        pending = lambda: effective_settings(load_config(), status.overrides.get(name))
        refined = None   # (folder, stats), computed once per job
        warnings = []
        job["trainer"] = "gsplat" if job["settings"].get("trainer") == "gsplat" and worker.gsplat else "brush"
        if job["settings"].get("trainer") == "gsplat" and not worker.gsplat:
            warnings.append(f"gsplat needs a CUDA worker; trained with Brush on {worker.label}")
        while True:
            source = colmap
            if job["settings"].get("refine_poses"):
                if refined is None:
                    remote_refine = worker.remote and worker.can_refine()
                    if worker.remote and not remote_refine:   # COLMAP on the Mac: don't squeeze a Mac training job
                        wait_for_mac_memory(cfg, status, job, beat)
                    try:
                        refined = refined_model(cfg, colmap, job_dir / "refine", beat, job,
                                                worker if remote_refine else None)
                    except Exception as e:
                        log(f"pose refinement failed for {name}: {e}")
                        warnings.append(f"pose refinement skipped: {str(e)[:120]}")
                        refined = (None, None)
                    warnings += job.pop("warnings", [])   # e.g. refined on the Mac after all
                if refined[0]:
                    source = refined[0]
                job["refine"] = refined[1]
            if worker.remote:
                final = worker.train(cfg, paths, job, source, out_dir, beat, pending, sensor)
            else:
                final = train(cfg, paths, job, source, out_dir, beat, pending)
            if final is not RESTART:
                break
            for old in paths.previews.glob(f"{name}_step*.jpg"):
                old.unlink()
            job.pop("preview", None)
            job["settings"] = pending()
            imessage(cfg, f"🔁 {name}: restarting with new settings "
                          f"({settings_summary(job['settings'])})")

        # Pruning and the output flip are read at the end, so they can change while training runs.
        trained = final
        prune_pct = int(pending().get("prune_pct") or 0)
        job["settings"]["prune_pct"] = prune_pct
        if prune_pct:
            try:
                final, job["prune"] = prune_result(cfg, status, job, worker, source, sensor,
                                                   final, prune_pct, out_dir, beat)
            except Exception as e:
                log(f"pruning failed for {name}: {e}")
                warnings.append(f"prune: {str(e)[:120]}")
        flip = bool(pending().get("flip_up"))
        job["settings"]["flip_up"] = int(flip)
        want_view = bool(pending().get("start_view"))
        job["settings"]["start_view"] = int(want_view)
        view = start_view_pose(source, flip, cfg.get("start_view_forward", "-z")) if want_view else None
        result_ply, cerrors = final, list(warnings)
        if flip or view:
            job["phase"] = "orienting"
            beat()
            oriented = job_dir / "compressed" / f"{name}.ply"
            oriented.parent.mkdir(parents=True, exist_ok=True)
            err = orient_ply(cfg, final, oriented, flip, view, beat)
            if err and view:   # e.g. no gq_spz: still flip
                cerrors.append(f"start view: {err}")
                view = None
                err = orient_ply(cfg, final, oriented, flip, None, beat) if flip else None
            if err:
                cerrors.append(f"flip: {err}")
                flip = False
            elif flip or view:
                result_ply = oriented
        if view:
            job["start_view"] = {"photo": view[0], "origin": view[1], "yaw_degrees": view[2],
                                 "forward": cfg.get("start_view_forward", "-z")}
        job["phase"] = "compressing"
        beat()
        versions = spz_versions(pending())
        for v in (3, 4):
            job["settings"][f"spz_v{v}"] = int(v in versions)
        sizes, more = compress(cfg, result_ply, job_dir / "compressed", name, beat, versions=versions)
        cerrors += more

        job["phase"] = "uploading result to iCloud"
        beat()
        dest = paths.outbox / name
        if cfg.get("keep_previous_results") and dest.is_dir() and any(
                f.name != "previous" for f in dest.iterdir()):
            prev = dest / "previous"
            shutil.rmtree(prev, ignore_errors=True)
            prev.mkdir()
            for f in list(dest.iterdir()):
                if f.name != "previous":
                    shutil.move(str(f), prev / f.name)
            log(f"kept earlier result of {name} in {prev}")
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(result_ply, dest / f"{name}.ply")
        for fmt in sizes:   # built locally, copied whole, so the outbox never sees partial files
            shutil.copyfile(job_dir / "compressed" / f"{name}.{fmt}", dest / f"{name}.{fmt}")
        shutil.copyfile(job_dir / "train.log", dest / "train.log")
        if job.get("preview"):
            shutil.copyfile(paths.previews / job["preview"], dest / "preview.jpg")
        for p in out_dir.glob("export_*.ply"):   # keep only the final (+ unpruned) result locally
            if p not in (final, trained):
                p.unlink()
        shutil.move(str(item.path), paths.done / item.path.name)
        dur = fmt_dur(time.time() - t0)
        size_mb = (dest / f"{name}.ply").stat().st_size / 1e6
        files = f"{size_mb:.0f} MB ply" + "".join(f" · {b / 1e6:.0f} MB {f}" for f, b in sizes.items())
        if job.get("prune"):
            pr = job["prune"]
            files += (f" · pruned to {pr['splats_out']:,} splats ({prune_pct}% removed, PSNR "
                      f"{pr['before']['psnr']:.1f} → {(pr.get('finetuned') or pr['pruned'])['psnr']:.1f})")
        job["files"] = {"ply": int(size_mb * 1e6), **sizes}
        write_meta(dest, job, job["files"], flip)
        try:
            shared = share_result(cfg, dest)
            if shared:
                log(f"shared {', '.join(shared)} to {cfg['share_dir']}")
        except Exception as e:
            cerrors.append(f"share: {e}")
        record(state, job, "done", dur, files + ("  (" + "; ".join(cerrors) + ")" if cerrors else ""))
        log(f"done: {name} in {dur}")
        img = None
        if job.get("preview"):
            img = paths.attach / job["preview"]
            shutil.copyfile(paths.previews / job["preview"], img)
        imessage(cfg, f"✅ {name}: done in {dur} on {worker.label}. In the outbox: {files}"
                      + (f". Compression problem: {'; '.join(cerrors)}" if cerrors else ""), img)
    except Exception as e:
        if not item.path.exists():
            # Pulled out of the inbox (moved, renamed, re-zipped): not a failure.
            log(f"removed from inbox while processing, dropped: {name}")
            shutil.rmtree(job_dir, ignore_errors=True)
            return
        log(f"FAILED: {name}: {e}\n{traceback.format_exc()}")
        dur = fmt_dur(time.time() - t0)
        if item.path.exists():
            shutil.move(str(item.path), paths.failed / item.path.name)
        record(state, job, "FAILED", dur, str(e)[:160])
        imessage(cfg, f"❌ {name}: failed after {dur}: {e}")
    finally:
        try:
            worker.finish(name)
        except Exception as e:
            log(f"cleanup on {worker.label} failed: {e}")
        status.overrides.clear(name)
        status.jobs.pop(worker.name, None)
        beat(force=True)


def find_sensor_data(data_dir):
    """SplatKing's per-photo LiDAR depth/confidence folder, if the capture has one."""
    for p in sorted(Path(data_dir).glob("**/sensor_data")):
        if p.is_dir() and any(p.glob("*_depth.bin")):
            return p
    return None


def prune_result(cfg, status, job, worker, colmap_dir, sensor, ply, pct, out_dir, beat):
    """Prune on the job's own worker when it can (no re-upload), else on any online
    CUDA worker. Returns (pruned .ply, report)."""
    if worker.remote and worker.gsplat:
        pw = worker
    else:
        pw = next((w for w in status.workers.values() if w.remote and w.gsplat and w.online), None)
        if pw is None:
            raise RuntimeError("needs an online CUDA worker with gsplat")
    return pw.prune(cfg, job, colmap_dir, sensor, ply, pct, out_dir, beat, trained_here=pw is worker)


DOWNLOAD_SUFFIXES = (".ply", ".v3.spz", ".v4.spz", ".spz")


def result_downloads(paths, name):
    """[{"file", "size"}] for the result files the dashboard offers for download."""
    out = []
    for suffix in DOWNLOAD_SUFFIXES:
        f = paths.outbox / name / f"{name}{suffix}"
        try:
            if f.is_file():
                out.append({"file": f.name, "size": f.stat().st_size})
        except OSError:
            pass
    return out


def download_path(paths, name, fname):
    """The outbox file behind /download/<name>/<file>, or None if it isn't one we offer."""
    if not name or "/" in name or "\\" in name or name.startswith(".") or fname not in [
            f"{name}{s}" for s in DOWNLOAD_SUFFIXES]:
        return None
    f = paths.outbox / name / fname
    try:
        return f if f.is_file() and f.resolve().parent == (paths.outbox / name).resolve() else None
    except OSError:
        return None


def history_id(h):
    """Stable id of a history entry (for dismissing it from the dashboard)."""
    return str(h.get("finished_ts") or f"{h.get('name')}|{h.get('finished')}")


def dismiss_history(state, entry_id):
    """Hide an entry from Recent. It stays in state.json, so time estimates still use it."""
    with STATE_LOCK:
        for h in state.get("history", []):
            if history_id(h) == entry_id:
                h["dismissed"] = True
                save_state(state)
                return
    raise ValueError("no such history entry")


def record(state, job, result, duration, detail):
    with STATE_LOCK:
        state.setdefault("history", []).append({
            "name": job["name"], "result": result, "duration": duration, "detail": detail,
            "finished": f"{dt.datetime.now():%m-%d %H:%M}", "finished_ts": time.time(),
            "preview": job.get("preview"), "thumb": job.get("thumb"),
            "kind": job.get("kind"), "frames": job.get("frames"),
            "settings": job.get("settings"), "train_seconds": job.get("train_seconds"),
            "worker": job.get("worker"), "trainer": job.get("trainer"),
            "file": job.get("file"),
            "files": job.get("files"), "quality": job.get("quality")})
        state["history"] = state["history"][-50:]
        save_state(state)


# ---------------------------------------------------------------- snapshots

def memory_free_pct():
    r = subprocess.run(["memory_pressure"], capture_output=True, text=True)
    m = re.search(r"free percentage: (\d+)%", r.stdout)
    return int(m.group(1)) if m else 0


class Snapshots:
    """Mid-training snapshots. request() hard-links the latest checkpoint aside (instant,
    no extra disk or memory); a worker thread turns queued snapshots into normal results
    (<name>-step<N>) only when that can't squeeze the running training job."""

    def __init__(self, paths, state, status):
        self.paths, self.state, self.status = paths, state, status
        self.wake = threading.Event()
        with STATE_LOCK:
            for snap in state.setdefault("snapshots", []):
                if snap["state"] == "processing":   # interrupted by a restart
                    snap["state"] = "queued"

    def for_job(self, name):
        with STATE_LOCK:
            return [dict(s) for s in self.state.get("snapshots", []) if s["name"] == name][-6:]

    def pending(self):
        with STATE_LOCK:
            return sum(s["state"] in ("queued", "processing") for s in self.state.get("snapshots", []))

    def request(self, job):
        w = self.status.workers.get(job.get("worker"))
        if w and w.remote:
            return self.request_remote(job, w)
        out_dir = Path(job.get("out_dir") or "")
        step = latest_step(out_dir, "export") if out_dir.is_dir() else None
        if not step:
            raise ValueError("no checkpoint yet (Brush saves one every checkpoint_every steps)")
        with STATE_LOCK:
            if any(s["name"] == job["name"] and s["step"] == step and s["state"] != "failed"
                   for s in self.state["snapshots"]):
                raise ValueError(f"step {step:,} is already snapshotted; wait for the next checkpoint")
        keep = Path(job["job_dir"]) / "snapshots" / f"export_{step}.ply"
        keep.parent.mkdir(parents=True, exist_ok=True)
        if not keep.exists():
            try:
                os.link(out_dir / f"export_{step}.ply", keep)   # same volume: instant
            except OSError:
                shutil.copyfile(out_dir / f"export_{step}.ply", keep)
        preview = f"{job['name']}_step{step:05d}.jpg"
        snap = {"name": job["name"], "step": step, "total": job.get("total"), "src": str(keep),
                "state": "queued", "requested": time.time(), "error": None,
                "settings": dict(job.get("settings") or {}), "frames": job.get("frames"),
                "quality": job.get("quality"), "thumb": job.get("thumb"),
                "preview": preview if (self.paths.previews / preview).exists() else job.get("preview")}
        with STATE_LOCK:
            self.state["snapshots"].append(snap)
            self.state["snapshots"] = self.state["snapshots"][-30:]
            save_state(self.state)
        log(f"snapshot queued: {job['name']} step {step}")
        self.wake.set()
        return step

    def request_remote(self, job, w):
        """Remote job: copy the latest checkpoint aside on the remote machine (so the
        job's own cleanup can't delete it); the worker downloads it when processing."""
        step = w.latest_export(job["name"])
        if not step:
            raise ValueError("no checkpoint yet (Brush saves one every checkpoint_every steps)")
        with STATE_LOCK:
            if any(s["name"] == job["name"] and s["step"] == step and s["state"] != "failed"
                   for s in self.state["snapshots"]):
                raise ValueError(f"step {step:,} is already snapshotted; wait for the next checkpoint")
        remote_path = w.keep_export(job["name"], step)
        preview = f"{job['name']}_step{step:05d}.jpg"
        snap = {"name": job["name"], "step": step, "total": job.get("total"),
                "src": f"remote:{w.name}:{remote_path}", "state": "queued",
                "requested": time.time(), "error": None,
                "settings": dict(job.get("settings") or {}), "frames": job.get("frames"),
                "quality": job.get("quality"), "thumb": job.get("thumb"),
                "preview": preview if (self.paths.previews / preview).exists() else job.get("preview")}
        with STATE_LOCK:
            self.state["snapshots"].append(snap)
            self.state["snapshots"] = self.state["snapshots"][-30:]
            save_state(self.state)
        log(f"snapshot queued: {job['name']} step {step} (on {w.label})")
        self.wake.set()
        return step

    def safe_to_run(self, cfg):
        c = self.status.jobs.get("mac")
        if not c:
            return True, ""
        if c.get("phase") in ("compressing", "orienting") or "refining" in str(c.get("phase")):
            return False, "waiting for the running capture's own processing"
        free = memory_free_pct()
        need = int(cfg.get("snapshot_min_free_pct", 25))
        if free >= need:
            return True, ""
        return False, f"waiting for memory ({free}% free, needs {need}%) or for training to finish"

    def run_forever(self):
        while True:
            self.wake.wait(20)
            self.wake.clear()
            try:
                cfg = load_config()
                with STATE_LOCK:
                    queued = [s for s in self.state.get("snapshots", []) if s["state"] == "queued"]
                if not queued:
                    continue
                ok, why = self.safe_to_run(cfg)
                with STATE_LOCK:
                    for s in queued:
                        s["waiting"] = why
                if ok:
                    self.process(cfg, queued[0])
            except Exception as e:
                log(f"snapshot worker error: {e}\n{traceback.format_exc()}")

    def process(self, cfg, snap):
        with STATE_LOCK:
            snap["state"], snap["waiting"] = "processing", ""
            save_state(self.state)
        label = f"{snap['name']}-step{snap['step']}"
        t0 = time.time()
        work = self.paths.jobs / snap["name"] / "snapshots" / label
        try:
            if snap["src"].startswith("remote:"):   # fetch the checkpoint from the worker first
                _, wname, rpath = snap["src"].split(":", 2)
                w = self.status.workers.get(wname)
                if not w:
                    raise RuntimeError(f"machine {wname} is not configured any more")
                local = self.paths.jobs / snap["name"] / "snapshots" / f"export_{snap['step']}.ply"
                local.parent.mkdir(parents=True, exist_ok=True)
                w.download(rpath, local)
                w.remove(rpath)
                with STATE_LOCK:
                    snap["src"] = str(local)
                    save_state(self.state)
            src = Path(snap["src"])
            if not src.exists():
                raise RuntimeError("checkpoint file is gone")
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            flip = bool(snap["settings"].get("flip_up"))
            model = trainable_model(self.paths.jobs / snap["name"]) if snap["settings"].get("start_view") else None
            view = start_view_pose(model, flip, cfg.get("start_view_forward", "-z")) if model else None
            # Pruning without fine-tuning: the GPU is busy training, so score splats by opacity
            # and size only (gq_spz --keep) instead of running the full pruning stage.
            pct = int(snap["settings"].get("prune_pct") or 0)
            keep = 1 - pct / 100 if 0 < pct < 100 else 1.0
            ply, errors = src, []
            if keep < 1:
                err = orient_ply(cfg, src, work / f"{label}.ply", flip, view, keep=keep)
                if err:
                    errors.append(f"prune: {err}")
                    keep = 1.0
                else:
                    ply = work / f"{label}.ply"
            if keep == 1 and (flip or view):
                err = orient_ply(cfg, src, work / f"{label}.ply", flip, view)
                if err and view:
                    errors.append(f"start view: {err}")
                    view = None
                    err = orient_ply(cfg, src, work / f"{label}.ply", flip, None) if flip else None
                if err:
                    errors.append(f"flip: {err}")
                    flip = False
                elif flip or view:
                    ply = work / f"{label}.ply"
            versions = spz_versions(snap["settings"]) if any(k.startswith("spz_v") for k in snap["settings"]) else None
            sizes, more = compress(cfg, ply, work, label, lambda: None, versions=versions)
            errors += more
            dest = self.paths.outbox / label
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ply, dest / f"{label}.ply")
            for fmt in sizes:
                shutil.copyfile(work / f"{label}.{fmt}", dest / f"{label}.{fmt}")
            if snap.get("preview") and (self.paths.previews / snap["preview"]).exists():
                shutil.copyfile(self.paths.previews / snap["preview"], dest / "preview.jpg")
            files = {"ply": (dest / f"{label}.ply").stat().st_size, **sizes}
            job = {"name": label, "settings": snap["settings"], "frames": snap["frames"],
                   "quality": snap["quality"], "preview": snap.get("preview"), "thumb": snap.get("thumb"),
                   "files": files}
            if view:
                job["start_view"] = {"photo": view[0], "origin": view[1], "yaw_degrees": view[2],
                                     "forward": cfg.get("start_view_forward", "-z")}
            write_meta(dest, job, files, flip)
            meta = json.loads((dest / "meta.json").read_text())
            meta["snapshot"] = {"of": snap["name"], "step": snap["step"], "total_steps": snap["total"],
                                "quick_prune_pct": pct if keep < 1 else 0}
            (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            share_result(cfg, dest)
            detail = (f"step {snap['step']:,}/{(snap['total'] or 0):,} · "
                      + (f"quick-pruned {pct}% · " if keep < 1 else "")
                      + f"{files['ply'] / 1e6:.0f} MB ply" + "".join(f" · {b / 1e6:.0f} MB {f}" for f, b in sizes.items())
                      + (f"  ({'; '.join(errors)})" if errors else ""))
            record(self.state, job, "snapshot", fmt_dur(time.time() - t0), detail)
            with STATE_LOCK:
                snap["state"] = "done"
                save_state(self.state)
            src.unlink(missing_ok=True)
            shutil.rmtree(work, ignore_errors=True)
            log(f"snapshot done: {label}")
            img = None
            if snap.get("preview") and (self.paths.previews / snap["preview"]).exists():
                img = self.paths.attach / snap["preview"]
                shutil.copyfile(self.paths.previews / snap["preview"], img)
            imessage(cfg, f"📸 {label}: snapshot ready ({detail})", img)
        except Exception as e:
            log(f"snapshot failed: {label}: {e}\n{traceback.format_exc()}")
            with STATE_LOCK:
                snap["state"], snap["error"] = "failed", str(e)[:160]
                save_state(self.state)


def current_rate(job, state, worker=None, trainer=None):
    """Seconds per step per megapixel-of-training-resolution, from the running job
    if it has progress, else the most recent finished job that recorded it. With worker
    (and trainer), only runs on that machine count, preferring the same trainer; runs from
    before history recorded a machine count as the Mac's, with Brush."""
    def norm(sec_per_step, st):
        return sec_per_step / ((st["max_resolution"] / 1000) ** 2)
    if job and job.get("step") and job.get("settings") and job.get("train_started"):
        if worker is None or (job.get("worker") == worker and job.get("trainer", "brush") == (trainer or "brush")):
            return norm((time.time() - job["train_started"]) / job["step"], job["settings"])
    done = [h for h in state.get("history", [])
            if h.get("result") == "done" and h.get("train_seconds") and h.get("settings")]
    if worker is not None:
        mine = [h for h in done if (h.get("worker") or "mac") == worker]
        same = [h for h in mine if (h.get("trainer") or "brush") == (trainer or "brush")]
        done = same or mine
    for h in reversed(done):
        return norm(h["train_seconds"] / h["settings"]["total_steps"], h["settings"])
    return None


def estimate_seconds(st, frames, rate):
    if not rate:
        return None
    return rate * st["total_steps"] * (st["max_resolution"] / 1000) ** 2


def cleanup_jobs(paths, days):
    cutoff = time.time() - days * 86400
    protected = {paths.outbox.resolve(), paths.inbox.resolve(), paths.status.resolve()}
    for d in paths.jobs.iterdir():
        if d.resolve() in protected or any(p in d.resolve().parents for p in protected):
            continue
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            log(f"cleaned local job dir {d.name}")


SCRIPT = Path(__file__).resolve()
SCRIPT_MTIME = SCRIPT.stat().st_mtime


def can_take(cfg, worker, ov, any_gsplat):
    """Whether a worker may start a queued capture: honours "Run on", and keeps gsplat
    captures for workers that have gsplat (if there are any; pinned ones fall back to Brush)."""
    run_on = ov.get("run_on", "auto")
    if run_on not in ("auto", worker.name):
        return False
    if run_on == "auto" and any_gsplat and not worker.gsplat:
        return effective_settings(cfg, ov).get("trainer") != "gsplat"
    return True


def reload_if_changed():
    """When idle, re-exec ourselves if this script was edited, so code changes
    never need a manual restart that would kill a running training job."""
    if SCRIPT.stat().st_mtime == SCRIPT_MTIME:
        return
    r = subprocess.run([sys.executable, "-I", "-m", "py_compile", str(SCRIPT)],
                       capture_output=True, text=True)
    if r.returncode:
        log(f"script changed but doesn't compile; not reloading: {r.stderr.strip()[:200]}")
        globals()["SCRIPT_MTIME"] = SCRIPT.stat().st_mtime
        return
    log("script changed; reloading")
    os.execv(sys.executable, [sys.executable, "-I", "-u"] + sys.argv)


def main():
    APP_HOME.mkdir(parents=True, exist_ok=True)
    try:
        cfg = load_config()
    except ConfigError as e:
        log(f"config error: {e}")
        sys.exit(78)  # EX_CONFIG; launchd retries after ThrottleInterval
    paths = Paths(cfg)
    state = load_state()
    inbox = Inbox(paths)
    overrides = Overrides()
    status = Status(paths, state, inbox, overrides)
    status.snapshots = Snapshots(paths, state, status)
    status.workers = make_workers(cfg)
    threading.Thread(target=status.snapshots.run_forever, daemon=True).start()

    def monitor():   # machine status for the dashboard and the scheduler
        while True:
            for w in list(status.workers.values()):
                try:
                    w.refresh()
                except Exception as e:
                    log(f"status of {w.label}: {e}")
            time.sleep(20)
    threading.Thread(target=monitor, daemon=True).start()
    start_dashboard(cfg, paths, status, overrides)
    log(f"GaussQueue started, watching {paths.inbox}; machines: "
        + ", ".join(w.label for w in status.workers.values()))
    last_beat = [0.0]
    beat_lock = threading.Lock()
    running = {}   # worker name -> (thread, capture name)

    def beat(force=False):
        """Called often from every job thread's wait loops: keep the inbox (thumbnails!)
        and status fresh even while jobs are training."""
        if not force and time.time() - last_beat[0] < 2:
            return
        if not beat_lock.acquire(blocking=force):
            return   # another thread is already refreshing
        try:
            last_beat[0] = time.time()
            c = load_config()
            try:
                busy = status.busy_names() | {n for _, n in list(running.values())}
                inbox.tick(c, busy)
            except Exception as e:
                log(f"inbox error: {e}")
            status.write(c)
        finally:
            beat_lock.release()

    def run_job(cfg, item, worker):
        try:
            process(cfg, paths, state, status, item, beat, worker)
        finally:
            running.pop(worker.name, None)

    while True:
        try:
            cfg = load_config()
            last_beat[0] = 0
            beat(force=True)
            for name in [n for n, (t, _) in list(running.items()) if not t.is_alive()]:
                running.pop(name, None)
            # Reload edited code before picking up more work: with a backlog the queue is
            # never idle after scheduling, so checking only afterwards would never reload.
            if not running and not status.snapshots.pending():
                reload_if_changed()
            claimed = {n for _, n in list(running.values())}
            any_gsplat = any(w.gsplat for w in status.workers.values())
            for w in list(status.workers.values()):   # remote GPUs first, then the Mac
                if w.name in running or (w.remote and not w.online):
                    continue
                item = next((i for i in inbox.ready(overrides) if i.name not in claimed
                             and can_take(cfg, w, overrides.get(i.name), any_gsplat)), None)
                if item is None:
                    continue
                t = threading.Thread(target=run_job, args=(cfg, item, w), daemon=True)
                running[w.name] = (t, item.name)
                claimed.add(item.name)
                log(f"starting {item.name} on {w.label}")
                t.start()
            if not running:
                cleanup_jobs(paths, cfg["keep_work_days"])
        except Exception as e:
            log(f"loop error: {e}\n{traceback.format_exc()}")
        time.sleep(cfg["poll_seconds"])


def flip_results(folders):
    """`--flip DIR...`: rotate existing results 180° about X in place (the .ply,
    then regenerate the .spz next to it) and toggle meta.json. Flipping
    twice restores the original orientation."""
    cfg = load_config()
    tmp = APP_HOME / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    status = 0
    for folder in map(Path, folders):
        name = folder.name
        ply = folder / f"{name}.ply"
        if not ply.exists():
            print(f"✗ {folder}: no {name}.ply")
            status = 1
            continue
        for f in folder.iterdir():  # make sure iCloud placeholders are real files
            if is_dataless(f):
                request_download(f)
        while any(is_dataless(f) for f in folder.iterdir()):
            time.sleep(2)
        meta_path = folder / "meta.json"
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {"name": name, "rotation_xyz_degrees": [0, 0, 0]}
        print(f"↻ {name}: rotating {ply.stat().st_size / 1e6:.0f} MB .ply…", flush=True)
        out = tmp / f"{name}.ply"
        err = flip_ply(cfg, ply, out)
        if err:
            print(f"✗ {name}: {err}")
            status = 1
            continue
        # rebuild whichever SPZ files the result has (the configured ones if it has none)
        found = [v for v in (3, 4) if (folder / f"{name}.v{v}.spz").exists()]
        legacy = (folder / f"{name}.spz").exists()
        if not found and not legacy:
            found = [int(v) for v in cfg.get("spz_versions") or [3]]
        formats = (["spz"] if found else []) + ([("spz", 3)] if legacy else [])
        sizes, errors = compress(cfg, out, tmp, name, lambda: None, formats, found)
        shutil.copyfile(out, ply)   # replace originals only after everything built
        out.unlink()
        for fmt in sizes:
            shutil.copyfile(tmp / f"{name}.{fmt}", folder / f"{name}.{fmt}")
            (tmp / f"{name}.{fmt}").unlink()
        x = (meta.get("rotation_xyz_degrees", [0, 0, 0])[0] + 180) % 360
        meta["rotation_xyz_degrees"] = [x, 0, 0]
        meta["files"] = {"ply": ply.stat().st_size, **sizes}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        share_result(cfg, folder)
        print(f"✓ {name}: now rotated {x}° about X"
              + (f" (problems: {'; '.join(errors)})" if errors else ""))
        status |= bool(errors)
    return status


def check():
    """`--check`: validate config and tools; used by install.sh."""
    ok = True
    try:
        cfg = load_config()
    except ConfigError as e:
        print(f"✗ config: {e}")
        return 1
    print(f"✓ config: {CONFIG_PATH}")
    for k in ("inbox_dir", "outbox_dir", "work_dir", "status_dir", "share_dir"):
        if cfg.get(k):
            print(f"  {k:<11} {cfg[k]}")
    if Path(cfg["work_dir"]).resolve() in Path(cfg["outbox_dir"]).resolve().parents or \
            Path(cfg["outbox_dir"]).resolve() == Path(cfg["work_dir"]).resolve():
        print("✗ outbox_dir is inside work_dir, whose old folders are cleaned up; move it.")
        ok = False
    if "Mobile Documents" in cfg["work_dir"]:
        print("! work_dir is inside iCloud Drive; training files will upload. Use a local folder.")
    if os.access(cfg["brush"], os.X_OK):
        print(f"✓ brush: {cfg['brush']}")
    else:
        print(f"✗ brush not found/executable: {cfg['brush']}")
        ok = False
    if cfg.get("compress") and not native_spz(cfg):
        if os.access(cfg["splat_transform"], os.X_OK):
            print(f"✓ splat-transform: {cfg['splat_transform']} (compress: {', '.join(cfg['compress'])})")
        else:
            print(f"! splat-transform not found ({cfg['splat_transform']}); "
                  "results will be .ply only. install.sh installs it when Node.js is available.")
    if native_spz(cfg):
        print(f"✓ gq_spz (native .spz/flip): {cfg['spz_native']}")
    else:
        print("  gq_spz not built: .spz and flips use SplatTransform (WebAssembly; 4 GB limit "
              "can fail on ~10M-splat captures). install.sh builds it when cmake + a C++ compiler exist.")
    colmap = shutil.which(cfg.get("colmap") or "colmap")
    if colmap:
        print(f"✓ colmap: {colmap}" + (" (refine_poses on by default)" if cfg.get("refine_poses") else ""))
    elif cfg.get("refine_poses"):
        print("! colmap not found (brew install colmap); captures will train on unrefined poses")
    for spec in cfg.get("workers") or []:
        w = RemoteWorker(spec)
        print(f"  worker {w.label}: ssh {w.ssh}, brush" + (", gsplat + pruning" if w.gsplat else ""))
    if cfg.get("trainer") == "gsplat" and not any((s.get("gsplat") or {}).get("python") for s in cfg.get("workers") or []):
        print("! trainer is gsplat but no worker has gsplat; captures will train with Brush")
    for tool in ("brctl", "sips", "caffeinate", "osascript"):
        if not shutil.which(tool):
            print(f"✗ missing macOS tool: {tool}")
            ok = False
    print(f"✓ dashboard: http://127.0.0.1:{cfg['dashboard_port']}")
    print(f"  iMessage: {'to ' + cfg['imessage_to'] if cfg['imessage_to'] else 'off'}")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        sys.exit(check())
    elif len(sys.argv) > 2 and sys.argv[1] == "--flip":
        sys.exit(flip_results(sys.argv[2:]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--notify-test":
        cfg = load_config()
        paths = Paths(cfg)
        img = None
        if len(sys.argv) > 2:
            img = paths.attach / "test_preview.jpg"
            make_jpeg(Path(sys.argv[2]), img, 1280)
        imessage(cfg, "🧪 GaussQueue test: notifications work.", img)
    else:
        main()
