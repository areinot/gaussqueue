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
import datetime as dt
import http.server
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
import zipfile
from pathlib import Path

APP_HOME = Path(os.environ.get("GAUSSQUEUE_HOME",
                               Path.home() / "Library/Application Support/GaussQueue")).expanduser()
CONFIG_PATH = APP_HOME / "config.json"
STATE_PATH = APP_HOME / "state.json"
OVERRIDES_PATH = APP_HOME / "overrides.json"
DASHBOARD_HTML = Path(__file__).resolve().parent / "dashboard.html"
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
    "share_files": ["sog", "preview.jpg", "meta.json"],
    "attach_dir": "~/Pictures/GaussQueue",   # staging for iMessage image attachments
    "brush": str(APP_HOME / "bin/brush_app"),
    # Compressed copies written next to the .ply (needs SplatTransform; [] = off)
    "compress": ["sog", "spz"],
    # Rotate results 180° about X so "up" is up in SuperSplat & co. (SplatKing/COLMAP
    # captures are +Y-up, most splat viewers expect -Y-up). Per-capture override on the dashboard.
    "flip_up": False,
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
}


# Per-item processing settings, editable from the dashboard. "standard" comes
# from config.json, so editing config still changes the default.
TRAIN_KEYS = ("total_steps", "max_resolution", "sh_degree", "max_splats", "subsample_frames")
SETTING_KEYS = TRAIN_KEYS + ("flip_up",)   # flip_up only affects output files
SETTING_LIMITS = {"total_steps": (500, 100_000), "max_resolution": (256, 4096),
                  "sh_degree": (0, 3), "max_splats": (50_000, 20_000_000),
                  "subsample_frames": (1, 20), "flip_up": (0, 1)}
PRESETS = {
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
    out.update({k: ov[k] for k in SETTING_KEYS if k in ov})
    out["base"] = preset
    out["preset"] = preset if not any(k in ov for k in TRAIN_KEYS) else "custom"
    return out


def settings_summary(st):
    return (f"{st['preset']}: {st['total_steps'] / 1000:g}k steps @ {st['max_resolution']}px, "
            f"SH{st['sh_degree']}, <={st['max_splats'] / 1e6:g}M splats"
            + (f", every {st['subsample_frames']} frames" if st["subsample_frames"] > 1 else "")
            + (", flipped" if st.get("flip_up") else ""))


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
              "splat_transform", "share_dir"):
        if cfg.get(k):
            cfg[k] = str(Path(cfg[k]).expanduser())
    _last_good_config = cfg
    return cfg


def load_state():
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {"history": []}


def save_state(state):
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
                "thumb": self.thumb, "thumb_note": self.thumb_note}


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

    def tick(self, cfg, busy_name=None):
        present = {p.name: p for p in self.entries()}
        with self.lock:
            for gone in set(self.items) - set(present):
                del self.items[gone]
            for fname, p in present.items():
                if fname not in self.items:
                    self.items[fname] = InboxItem(p)
                    log(f"inbox: new item {fname}")
        for item in list(self.items.values()):
            if item.name == busy_name:
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

    def order(self, overrides):
        """Queue order: 'run next' items first (latest press first), then oldest."""
        def key(i):
            return (-overrides.get(i.name).get("priority", 0), i.first_seen)
        return sorted(self.items.values(), key=key)

    def ready(self, overrides):
        """Settled, not-held items in queue order."""
        return [i for i in self.order(overrides)
                if i.state == "queued" and not overrides.get(i.name).get("hold")]

    def snapshot(self, cfg, overrides, rate, busy_name=None):
        out = []
        for i in self.order(overrides):
            if i.name == busy_name:
                continue
            ov = overrides.get(i.name)
            d = i.to_json()
            d["settings"] = effective_settings(cfg, ov)
            d["hold"] = bool(ov.get("hold"))
            d["run_next"] = bool(ov.get("priority"))
            d["est_seconds"] = estimate_seconds(d["settings"], d["frames"], rate)
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
        self.current = None   # dict describing the active job
        self.snapshot = {}    # served as JSON by the dashboard

    def write(self, cfg):
        c = self.current
        busy = c["name"] if c else None
        rate = current_rate(c, self.state)
        queue = self.inbox.snapshot(cfg, self.overrides, rate, busy)
        hist, seen = [], set()
        in_inbox = {i.name for i in list(self.inbox.items.values())}
        for h in self.state.get("history", [])[-20:][::-1]:
            h = dict(h)
            h["requeueable"] = (h["name"] not in seen and h["name"] not in in_inbox
                                and find_processed(self.paths, h["name"]) is not None)
            seen.add(h["name"])
            hist.append(h)
        current = None
        if c:
            current = {k: c.get(k) for k in
                       ("name", "phase", "step", "total", "started", "eta", "preview",
                        "kind", "frames", "thumb", "settings", "paused", "paused_at")}
            current["pending_settings"] = effective_settings(cfg, self.overrides.get(c["name"]))
            current["previews"] = sorted(p.name for p in
                                         self.paths.previews.glob(f"{c['name']}_step*.jpg"))
        self.snapshot = {
            "updated": time.time(), "host": socket.gethostname(),
            "current": current, "queue": queue, "history": hist,
            "settings": standard_settings(cfg),
            "inbox_dir": cfg["inbox_dir"],
            "presets": {k: (v or standard_settings(cfg)) for k, v in PRESETS.items()},
            "limits": SETTING_LIMITS,
        }
        self.write_text(current, queue, hist)

    def write_text(self, c, queue, hist):
        now = dt.datetime.now()
        lines = [
            "GaussQueue status",
            f"Updated: {now:%a %b %d %H:%M:%S} on {socket.gethostname()}",
            "(If this time is old, the Mac service is not running.)",
            "",
        ]
        if c:
            lines.append(f"NOW: {c['name']}")
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
        else:
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
                if not any(i.name == name for i in list(status.inbox.items.values())):
                    raise ValueError("no such inbox item")
                cur = status.current
                action = body.get("action")
                if cur and cur["name"] == name:
                    if action in ("pause", "resume"):
                        if cur.get("phase") not in ("training", "paused"):
                            raise ValueError("can only pause while training")
                        if action == "resume" and not cur.get("paused"):
                            raise ValueError("not paused")
                        cur["request"] = action
                        return self.send(200, b'{"ok":true}', "application/json")
                    if action in ("run_next", "hold"):
                        raise ValueError("already processing")
                    output_only = action == "set" and body.get("key") == "flip_up"
                    if not cur.get("paused") and not output_only:
                        raise ValueError("pause training before changing its settings")
                elif action in ("pause", "resume"):
                    raise ValueError("only the running capture can be paused")
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


RESTART = object()  # train() result: settings changed while paused, run again


def same_settings(a, b):
    return all(a[k] == b[k] for k in TRAIN_KEYS)


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
    rotate = capture_rotation(colmap_dir.parent)
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


COMPRESS_FORMATS = ("sog", "spz")
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


def compress(cfg, ply, out_dir, name, beat, formats=None):
    """Write <name>.<fmt> for each configured format with SplatTransform.
    Returns ({fmt: bytes}, [error strings]); failures never fail the job."""
    if formats is None:
        formats = [f for f in cfg.get("compress") or [] if f in COMPRESS_FORMATS]
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes, errors = {}, []
    for fmt in formats:
        dst = out_dir / f"{name}.{fmt}"
        # -N drops NaN/degenerate splats, which some viewers choke on
        err = splat_transform(cfg, [ply, "-N", dst], dst, beat)
        if err:
            errors.append(f"{fmt}: {err}")
        else:
            sizes[fmt] = dst.stat().st_size
    return sizes, errors


def flip_ply(cfg, src, dst, beat=lambda: None):
    """Rotate a splat 180° about X (turns a +Y-up scene -Y-up and vice versa)."""
    return splat_transform(cfg, [src, "-r", FLIP, "-N", dst], dst, beat)


def share_result(cfg, result_dir):
    """Copy the phone-friendly files of one result into share_dir/<name>/.
    Entries in share_files are either a format ("sog" -> <name>.sog) or a filename."""
    if not cfg.get("share_dir"):
        return []
    name = result_dir.name
    dest = Path(cfg["share_dir"]) / name
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for entry in cfg.get("share_files") or []:
        fname = f"{name}.{entry}" if "." not in entry else entry
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
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def process(cfg, paths, state, status, item, beat):
    name = item.name
    job_dir = paths.jobs / name
    job_dir.mkdir(parents=True, exist_ok=True)
    job = {"name": name, "phase": "downloading from iCloud", "step": None,
           "kind": item.kind, "frames": item.frames, "thumb": item.thumb,
           "file": item.path.name,
           "settings": effective_settings(cfg, status.overrides.get(name))}
    for old in paths.previews.glob(f"{name}_step*.jpg"):  # from an earlier run
        old.unlink()
    status.current = job
    beat()
    t0 = time.time()
    imessage(cfg, f"📥 {name}: received, starting")
    try:
        local_zip = fetch_local(item, job_dir, beat)
        if local_zip:
            job["phase"] = "unpacking"
            beat()
            unpack(local_zip, job_dir / "data")
        colmap = find_colmap(job_dir / "data")
        out_dir = job_dir / "out"
        pending = lambda: effective_settings(load_config(), status.overrides.get(name))
        while True:
            final = train(cfg, paths, job, colmap, out_dir, beat, pending)
            if final is not RESTART:
                break
            for old in paths.previews.glob(f"{name}_step*.jpg"):
                old.unlink()
            job.pop("preview", None)
            job["settings"] = pending()
            imessage(cfg, f"🔁 {name}: restarting with new settings "
                          f"({settings_summary(job['settings'])})")

        # Output flip is read at the end, so it can be changed while training runs.
        flip = bool(pending().get("flip_up"))
        job["settings"]["flip_up"] = int(flip)
        result_ply, cerrors = final, []
        if flip:
            job["phase"] = "flipping"
            beat()
            flipped = job_dir / "compressed" / f"{name}.ply"
            flipped.parent.mkdir(parents=True, exist_ok=True)
            err = flip_ply(cfg, final, flipped, beat)
            if err:
                cerrors.append(f"flip: {err}")
                flip = False
            else:
                result_ply = flipped
        job["phase"] = "compressing"
        beat()
        sizes, more = compress(cfg, result_ply, job_dir / "compressed", name, beat)
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
        for p in out_dir.glob("export_*.ply"):   # keep only the final locally
            if p != final:
                p.unlink()
        shutil.move(str(item.path), paths.done / item.path.name)
        dur = fmt_dur(time.time() - t0)
        size_mb = (dest / f"{name}.ply").stat().st_size / 1e6
        files = f"{size_mb:.0f} MB ply" + "".join(f" · {b / 1e6:.0f} MB {f}" for f, b in sizes.items())
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
        imessage(cfg, f"✅ {name}: done in {dur}. In the outbox: {files}"
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
        status.overrides.clear(name)
        status.current = None
        beat()


def record(state, job, result, duration, detail):
    state.setdefault("history", []).append({
        "name": job["name"], "result": result, "duration": duration, "detail": detail,
        "finished": f"{dt.datetime.now():%m-%d %H:%M}", "finished_ts": time.time(),
        "preview": job.get("preview"), "thumb": job.get("thumb"),
        "kind": job.get("kind"), "frames": job.get("frames"),
        "settings": job.get("settings"), "train_seconds": job.get("train_seconds"),
        "file": job.get("file"),
        "files": job.get("files")})
    state["history"] = state["history"][-50:]
    save_state(state)


def current_rate(job, state):
    """Seconds per step per megapixel-of-training-resolution, from the running job
    if it has progress, else the most recent finished job that recorded it."""
    def norm(sec_per_step, st):
        return sec_per_step / ((st["max_resolution"] / 1000) ** 2)
    if job and job.get("step") and job.get("settings") and job.get("train_started"):
        return norm((time.time() - job["train_started"]) / job["step"], job["settings"])
    for h in reversed(state.get("history", [])):
        if h.get("result") == "done" and h.get("train_seconds") and h.get("settings"):
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
    start_dashboard(cfg, paths, status, overrides)
    log(f"GaussQueue started, watching {paths.inbox}")
    last_beat = [0.0]

    def beat(force=False):
        """Called often from every wait loop: keep the inbox (thumbnails!) and
        status fresh even while a job is training."""
        if not force and time.time() - last_beat[0] < 2:
            return
        last_beat[0] = time.time()
        c = load_config()
        try:
            inbox.tick(c, status.current["name"] if status.current else None)
        except Exception as e:
            log(f"inbox error: {e}")
        status.write(c)

    while True:
        try:
            cfg = load_config()
            last_beat[0] = 0
            beat()
            ready = inbox.ready(overrides)
            if ready:
                process(cfg, paths, state, status, ready[0], beat)
                continue  # rescan immediately
            cleanup_jobs(paths, cfg["keep_work_days"])
            reload_if_changed()
        except Exception as e:
            log(f"loop error: {e}\n{traceback.format_exc()}")
        time.sleep(cfg["poll_seconds"])


def flip_results(folders):
    """`--flip DIR...`: rotate existing results 180° about X in place (the .ply,
    then regenerate any .sog/.spz next to it) and toggle meta.json. Flipping
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
        formats = [f for f in COMPRESS_FORMATS
                   if (folder / f"{name}.{f}").exists() or f in (cfg.get("compress") or [])]
        sizes, errors = compress(cfg, out, tmp, name, lambda: None, formats)
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
    if cfg.get("compress"):
        if os.access(cfg["splat_transform"], os.X_OK):
            print(f"✓ splat-transform: {cfg['splat_transform']} (compress: {', '.join(cfg['compress'])})")
        else:
            print(f"! splat-transform not found ({cfg['splat_transform']}); "
                  "results will be .ply only. install.sh installs it when Node.js is available.")
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
