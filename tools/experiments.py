"""Run a batch of training experiments, one variant after another, with progress on the
dashboard (an "Experiments" card per batch) and the queue keeping off that machine until
the batch is done.

  python3 tools/experiments.py plan.json

plan.json:
  {"name": "Shelf box: lens fix",          shown on the card
   "worker": "pc",                          a worker from config.json, or "mac" (this Mac)
   "program": "gsplat",                     "gsplat" (default; the worker's gs scripts), "brush", or
                                            "command" (each variant's "run": a command line run as is
                                            on the worker, e.g. a COLMAP test; nothing is published)
   "out": "D:/experiments/round2",          each variant writes to <out>/<variant name>
   "script": "train.py",                    gsplat: in the gs scripts folder (default train.py)
   "common": ["<data dir>", "--steps", "15000", ...],
   "variants": [{"name": "A", "desc": "depth on", "args": ["--depth-weight", "0.3"]}, ...],
   "note": "optional line for the card",
   "greedy": {...}}                         optional, gsplat: see below

Greedy batches ("greedy": {} or with settings) treat each variant's args as a change to try on
top of the changes kept so far: the first variant is the baseline (unless "start" gives kept
args and "best" their scores), and a change is kept when its held-out PSNR beats the best by
"min_gain" dB (default 0.05), or costs at most "max_loss" dB (default 0.1) while cutting needles
or in-front (floater) share by "count_gain" (default 20%). The card shows kept/dropped per variant.

gsplat runs report steps and held-out scores from train.py's output. Brush runs (Brush is
quiet when not on a terminal) are followed through the eval_<step> folders it writes, so give
them --eval-every; the card then shows the splat count of the final export.

A variant waits while the queue is processing a capture on that machine. Logs go to
$GAUSSQUEUE_HOME/experiments/<batch>/<variant>.log. Ctrl-C stops the running variant.

Each finished variant's final export is brought to this Mac and published to the outbox as
exp-<batch>-<variant>/ (.ply and .v3.spz, flipped right way up like queue results), so the
dashboard's card offers download buttons for it.

  python3 tools/experiments.py --publish <batch json> [--force]   publish the variants of a finished
                                                         batch (--force: again, if already published)
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

APP_HOME = Path(os.environ.get("GAUSSQUEUE_HOME", Path.home() / "Library/Application Support/GaussQueue"))
BATCHES = APP_HOME / "experiments" / "batches"
STEP = re.compile(r"step (\d+)/(\d+) .*?([\d.]+) it/s")
RESULT = re.compile(r"held-out PSNR ([\d.]+) SSIM ([\d.]+)(?: depth error ([\d.]+)%)?")
COUNTS = re.compile(r"(needles|holes|in front) ([\d.]+)%")


def greedy_keep(res, best, g):
    """Whether a greedy step's result beats the best so far (see the module docstring)."""
    if best is None:
        return True
    if res.get("psnr") is None:
        return False
    d = res["psnr"] - best["psnr"]
    if d >= g.get("min_gain", 0.05):
        return True
    cut = any(best.get(k) and res.get(k) is not None and res[k] <= (1 - g.get("count_gain", 0.2)) * best[k]
              for k in ("needles", "in_front"))
    return d >= -g.get("max_loss", 0.1) and cut


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "batch"


class Batch:
    def __init__(self, plan, worker_name):
        self.lock = threading.Lock()
        BATCHES.mkdir(parents=True, exist_ok=True)
        self.path = BATCHES / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug(plan.get('name', 'batch'))}.json"
        self.data = {"name": plan.get("name", "Experiments"), "worker": worker_name, "out": plan.get("out"),
                     "program": plan.get("program", "gsplat"),
                     "note": plan.get("note"), "common": plan.get("common", []), "started": time.time(), "finished": None,
                     "variants": [{"name": v["name"], "desc": v.get("desc"), "state": "queued", "args": v.get("args", [])}
                                  for v in plan["variants"]]}
        self.write()

    def write(self):
        with self.lock:
            self.data["updated"] = time.time()
            self.data["eta"] = self.eta()
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=1))
            tmp.replace(self.path)

    def eta(self):
        """Current variant's end plus the rest at the average duration so far."""
        vs = self.data["variants"]
        took = [v["finished"] - v["began"] for v in vs if v.get("finished") and v.get("began")]
        run = next((v for v in vs if v["state"] == "running"), None)
        if run and run.get("eta"):
            each = sum(took) / len(took) if took else run["eta"] - run["began"]
            rest = sum(1 for v in vs if v["state"] in ("queued", "waiting"))
            return run["eta"] + rest * each
        return None

    def heartbeat(self, stop):
        while not stop.wait(20):
            self.write()


def queue_job_on(worker_name, port):
    """The capture the queue is processing on this machine, if any."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=10) as r:
            st = json.loads(r.read())
    except Exception:
        return None
    return next((j["name"] for j in st.get("jobs", []) if j.get("worker") == worker_name), None)


def quote(a):
    a = str(a)
    return f'"{a}"' if re.search(r'[\s&|<>^]', a) else a


def ply_splats(path):
    try:
        with open(path, "rb") as f:
            m = re.search(rb"element vertex (\d+)", f.read(4096))
        return int(m.group(1)) if m else None
    except OSError:
        return None


def expand(p):
    return Path(os.path.expanduser(str(p)))


def start_view(runner_local, ssh, data):
    """The queue's start view for a training data folder (its first photo's camera, in the
    flipped output's coordinates): gq_spz --origin/--yaw arguments, or [] if unknown."""
    if not data:
        return []
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import gaussqueue
    if runner_local:
        model = Path(data)
    else:
        model = APP_HOME / "tmp" / "start_view"
        (model / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{ssh}:/{data}/sparse/0/images.txt",
                            str(model / "sparse" / "0" / "images.txt")], capture_output=True, text=True, timeout=120)
        if r.returncode:
            return []
    view = gaussqueue.start_view_pose(model, True)
    return ["--origin", *map(str, view[1]), "--yaw", str(view[2])] if view else []


def data_candidates(*arg_lists):
    """Arguments that may be the training data folder (the positional one can come anywhere,
    e.g. in a variant's args): every argument that isn't an option, in order."""
    return [a for args in arg_lists for a in (args or []) if isinstance(a, str) and a and not a.startswith("-")]


def publish(cfg, runner_local, ssh, out, label, data=None):
    """Fetch <out>'s newest export_<step>.ply and write outbox/<label>/<label>.ply + .v3.spz
    (flipped, and moved so viewers open at the first photo's camera, like queue results, when the
    training data folder is found among data: a folder or a list of candidates, the first with a
    COLMAP model wins). Returns [{"file", "size"}] or [] if there's nothing to publish."""
    outbox = expand(cfg.get("outbox_dir") or APP_HOME / "outbox")
    tool = expand(cfg.get("spz_native") or APP_HOME / "bin" / "gq_spz")
    if runner_local:
        exports = sorted(Path(out).glob("export_*.ply"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
        if not exports:
            return []
        src, tmp = exports[-1], None
    else:
        win = str(out).replace("/", "\\")   # cmd's dir reads "/x" as a switch
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", ssh, f'dir /b "{win}\\export_*.ply"'],
                           capture_output=True, text=True, timeout=60)
        names = sorted((n.strip() for n in r.stdout.split() if n.strip().endswith(".ply")),
                       key=lambda n: int(re.sub(r"\D", "", n) or 0))
        if not names:
            return []
        tmp = APP_HOME / "tmp" / f"{label}.ply"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{ssh}:/{out}/{names[-1]}", str(tmp)],
                           capture_output=True, text=True, timeout=3600)
        if r.returncode:
            return []
        src = tmp
    dest = outbox / label
    dest.mkdir(parents=True, exist_ok=True)
    view = []
    for cand in ([data] if isinstance(data, str) else (data or [])):
        try:
            view = start_view(runner_local, ssh, cand)
        except Exception:
            view = []
        if view:
            break
    if data and not view:
        print(f"    {label}: no start view found (published flipped only, not in the queue's frame)", flush=True)
    files = []
    for suffix, extra in ((".ply", []), (".v3.spz", ["--spz-version", "3"])):
        f = dest / f"{label}{suffix}"
        r = subprocess.run([str(tool), str(src), str(f), "--flip"] + view + extra, capture_output=True, text=True)
        if r.returncode == 0 and f.exists():
            files.append({"file": f.name, "size": f.stat().st_size})
    if tmp:
        tmp.unlink(missing_ok=True)
    return files


def arg_value(args, flag):
    try:
        return args[args.index(flag) + 1]
    except (ValueError, IndexError):
        return None


class Runner:
    """Where and how a variant runs: locally or over SSH, gsplat or Brush."""

    def __init__(self, plan, cfg):
        self.plan, self.program = plan, plan.get("program", "gsplat")
        name = plan["worker"]
        self.local = name == "mac"
        if self.local:
            self.worker = {"name": "mac"}
            self.brush = cfg.get("brush") or str(APP_HOME / "bin" / "brush_app")
            self.gsplat = None
        else:
            self.worker = next((w for w in cfg.get("workers", []) if w.get("name") == name), None)
            if not self.worker:
                sys.exit(f"no worker {name!r} in config.json")
            self.brush = self.worker.get("brush")
            self.gsplat = self.worker.get("gsplat")
        if self.program == "gsplat" and not self.gsplat:
            sys.exit(f"{name} has no gsplat set up (gsplat needs a CUDA worker)")
        if self.program == "brush" and not self.brush:
            sys.exit(f"no Brush configured for {name}")
        if self.program == "command" and any(not v.get("run") for v in plan["variants"]):
            sys.exit('program "command" needs a "run" command line in every variant')

    def command(self, variant):
        out = f"{self.plan.get('out', '')}/{variant['name']}"
        if self.program == "command":
            if self.local:
                return ["/bin/sh", "-c", variant["run"] + " 2>&1"], None, out
            return ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", self.worker["ssh"],
                    variant["run"] + " 2>&1"], None, out
        args = self.plan.get("common", []) + variant.get("args", [])
        if self.program == "brush":
            argv = [self.brush] + args + ["--export-path", out]
            cwd = None
        else:
            g = self.gsplat
            argv = [g["python"], "-u", f"{g['scripts']}/{self.plan.get('script', 'train.py')}"] + args + ["--out", out]
            cwd = g["scripts"]
        if self.local:
            Path(out).mkdir(parents=True, exist_ok=True)
            return argv, cwd, out
        pre = f'if not exist "{out}" mkdir "{out}" && ' if self.program == "brush" else ""
        line = pre + (f'cd /d "{cwd}" && ' if cwd else "") + " ".join(quote(a) for a in argv) + " 2>&1"
        return ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", self.worker["ssh"], line], None, out

    def brush_progress(self, out):
        """Newest eval_<step> folder Brush has written (None if none yet)."""
        try:
            if self.local:
                names = [p.name for p in Path(out).glob("eval_*")]
            else:
                r = subprocess.run(["ssh", "-o", "BatchMode=yes", self.worker["ssh"],
                                    f'dir /b /ad "{str(out).replace("/", chr(92))}"'], capture_output=True, text=True, timeout=30)
                names = r.stdout.split()
        except Exception:
            return None
        steps = [int(m.group(1)) for n in names if (m := re.fullmatch(r"eval_(\d+)", n))]
        return max(steps) if steps else None

    def final_splats(self, out, steps):
        if self.local:
            exports = sorted(Path(out).glob("export_*.ply"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
            return ply_splats(exports[-1]) if exports else None
        return None


def main(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    cfg = json.loads((APP_HOME / "config.json").read_text())
    runner = Runner(plan, cfg)
    port = cfg.get("dashboard_port", 8765)
    batch = Batch(plan, runner.worker["name"])
    logs = APP_HOME / "experiments" / batch.path.stem
    logs.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    threading.Thread(target=batch.heartbeat, args=(stop,), daemon=True).start()
    greedy = plan.get("greedy")
    kept = list((greedy or {}).get("start", []))
    best = (greedy or {}).get("best")
    try:
        for v, rec in zip(plan["variants"], batch.data["variants"]):
            if greedy is not None:
                v = dict(v, args=kept + v.get("args", []))
            while (busy := queue_job_on(runner.worker["name"], port)):
                rec.update(state="waiting", waiting=f"the queue is processing {busy}")
                batch.write()
                time.sleep(20)
            rec.update(state="running", began=time.time(), step=None, total=None, eta=None, waiting=None)
            all_args = plan.get("common", []) + v.get("args", [])
            total = arg_value(all_args, "--total-steps") or arg_value(all_args, "--steps")
            rec["total"] = int(total) if total else None
            batch.write()
            argv, cwd, out = runner.command(v)
            print(f"=== {v['name']}: {' '.join(map(str, v.get('args', [])))}", flush=True)
            proc = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            done = threading.Event()
            if runner.program == "brush":   # quiet: follow its eval folders instead
                def follow():
                    while not done.wait(10):
                        step = runner.brush_progress(out)
                        if step and rec.get("total"):
                            rate = step / max(1.0, time.time() - rec["began"])
                            rec.update(step=step, eta=time.time() + (rec["total"] - step) / rate)
                threading.Thread(target=follow, daemon=True).start()
            last_write = 0.0
            with open(logs / f"{v['name']}.log", "w") as lf:
                for line in proc.stdout:
                    lf.write(line)
                    m = STEP.search(line)
                    if m:
                        step, tot, rate = int(m[1]), int(m[2]), float(m[3])
                        rec.update(step=step, total=tot, eta=time.time() + (tot - step) / rate if rate else None)
                    m = RESULT.search(line)
                    if m:
                        rec.update(psnr=float(m[1]), ssim=float(m[2]),
                                   depth_err=float(m[3]) / 100 if m[3] else None)
                        for k, val in COUNTS.findall(line):
                            rec[k.replace(" ", "_")] = float(val) / 100
                        print(line.strip(), flush=True)
                    if time.time() - last_write > 5:
                        batch.write()
                        last_write = time.time()
            proc.wait()
            done.set()
            rec["finished"] = time.time()
            rec["seconds"] = round(rec["finished"] - rec["began"])
            if proc.returncode:
                tail = (logs / f"{v['name']}.log").read_text(errors="replace").strip().splitlines()
                rec.update(state="failed", error=(tail[-1] if tail else f"exit {proc.returncode}")[:160])
            else:
                rec["state"] = "done"
                if runner.program == "command":
                    pass
                elif runner.program == "brush":
                    rec.update(step=rec.get("total"), splats=runner.final_splats(out, rec.get("total")))
                if greedy is not None:
                    res = {k: rec.get(k) for k in ("psnr", "needles", "in_front")}
                    keep = greedy_keep(res, best, greedy)
                    rec["greedy"] = "kept" if keep else "dropped"
                    rec["desc"] = f"{rec.get('desc') or ''} [{rec['greedy']}]".strip()
                    if keep:
                        kept, best = v["args"], res
                    print(f"    greedy: {rec['greedy']}" + (f" (best PSNR {best['psnr']:.2f})" if best else ""),
                          flush=True)
                if runner.program != "command":
                    label = f"exp-{slug(plan.get('name', 'batch'))[:40]}-{slug(v['name'])}"
                    try:
                        rec["downloads"] = publish(cfg, runner.local, runner.worker.get("ssh"), out, label,
                                                   data=data_candidates(v.get("args"), plan.get("common")))
                        rec["result"] = label if rec["downloads"] else None
                    except Exception as e:
                        print(f"    publishing {v['name']} failed: {e}", flush=True)
            print(f"    {rec['state']} in {rec['seconds']} s" + (f", {rec['splats']:,} splats" if rec.get("splats") else ""),
                  flush=True)
            batch.write()
    except KeyboardInterrupt:
        for rec in batch.data["variants"]:
            if rec["state"] in ("running", "waiting"):
                rec.update(state="failed", error="stopped")
    finally:
        stop.set()
        batch.data["finished"] = time.time()
        batch.write()
        print("=== all done", flush=True)


def publish_batch(path, force=False):
    """Publish the finished variants of an existing batch file (and record the downloads); with
    force, again even when already published (e.g. to put them in the queue's frame)."""
    path = Path(path)
    data = json.loads(path.read_text())
    cfg = json.loads((APP_HOME / "config.json").read_text())
    plan = data.get("plan") or {}
    local = data.get("worker") == "mac"
    ssh = None if local else next(w["ssh"] for w in cfg.get("workers", []) if w.get("name") == data["worker"])
    out_root = data.get("out") or plan.get("out")
    if not out_root:
        sys.exit("this batch file doesn't record its output folder")
    for rec in data["variants"]:
        if rec.get("state") != "done" or (rec.get("downloads") and not force):
            continue
        label = f"exp-{slug(data.get('name', 'batch'))[:40]}-{slug(rec['name'])}"
        rec["downloads"] = publish(cfg, local, ssh, f"{out_root}/{rec['name']}", label,
                                   data=data_candidates(rec.get("args"), data.get("common") or plan.get("common")))
        rec["result"] = label if rec["downloads"] else None
        print(f"{rec['name']}: {', '.join(d['file'] for d in rec['downloads']) or 'nothing to publish'}", flush=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(path)


if __name__ == "__main__":
    if len(sys.argv) in (3, 4) and sys.argv[1] == "--publish" and sys.argv[3:] in ([], ["--force"]):
        publish_batch(sys.argv[2], force=sys.argv[3:] == ["--force"])
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        sys.exit(__doc__)
