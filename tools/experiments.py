"""Run a batch of training experiments on a GaussQueue worker, one variant after another,
with progress on the dashboard (an "Experiments" card) and the queue keeping off that
machine until the batch is done.

  python3 tools/experiments.py plan.json

plan.json:
  {"name": "Shelf box: lens fix",          shown on the card
   "worker": "pc",                          a worker from config.json (with gsplat)
   "out": "D:/experiments/round2",          each variant writes to <out>/<variant name>
   "script": "train.py",                    in the worker's gsplat scripts folder (default)
   "common": ["<data dir>", "--steps", "15000", ...],
   "variants": [{"name": "A", "args": ["--depth-weight", "0.3"]}, ...],
   "note": "optional line for the card"}

A variant waits while the queue is processing a capture on that machine. The runner's log
for each variant goes to $GAUSSQUEUE_HOME/experiments/<variant>.log. Stopping the runner
(Ctrl-C) stops the variant that's running.
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
STATE = APP_HOME / "experiments.json"
STEP = re.compile(r"step (\d+)/(\d+) .*?([\d.]+) it/s")
RESULT = re.compile(r"held-out PSNR ([\d.]+) SSIM ([\d.]+)(?: depth error ([\d.]+)%)?")


class Batch:
    def __init__(self, plan, worker):
        self.lock = threading.Lock()
        self.data = {"name": plan.get("name", "Experiments"), "worker": worker["name"],
                     "note": plan.get("note"), "started": time.time(), "finished": None,
                     "variants": [{"name": v["name"], "state": "queued", "args": v.get("args", [])}
                                  for v in plan["variants"]]}
        self.write()

    def write(self):
        with self.lock:
            self.data["updated"] = time.time()
            self.data["eta"] = self.eta()
            tmp = STATE.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=1))
            tmp.replace(STATE)

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


def main(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    cfg = json.loads((APP_HOME / "config.json").read_text())
    worker = next((w for w in cfg.get("workers", []) if w.get("name") == plan["worker"]), None)
    if not worker or not worker.get("gsplat"):
        sys.exit(f"no worker {plan['worker']!r} with gsplat in config.json")
    port = cfg.get("dashboard_port", 8765)
    g = worker["gsplat"]
    logs = APP_HOME / "experiments"
    logs.mkdir(parents=True, exist_ok=True)
    batch = Batch(plan, worker)
    stop = threading.Event()
    threading.Thread(target=batch.heartbeat, args=(stop,), daemon=True).start()
    try:
        for v, rec in zip(plan["variants"], batch.data["variants"]):
            while (busy := queue_job_on(worker["name"], port)):
                rec.update(state="waiting", waiting=f"the queue is processing {busy}")
                batch.write()
                time.sleep(20)
            rec.update(state="running", began=time.time(), step=None, total=None, eta=None, waiting=None)
            batch.write()
            args = ([f"{g['scripts']}/{plan.get('script', 'train.py')}"] + plan.get("common", [])
                    + ["--out", f"{plan['out']}/{v['name']}"] + v.get("args", []))
            cmd = f'cd /d "{g["scripts"]}" && ' + " ".join(quote(a) for a in [g["python"], "-u"] + args)
            print(f"=== {v['name']}: {' '.join(map(str, v.get('args', [])))}", flush=True)
            proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30",
                                     worker["ssh"], cmd + " 2>&1"],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            last_write = 0.0
            with open(logs / f"{v['name']}.log", "w") as lf:
                for line in proc.stdout:
                    lf.write(line)
                    m = STEP.search(line)
                    if m:
                        step, total, rate = int(m[1]), int(m[2]), float(m[3])
                        rec.update(step=step, total=total,
                                   eta=time.time() + (total - step) / rate if rate else None)
                    m = RESULT.search(line)
                    if m:
                        rec.update(psnr=float(m[1]), ssim=float(m[2]),
                                   depth_err=float(m[3]) / 100 if m[3] else None)
                        print(line.strip(), flush=True)
                    if time.time() - last_write > 5:
                        batch.write()
                        last_write = time.time()
            proc.wait()
            rec["finished"] = time.time()
            if proc.returncode:
                tail = (logs / f"{v['name']}.log").read_text(errors="replace").strip().splitlines()
                rec.update(state="failed", error=(tail[-1] if tail else f"exit {proc.returncode}")[:160])
            else:
                rec["state"] = "done"
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
