"""Show any long job (a test run, a conversion, a script on the PC) as a card with progress
bars on the dashboard, next to the experiment batches: one row per step, the running one
with its own bar and ETA.

  python3 tools/track.py start ID --name "LIMAP line test" [--worker pc] [--note ...]
                         [--steps "matching,lines,triangulation"] [--no-hold]
  python3 tools/track.py step ID NAME [--done N --total M --unit pairs] [--expected SECONDS]
  python3 tools/track.py finish ID [--failed "message"]
  python3 tools/track.py follow ID --log-cmd "<shell command printing the log's tail>"
                         --rule "matching=Matching local features" --rule "lines=Start 2D line"
                         [--end "completed"] [--every 15]

A step command marks the steps before it done. Rows show done/total with a rate-based ETA,
or with only --expected, a bar filled by elapsed time (drawn fainter: an estimate).

follow keeps a card up to date from a log (local or over ssh): every --every seconds it runs
--log-cmd (for a long log, have it print the lines that mark steps plus the last few, so the
markers stay in view), moves to the latest step whose --rule regex has appeared after all the
earlier steps' markers, in order (steps only go forward),
reads the step's progress from the newest tqdm-style "N/M [elapsed<left" bar (or, with
--count STEP=COMMAND, from a command printing "done total", e.g. rows in a database), ends the card at
--end, and fails it at a Python traceback. Stopping follow leaves the card to go stale (it
reads as stopped after 2 minutes without updates).

A tracked card normally keeps the queue off its worker while it runs, like an experiment
batch; --no-hold lets captures start there anyway.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

APP_HOME = Path(os.environ.get("GAUSSQUEUE_HOME", Path.home() / "Library/Application Support/GaussQueue"))
BATCHES = APP_HOME / "experiments" / "batches"
TQDM = re.compile(r"(\d+(?:\.\d+)?)([kMG]?)(?:B|it)?/(\d+(?:\.\d+)?)([kMG]?)(?:B|it)? \[([\d:]+)<([\d:?]+)")
SCALE = {"": 1, "k": 1e3, "M": 1e6, "G": 1e9}


def path(id_):
    return BATCHES / f"track-{re.sub(r'[^A-Za-z0-9_.-]+', '-', id_)}.json"


def load(id_):
    p = path(id_)
    return json.loads(p.read_text()) if p.exists() else None


def save(id_, data):
    BATCHES.mkdir(parents=True, exist_ok=True)
    data["updated"] = time.time()
    p = path(id_)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(p)


def start(id_, name, worker=None, note=None, steps=(), hold=True):
    data = {"id": f"track-{id_}", "name": name, "worker": worker, "note": note, "hold": hold,
            "tracked": True, "started": time.time(), "finished": None, "eta": None,
            "variants": [{"name": s, "state": "queued"} for s in steps]}
    save(id_, data)
    return data


def step(id_, name, done=None, total=None, unit="", expected=None, label=None, eta=None, data=None):
    data = data or load(id_)
    if data is None:
        raise SystemExit(f"no tracked card {id_}; run start first")
    rows, now = data["variants"], time.time()
    i = next((k for k, v in enumerate(rows) if v["name"] == name), None)
    if i is None:
        rows.append({"name": name, "state": "queued"})
        i = len(rows) - 1
    for v in rows[:i]:
        if v["state"] in ("queued", "running"):
            v.update(state="done", finished=v.get("finished") or now)
            v.pop("eta", None)
    v = rows[i]
    if v["state"] != "running":
        v.update(state="running", started=now)
    v.update(done=done, total=total, unit=unit, expected=expected, label=label)
    if eta:
        v["eta"] = eta
    elif expected:
        v["eta"] = v["started"] + expected
    elif done and total and now - v["started"] > 5:
        v["eta"] = v["started"] + (now - v["started"]) * total / done
    else:
        v["eta"] = None
    data["eta"] = v["eta"] if i == len(rows) - 1 else None
    save(id_, data)
    return data


def finish(id_, failed=None, data=None):
    data = data or load(id_)
    if data is None:
        return
    now = time.time()
    for v in data["variants"]:
        if v["state"] == "running" or (v["state"] == "queued" and not failed):   # a finished job did every step
            v.update(state="failed" if failed and v["state"] == "running" else "done", finished=now, error=failed)
            v.pop("eta", None)
    data.update(finished=now, eta=None, failed=bool(failed), error=failed)
    save(id_, data)


def secs(t):
    if "?" in t:
        return None
    s = 0
    for part in t.split(":"):
        s = s * 60 + int(part)
    return s


def follow(id_, log_cmd, rules, end=None, every=15, counts=None):
    data = load(id_)
    if data is None:
        raise SystemExit(f"no tracked card {id_}; run start first")
    names = [r[0] for r in rules]
    have = [v["name"] for v in data["variants"]]
    for n in names:
        if n not in have:
            data["variants"].append({"name": n, "state": "queued"})
    save(id_, data)
    cur = next((k for k, n in enumerate(names) for v in data["variants"]
                if v["name"] == n and v["state"] == "running"), -1)
    while True:
        r = subprocess.run(log_cmd, shell=True, capture_output=True, text=True, errors="replace")
        text = r.stdout.replace("\r", "\n")
        # The steps in order: each marker counts only after the one before it (so a word in a
        # config dump at the top can't jump ahead); never back from where it got to.
        k, start_at = -1, 0
        while k + 1 < len(rules):
            m = re.search(rules[k + 1][1], text[start_at:])
            if not m:
                break
            k, start_at = k + 1, start_at + m.end()
        cur = max(cur, k)
        if re.search(r"^Traceback \(most recent call last\)", text, re.M):
            last = [l for l in text.splitlines() if l.strip()][-1].strip()
            finish(id_, failed=last[:200])
            print("failed:", last)
            return 1
        if end and re.search(end, text):
            finish(id_)
            print("done")
            return 0
        if cur >= 0:
            # Bars printed after the step's latest marker only (a log may repeat the marker
            # after the previous step's finished bar).
            last = [m.end() for m in re.finditer(rules[cur][1], text)] if cur == k else []
            bars = TQDM.findall(text[max(last) if last else start_at:])
            done = total = eta = None
            if bars:
                a, sa, b, sb, _, left = bars[-1]
                done, total = round(float(a) * SCALE[sa]), round(float(b) * SCALE[sb])
                left = secs(left)
                eta = time.time() + left if left is not None else None
            probe = (counts or {}).get(names[cur])
            if probe:   # a step that prints no progress: ask a command for "done total"
                out = subprocess.run(probe, shell=True, capture_output=True, text=True).stdout.split()
                if len(out) >= 2 and all(x.replace(".", "", 1).isdigit() for x in out[:2]):
                    done, total, eta = round(float(out[0])), round(float(out[1])), None
            data = step(id_, names[cur], done, total, data=None, eta=eta)
        else:
            save(id_, load(id_))   # keep the card live while waiting for the first step
        time.sleep(every)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("id")
    s.add_argument("--name", required=True)
    s.add_argument("--worker")
    s.add_argument("--note")
    s.add_argument("--steps", default="")
    s.add_argument("--no-hold", action="store_true")
    s = sub.add_parser("step")
    s.add_argument("id")
    s.add_argument("name")
    s.add_argument("--done", type=float)
    s.add_argument("--total", type=float)
    s.add_argument("--unit", default="")
    s.add_argument("--label")
    s.add_argument("--expected", type=float)
    s = sub.add_parser("finish")
    s.add_argument("id")
    s.add_argument("--failed")
    s = sub.add_parser("follow")
    s.add_argument("id")
    s.add_argument("--log-cmd", required=True)
    s.add_argument("--rule", action="append", default=[], help="STEP=REGEX, in order")
    s.add_argument("--end")
    s.add_argument("--count", action="append", default=[],
                   help="STEP=COMMAND printing 'done total', for a step whose log shows no progress")
    s.add_argument("--every", type=float, default=15)
    a = p.parse_args(argv)
    if a.cmd == "start":
        start(a.id, a.name, a.worker, a.note, [x for x in a.steps.split(",") if x], not a.no_hold)
    elif a.cmd == "step":
        step(a.id, a.name, a.done, a.total, a.unit, a.expected, a.label)
    elif a.cmd == "finish":
        finish(a.id, a.failed)
    else:
        rules = [tuple(r.split("=", 1)) for r in a.rule]
        return follow(a.id, a.log_cmd, rules, a.end, a.every, dict(c.split("=", 1) for c in a.count))


if __name__ == "__main__":
    sys.exit(main())
