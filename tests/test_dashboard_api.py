"""Tests for per-item settings, queue ordering and the dashboard API.

Runs a throwaway GaussQueue instance (temp GAUSSQUEUE_HOME, free port) with fake
inbox items. No Brush, iCloud or Messages needed:

    python3 -m unittest discover -s tests
"""
import importlib.util
import json
import os
import socket
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class DashboardAPITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        os.environ["GAUSSQUEUE_HOME"] = str(root / "home")
        spec = importlib.util.spec_from_file_location("gaussqueue", REPO / "gaussqueue.py")
        gq = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gq)
        cls.gq = gq

        cls.port = free_port()
        gq.APP_HOME.mkdir(parents=True)
        (root / "inbox" / "capA").mkdir(parents=True)
        (root / "inbox" / "capB").mkdir(parents=True)
        gq.CONFIG_PATH.write_text(json.dumps({
            "inbox_dir": str(root / "inbox"), "outbox_dir": str(root / "outbox"),
            "status_dir": str(root / "status"), "attach_dir": str(root / "attach"),
            "dashboard_port": cls.port, "settle_seconds": 0}))
        cls.cfg = gq.load_config()
        paths = gq.Paths(cls.cfg)
        state = {"history": [{"name": "old", "result": "done", "duration": "30m",
                              "finished": "01-01 00:00", "train_seconds": 1800,
                              "settings": {"total_steps": 15000, "max_resolution": 1440}}]}
        cls.inbox = gq.Inbox(paths)
        cls.overrides = gq.Overrides()
        cls.status = gq.Status(paths, state, cls.inbox, cls.overrides)
        cls.inbox.tick(cls.cfg)
        cls.inbox.tick(cls.cfg)
        gq.start_dashboard(cls.cfg, paths, cls.status, cls.overrides)
        cls.status.write(cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def post(self, body, origin=None, ctype="application/json"):
        origin = origin or f"http://127.0.0.1:{self.port}"
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/item", data=json.dumps(body).encode(),
            headers={"Content-Type": ctype, "Origin": origin}, method="POST")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    def snapshot(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/status") as r:
            return json.loads(r.read())

    def setUp(self):
        for name in ("capA", "capB", "capC"):
            self.overrides.clear(name)

    def test_preset_then_field_override_is_custom(self):
        self.assertEqual(self.post({"name": "capB", "action": "preset", "value": "fast"})[0], 200)
        self.assertEqual(self.post({"name": "capB", "action": "set", "key": "max_resolution",
                                    "value": 960})[0], 200)
        st = self.gq.effective_settings(self.cfg, self.overrides.get("capB"))
        self.assertEqual((st["preset"], st["base"], st["total_steps"], st["max_resolution"]),
                         ("custom", "fast", 7000, 960))

    def test_reset_returns_to_standard(self):
        self.post({"name": "capA", "action": "preset", "value": "high"})
        self.post({"name": "capA", "action": "reset"})
        st = self.gq.effective_settings(self.cfg, self.overrides.get("capA"))
        self.assertEqual(st["preset"], "standard")
        self.assertEqual(st["total_steps"], self.cfg["standard"]["total_steps"])

    def test_validation(self):
        code, body = self.post({"name": "capA", "action": "set", "key": "sh_degree", "value": 7})
        self.assertEqual(code, 400)
        self.assertIn("between", body)
        self.assertEqual(self.post({"name": "capA", "action": "set", "key": "brush",
                                    "value": "/bin/sh"})[0], 400)
        self.assertEqual(self.post({"name": "nope", "action": "hold"})[0], 400)

    def test_cross_site_requests_are_refused(self):
        self.assertEqual(self.post({"name": "capA", "action": "hold"},
                                   origin="https://example.com")[0], 403)
        self.assertEqual(self.post({"name": "capA", "action": "hold"}, ctype="text/plain")[0], 403)
        self.assertFalse(self.overrides.get("capA").get("hold"))

    def test_hold_and_run_next_order_the_queue(self):
        self.assertEqual([i.name for i in self.inbox.ready(self.overrides)], ["capA", "capB"])
        self.post({"name": "capB", "action": "run_next"})
        self.post({"name": "capA", "action": "hold", "value": True})
        self.assertEqual([i.name for i in self.inbox.ready(self.overrides)], ["capB"])
        queue = self.snapshot()["queue"]
        self.assertEqual([(q["name"], q["hold"], q["run_next"]) for q in queue],
                         [("capB", False, True), ("capA", True, False)])

    def test_pause_rules(self):
        # Queued (not running) items can't be paused or resumed.
        code, body = self.post({"name": "capA", "action": "pause"})
        self.assertEqual(code, 400)
        self.assertIn("only the running capture", body)
        # The running item: edits need a pause first; pause/resume set a request.
        job = {"name": "capA", "phase": "training", "paused": False}
        self.status.current = job
        try:
            code, body = self.post({"name": "capA", "action": "set", "key": "total_steps",
                                    "value": 1000})
            self.assertEqual(code, 400)
            self.assertIn("pause training", body)
            self.assertEqual(self.post({"name": "capA", "action": "resume"})[0], 400)
            self.assertEqual(self.post({"name": "capA", "action": "pause"})[0], 200)
            self.assertEqual(job["request"], "pause")
            job.update(paused=True, phase="paused")
            self.assertEqual(self.post({"name": "capA", "action": "set", "key": "total_steps",
                                        "value": 1000})[0], 200)
            snap = self.snapshot()["current"]
            self.assertEqual(snap["pending_settings"]["total_steps"], 1000)
            self.assertEqual(self.post({"name": "capA", "action": "hold"})[0], 400)
        finally:
            self.status.current = None

    def test_flip_is_output_only(self):
        gq = self.gq
        # Presets and reset leave the flip choice alone.
        self.post({"name": "capA", "action": "set", "key": "flip_up", "value": 1})
        self.post({"name": "capA", "action": "preset", "value": "fast"})
        st = gq.effective_settings(self.cfg, self.overrides.get("capA"))
        self.assertEqual((st["flip_up"], st["preset"]), (1, "fast"))
        self.post({"name": "capA", "action": "reset"})
        self.assertEqual(gq.effective_settings(self.cfg, self.overrides.get("capA"))["flip_up"], 1)
        # Changing only the flip never forces a training restart.
        a = gq.effective_settings(self.cfg, {})
        self.assertTrue(gq.same_settings(a, dict(a, flip_up=1 - a["flip_up"])))
        # The running capture can be flipped without pausing; other edits can't.
        self.status.current = {"name": "capB", "phase": "training", "paused": False}
        try:
            self.assertEqual(self.post({"name": "capB", "action": "set", "key": "flip_up",
                                        "value": 1})[0], 200)
            self.assertEqual(self.post({"name": "capB", "action": "set", "key": "sh_degree",
                                        "value": 1})[0], 400)
        finally:
            self.status.current = None

    def test_requeue_puts_finished_capture_back_on_hold(self):
        paths = self.inbox.paths
        (paths.done / "capC").mkdir()
        self.status.state["history"].append({"name": "capC", "result": "done", "duration": "1m",
                                             "finished": "01-01 00:01"})
        self.status.write(self.cfg)
        rows = {h["name"]: h for h in self.snapshot()["history"]}
        self.assertTrue(rows["capC"]["requeueable"])
        self.assertFalse(rows["old"]["requeueable"])   # its original isn't in done/
        self.assertEqual(self.post({"name": "capC", "action": "requeue"})[0], 200)
        self.assertTrue((paths.inbox / "capC").is_dir())
        self.assertFalse((paths.done / "capC").exists())
        self.assertTrue(self.overrides.get("capC").get("hold"))
        self.inbox.tick(self.cfg)
        self.assertNotIn("capC", [i.name for i in self.inbox.ready(self.overrides)])
        code, body = self.post({"name": "capC", "action": "requeue"})
        self.assertEqual(code, 400)
        self.assertIn("already in the inbox", body)
        self.assertEqual(self.post({"name": "old", "action": "requeue"})[0], 400)

    def test_share_copies_only_listed_files(self):
        root = Path(self.tmp.name)
        res = root / "results" / "capZ"
        res.mkdir(parents=True)
        for f in ("capZ.ply", "capZ.sog", "capZ.spz", "preview.jpg", "meta.json", "train.log"):
            (res / f).write_text(f)
        cfg = dict(self.cfg, share_dir=str(root / "share"))
        copied = self.gq.share_result(cfg, res)
        self.assertEqual(sorted(copied), ["capZ.spz", "meta.json", "preview.jpg"])
        self.assertEqual(sorted(p.name for p in (root / "share" / "capZ").iterdir()),
                         ["capZ.spz", "meta.json", "preview.jpg"])
        self.assertEqual(self.gq.share_result(dict(cfg, share_dir=None), res), [])

    def test_refine_poses_is_a_restart_setting(self):
        gq = self.gq
        a = gq.effective_settings(self.cfg, {})
        self.assertEqual(a["refine_poses"], int(bool(self.cfg.get("refine_poses"))))
        self.assertFalse(gq.same_settings(a, dict(a, refine_poses=1 - a["refine_poses"])))
        self.assertEqual(self.post({"name": "capA", "action": "set", "key": "refine_poses",
                                    "value": 1})[0], 200)
        st = gq.effective_settings(self.cfg, self.overrides.get("capA"))
        self.assertEqual((st["refine_poses"], st["preset"]), (1, "standard"))
        # On the running capture it needs a pause, like other training settings.
        self.status.current = {"name": "capB", "phase": "training", "paused": False}
        try:
            self.assertEqual(self.post({"name": "capB", "action": "set", "key": "refine_poses",
                                        "value": 1})[0], 400)
        finally:
            self.status.current = None

    def test_capture_quality_from_splatpack(self):
        import zipfile
        gq, root = self.gq, Path(self.tmp.name)
        pairs = ([{"quality": {"band": "ok", "summary": "large pose jump", "score": 0.63}}] * 6
                 + [{"quality": {"band": "good", "summary": "stable ar tracking", "score": 0.88}}] * 4)
        q = gq.summarize_splatpack({"pairs": pairs})
        self.assertEqual((q["photos"], q["pose_jump_pct"], q["low_overlap"]), (10, 60, True))
        self.assertEqual(q["ratings"][0], {"band": "ok", "label": "large pose jump", "count": 6, "pct": 60})
        self.assertIn("low overlap", gq.quality_text(q))
        # Read straight out of a zip, as the inbox does.
        z = root / "inbox" / "capQ.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("capQ/splatpack.json", json.dumps({"pairs": pairs}))
        item = gq.InboxItem(z)
        self.inbox.read_quality(item)
        self.assertEqual(item.quality["pose_jump_pct"], 60)
        z.unlink()

    def test_pose_cache_roundtrip(self):
        gq, root = self.gq, Path(self.tmp.name)
        cap = root / "capP" / "COLMAP_Text_Model"
        (cap / "sparse" / "0").mkdir(parents=True)
        (cap / "images").mkdir()
        (cap / "sparse" / "0" / "cameras.txt").write_text("1 PINHOLE 1920 1440 1300 1300 960 720\n")
        (cap / "sparse" / "0" / "images.txt").write_text("1 1 0 0 0 0 0 0 1 a.jpg\n\n")
        (cap / "sparse" / "0" / "points3D.txt").write_text("1 0 0 0 0 0 0 -1\n")
        gq.POSE_CACHE = root / "pose_cache"
        cache = gq.pose_cache_dir(self.cfg, cap, "capP")
        self.assertEqual(cache, gq.pose_cache_dir(self.cfg, cap, "capP"))   # stable key
        model = root / "refined"
        (model / "sparse" / "0").mkdir(parents=True)
        (model / "sparse" / "0" / "cameras.txt").write_text("1 PINHOLE 1920 1440 1300 1300 960 720\n")
        (model / "sparse" / "0" / "images.txt").write_text("1 0.9 0.1 0 0 0.05 0 0 1 a.jpg\n\n")
        gq.save_pose_cache(cache, model, {"photos": 1})
        job = {"name": "capP"}
        # Without feature points the old-style cache is only used when they aren't wanted.
        out, stats = gq.refined_model(dict(self.cfg, init_feature_points=False), cap, root / "work",
                                      lambda **k: None, job)
        self.assertTrue(stats["cached"])
        self.assertIn("0.9 0.1", (out / "sparse" / "0" / "images.txt").read_text())
        self.assertTrue((out / "images").is_symlink())
        # With them: cached too, and merged into points3D.txt after the LiDAR points.
        (model / "sparse" / "0" / gq.FEATURE_POINTS).write_text("7 5 5 5 9 9 9 0.5 1 2\n")
        gq.save_pose_cache(cache, model, {"photos": 1})
        out, stats = gq.refined_model(dict(self.cfg, init_feature_points=True), cap, root / "work",
                                      lambda **k: None, job)
        self.assertTrue(stats["cached"])
        self.assertEqual(stats["feature_points"], 1)
        pts = [l.split() for l in (out / "sparse" / "0" / "points3D.txt").read_text().splitlines()
               if not l.startswith("#")]
        self.assertEqual(pts, [["1", "0", "0", "0", "0", "0", "0", "-1"], ["2", "5", "5", "5", "9", "9", "9", "0.5"]])
        # A different capture (changed poses) gets a different key.
        (cap / "sparse" / "0" / "images.txt").write_text("1 1 0 0 0 1 0 0 1 a.jpg\n\n")
        self.assertNotEqual(cache, gq.pose_cache_dir(self.cfg, cap, "capP"))

    def test_preview_preset_and_photo_size(self):
        gq = self.gq
        st = gq.effective_settings(self.cfg, {"preset": "preview"})
        self.assertEqual((st["total_steps"], st["max_resolution"]), (3000, 720))
        self.assertEqual(gq.Inbox.photo_size_from("# c\n1 PINHOLE 1920 1440 1 1 1 1\n"), 1920)

    def test_snapshot_requests(self):
        gq, root = self.gq, Path(self.tmp.name)
        snaps = gq.Snapshots(self.inbox.paths, self.status.state, self.status)
        self.status.snapshots = snaps
        out = root / "work" / "capA" / "out"
        out.mkdir(parents=True)
        job = {"name": "capA", "phase": "training", "paused": False, "total": 3000,
               "out_dir": str(out), "job_dir": str(out.parent), "settings": {}}
        self.status.current = job
        try:
            code, body = self.post({"name": "capA", "action": "snapshot"})
            self.assertEqual(code, 400)
            self.assertIn("no checkpoint yet", body)
            (out / "export_500.ply").write_text("ply")
            code, body = self.post({"name": "capA", "action": "snapshot"})
            self.assertEqual(code, 200)
            self.assertEqual(body["step"], 500)
            self.assertTrue((out.parent / "snapshots" / "export_500.ply").exists())  # set aside
            self.assertEqual(snaps.pending(), 1)
            self.assertIn("already snapshotted", self.post({"name": "capA", "action": "snapshot"})[1])
            self.assertEqual(self.post({"name": "capB", "action": "snapshot"})[0], 400)
        finally:
            self.status.current = None
            self.status.state["snapshots"] = []

    def test_workers_run_on_and_parallel_jobs(self):
        gq = self.gq
        pc = gq.RemoteWorker({"name": "pc", "label": "PC", "ssh": "gpu-pc",
                              "brush": "C:/b/brush_app.exe", "work_dir": "D:/gq/jobs"})
        self.status.workers = {"pc": pc, "mac": gq.LocalWorker()}
        try:
            # Run on: only known machines are accepted, and it's stored per capture.
            self.assertEqual(self.post({"name": "capA", "action": "run_on", "value": "pc"})[0], 200)
            self.assertEqual(self.overrides.get("capA").get("run_on"), "pc")
            self.assertEqual(self.post({"name": "capA", "action": "run_on", "value": "laptop"})[0], 400)
            self.post({"name": "capA", "action": "run_on", "value": "auto"})
            self.assertNotIn("run_on", self.overrides.get("capA"))
            # Two jobs at once, one per machine; both show up and both count as busy.
            self.status.jobs = {"pc": {"name": "capA", "phase": "training", "worker": "pc", "step": 10, "total": 100},
                                "mac": {"name": "capB", "phase": "training", "worker": "mac", "step": 5, "total": 100}}
            self.status.write(self.cfg)
            snap = self.snapshot()
            self.assertEqual([j["name"] for j in snap["jobs"]], ["capA", "capB"])   # remote first
            self.assertEqual(snap["jobs"][0]["worker_label"], "PC")
            self.assertTrue(snap["jobs"][0]["remote"])
            self.assertFalse({"capA", "capB"} & {q["name"] for q in snap["queue"]})   # busy, not queued
            self.assertEqual({w["name"] for w in snap["workers"]}, {"pc", "mac"})
            # Pause is refused on the remote machine, allowed on the Mac.
            self.assertIn("remote", self.post({"name": "capA", "action": "pause"})[1])
            self.assertEqual(self.post({"name": "capB", "action": "pause"})[0], 200)
        finally:
            self.status.jobs = {}
            self.status.workers = {}
            self.status.write(self.cfg)

    def test_gsplat_settings_and_scheduling(self):
        gq = self.gq
        # Choice settings take their listed strings only; prune is a bounded percentage.
        self.assertEqual(self.post({"name": "capA", "action": "set", "key": "trainer", "value": "gsplat"})[0], 200)
        self.assertEqual(self.post({"name": "capA", "action": "set", "key": "densify", "value": "mcmc"})[0], 200)
        self.assertEqual(self.post({"name": "capA", "action": "set", "key": "trainer", "value": "bash"})[0], 400)
        self.assertEqual(self.post({"name": "capA", "action": "set", "key": "prune_pct", "value": 99})[0], 400)
        self.assertEqual(self.post({"name": "capA", "action": "set", "key": "prune_pct", "value": 70})[0], 200)
        st = gq.effective_settings(self.cfg, self.overrides.get("capA"))
        self.assertEqual((st["trainer"], st["densify"], st["prune_pct"]), ("gsplat", "mcmc", 70))
        self.assertIn("gsplat mcmc", gq.settings_summary(st))
        # Trainer/densify restart training; pruning, like the flip, only affects the output.
        self.assertIn("trainer", gq.RESTART_KEYS)
        self.assertNotIn("prune_pct", gq.RESTART_KEYS)
        # Scheduling: gsplat captures wait for a gsplat worker unless pinned elsewhere.
        pc = gq.RemoteWorker({"name": "pc", "label": "PC", "ssh": "gpu-pc", "brush": "C:/b.exe",
                              "work_dir": "D:/j", "gsplat": {"python": "D:\\v\\python.exe", "scripts": "D:/gs"}})
        mac = gq.LocalWorker()
        self.assertEqual(pc.gsplat["python"], "D:/v/python.exe")
        ov = {"trainer": "gsplat"}
        self.assertTrue(gq.can_take(self.cfg, pc, ov, True))
        self.assertFalse(gq.can_take(self.cfg, mac, ov, True))
        self.assertTrue(gq.can_take(self.cfg, mac, ov, False))        # no gsplat anywhere: Brush
        self.assertTrue(gq.can_take(self.cfg, mac, dict(ov, run_on="mac"), True))
        self.assertTrue(gq.can_take(self.cfg, mac, {}, True))          # Brush captures go anywhere
        self.assertFalse(gq.can_take(self.cfg, mac, {"run_on": "pc"}, True))

    def test_start_view_pose(self):
        gq, root = self.gq, Path(self.tmp.name)
        m = root / "svmodel" / "sparse" / "0"
        m.mkdir(parents=True)
        # A camera at world (1, 2, 3) looking along world +X: the rows of R (world to camera)
        # are the camera's x, y, z axes in world coordinates, and z is the viewing direction.
        import math as _m
        R = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
        w = _m.sqrt(1 + R[0][0] + R[1][1] + R[2][2]) / 2
        q = [w, (R[2][1] - R[1][2]) / (4 * w), (R[0][2] - R[2][0]) / (4 * w), (R[1][0] - R[0][1]) / (4 * w)]
        c = [1, 2, 3]
        t = [-sum(R[i][j] * c[j] for j in range(3)) for i in range(3)]
        (m / "images.txt").write_text(
            f"1 {' '.join(map(str, q))} {' '.join(map(str, t))} 1 b_later.jpg\n\n"
            f"2 {' '.join(map(str, q))} {' '.join(map(str, t))} 1 a_first.jpg\n\n")
        name, origin, yaw = gq.start_view_pose(root / "svmodel", flip=False)
        self.assertEqual(name, "a_first.jpg")
        self.assertEqual(origin, [1, 2, 3])
        self.assertAlmostEqual(yaw, -90.0)          # heading +X turned onto +Z
        name, origin, yaw = gq.start_view_pose(root / "svmodel", flip=True)
        self.assertEqual(origin, [1, -2, -3])       # flip is 180° about X
        self.assertAlmostEqual(yaw, -90.0)          # heading +X is unchanged by the flip
        self.assertAlmostEqual(gq.start_view_pose(root / "svmodel", False, "-z")[2], 90.0)
        self.assertIsNone(gq.start_view_pose(root / "missing", False))
        self.assertIn("start_view", gq.OUTPUT_KEYS)

    def test_native_spz_is_used_when_installed(self):
        gq, root = self.gq, Path(self.tmp.name)
        fake = root / "fake_gq_spz"
        log = root / "fake_calls.txt"
        fake.write_text("#!/bin/sh\n"
                        f"echo \"$@\" >> '{log}'\n"
                        "printf x > \"$2\"\n")
        fake.chmod(0o755)
        ply = root / "in.ply"
        ply.write_text("ply")
        cfg = dict(self.cfg, spz_native=str(fake), spz_version=3)
        sizes, errors = gq.compress(cfg, ply, root / "cmp", "t", lambda: None, ["spz"])
        self.assertEqual((sizes, errors), ({"spz": 1}, []))
        self.assertIsNone(gq.flip_ply(cfg, ply, root / "flipped.ply"))
        calls = log.read_text().splitlines()
        self.assertIn("t.spz --spz-version 3", calls[0])
        self.assertTrue(calls[1].endswith("flipped.ply --flip"))
        # Not installed -> falls back (no native call recorded).
        self.assertIsNone(gq.native_spz(dict(cfg, spz_native=str(root / "missing"))))

    def test_estimates_per_machine(self):
        gq = self.gq
        pc = gq.RemoteWorker({"name": "pc", "label": "PC (RTX 4080)", "ssh": "gpu-pc",
                              "brush": "C:/b.exe", "work_dir": "D:/j"})
        hist = self.status.state["history"]
        hist.append({"name": "x", "result": "done", "train_seconds": 600, "worker": "pc", "trainer": "brush",
                     "duration": "10m", "finished": "01-02 00:00",
                     "settings": {"total_steps": 15000, "max_resolution": 1440}})
        self.status.workers = {"pc": pc, "mac": gq.LocalWorker()}
        try:
            self.status.write(self.cfg)
            q = next(i for i in self.snapshot()["queue"] if i["name"] == "capA")
            self.assertEqual(set(q["est"]), {"pc", "mac"})         # Auto: every machine
            self.assertAlmostEqual(q["est"]["pc"], 600, delta=1)   # same settings as the PC run
            self.assertAlmostEqual(q["est"]["mac"], 1800, delta=1) # old history counts as the Mac's
            self.post({"name": "capA", "action": "run_on", "value": "mac"})
            q = next(i for i in self.snapshot()["queue"] if i["name"] == "capA")
            self.assertEqual(set(q["est"]), {"mac"})               # pinned: just that machine
        finally:
            hist.pop()
            self.status.workers = {}
            self.overrides.clear("capA")
            self.status.write(self.cfg)

    def test_snapshot_has_estimates_and_presets(self):
        snap = self.snapshot()
        self.assertEqual(set(snap["presets"]), {"preview", "fast", "standard", "high"})
        self.assertTrue(all(q["est_seconds"] for q in snap["queue"]))
        self.assertTrue(snap["inbox_dir"].endswith("inbox"))


if __name__ == "__main__":
    unittest.main()
