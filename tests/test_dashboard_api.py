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
        self.assertEqual(sorted(copied), ["capZ.sog", "meta.json", "preview.jpg"])
        self.assertEqual(sorted(p.name for p in (root / "share" / "capZ").iterdir()),
                         ["capZ.sog", "meta.json", "preview.jpg"])
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
        out, stats = gq.refined_model(self.cfg, cap, root / "work", lambda **k: None, job)
        self.assertTrue(stats["cached"])
        self.assertIn("0.9 0.1", (out / "sparse" / "0" / "images.txt").read_text())
        self.assertTrue((out / "images").is_symlink())
        # A different capture (changed poses) gets a different key.
        (cap / "sparse" / "0" / "images.txt").write_text("1 1 0 0 0 1 0 0 1 a.jpg\n\n")
        self.assertNotEqual(cache, gq.pose_cache_dir(self.cfg, cap, "capP"))

    def test_preview_preset_and_photo_size(self):
        gq = self.gq
        st = gq.effective_settings(self.cfg, {"preset": "preview"})
        self.assertEqual((st["total_steps"], st["max_resolution"]), (3000, 720))
        self.assertEqual(gq.Inbox.photo_size_from("# c\n1 PINHOLE 1920 1440 1 1 1 1\n"), 1920)

    def test_snapshot_has_estimates_and_presets(self):
        snap = self.snapshot()
        self.assertEqual(set(snap["presets"]), {"preview", "fast", "standard", "high"})
        self.assertTrue(all(q["est_seconds"] for q in snap["queue"]))
        self.assertTrue(snap["inbox_dir"].endswith("inbox"))


if __name__ == "__main__":
    unittest.main()
