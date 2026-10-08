"""docker-entrypoint.sh keeps the install config on the /data volume as the one that counts (#1244): with REINSTALL=0
and a config on the volume, /opt/strata/strata-<tag>.json is a link to it, even when a regular file is already there.
Runs the real script in a temporary tree with a stub for .venv/bin/python (POSIX sh and symlinks needed: skipped on
Windows).

    python -m unittest tools.test_docker_entrypoint
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SH = shutil.which("sh")


@unittest.skipIf(os.name == "nt" or SH is None, "needs POSIX sh and symlinks")
class Entrypoint(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.opt = self.tmp / "opt"
        self.data = self.tmp / "data"
        (self.opt / ".venv" / "bin").mkdir(parents=True)
        (self.data / "config").mkdir(parents=True)
        # the stub "python": setup.py --setup writes the config the way setup does; a plain start prints what it would load
        stub = self.opt / ".venv" / "bin" / "python"
        stub.write_text('#!/bin/sh\nif [ "$2" = "--setup" ]; then echo \'{"args": ["from-setup"]}\' > strata-iq3_s.json\n'
                        'else echo "STARTED WITH: $(cat strata-iq3_s.json) KEY=${STRATA_API_KEY-unset}"; fi\n',
                        encoding="utf-8")
        stub.chmod(0o755)
        script = (ROOT / "docker-entrypoint.sh").read_text(encoding="utf-8").replace("/opt/strata", str(self.opt))
        self.script = self.tmp / "entry.sh"
        self.script.write_text(script, encoding="utf-8")

    def run_entry(self, reinstall, rc=0, **extra):
        env = dict(os.environ, STRATA_DATA=str(self.data), MODEL="IQ3_S", REINSTALL=reinstall)
        for k in ("MODEL_ALIASES", "EXPERT_CACHE", "API_KEY", "STRATA_API_KEY", "STRATA_BIND"):   # only what the test sets
            env.pop(k, None)
        env.update(extra)
        r = subprocess.run([SH, str(self.script)], env=env, capture_output=True, text=True, timeout=60)
        if rc == 0:
            self.assertEqual(r.returncode, 0, r.stderr)
        else:
            self.assertNotEqual(r.returncode, 0, r.stdout)
        return r.stdout + r.stderr

    def test_edited_volume_config_wins_over_a_regular_file_in_opt(self):
        (self.data / "config" / "strata-iq3_s.json").write_text('{"args": ["edited", "--kv-resident", "32768"]}\n')
        (self.opt / "strata-iq3_s.json").write_text('{"args": ["stale"]}\n')      # left by an earlier setup
        out = self.run_entry("0")
        self.assertIn("edited", out)
        self.assertNotIn("stale", out)
        self.assertTrue((self.opt / "strata-iq3_s.json").is_symlink())

    def test_link_is_made_when_there_is_no_file(self):
        (self.data / "config" / "strata-iq3_s.json").write_text('{"args": ["on-volume"]}\n')
        self.assertIn("on-volume", self.run_entry("0"))

    def test_first_setup_copies_the_config_to_the_volume_and_the_next_start_links_it(self):
        out = self.run_entry("0")                                    # no config on the volume: setup runs
        self.assertIn("from-setup", out)
        cfg = self.data / "config" / "strata-iq3_s.json"
        self.assertIn("from-setup", cfg.read_text())
        cfg.write_text('{"args": ["edited-later"]}\n')
        self.assertIn("edited-later", self.run_entry("0"))           # the regular file setup left is replaced by the link

    def test_model_aliases_reach_the_config_that_starts_on_the_first_setup(self):
        out = self.run_entry("0", MODEL_ALIASES=" qwen, local-model ,")
        self.assertIn('"aliases": [', out)                           # the copy setup left in opt, which starts now
        self.assertIn('"local-model"', out)
        cfg = json.loads((self.data / "config" / "strata-iq3_s.json").read_text())
        self.assertEqual(cfg["aliases"], ["qwen", "local-model"])
        self.assertEqual(cfg["args"], ["from-setup"])               # the rest of the config is kept

    def test_model_aliases_replace_the_volume_configs(self):
        cfg = self.data / "config" / "strata-iq3_s.json"
        cfg.write_text('{"args": ["on-volume"], "aliases": ["old"], "sampling": {"temperature": 0.6}}\n')
        self.run_entry("0", MODEL_ALIASES="gpt-local")
        got = json.loads(cfg.read_text())
        self.assertEqual(got["aliases"], ["gpt-local"])
        self.assertEqual(got["sampling"], {"temperature": 0.6})
        self.assertTrue((self.opt / "strata-iq3_s.json").is_symlink())   # still the link, not a copy

    def test_expert_cache_replaces_the_engine_argument(self):
        cfg = self.data / "config" / "strata-iq3_s.json"
        cfg.write_text('{"args": ["--native", "m.gguf", "--expert-cache", "auto", "--serve"], "aliases": ["kept"]}\n')
        self.run_entry("0", EXPERT_CACHE="4000")
        got = json.loads(cfg.read_text())
        self.assertEqual(got["args"], ["--native", "m.gguf", "--expert-cache", "4000", "--serve"])
        self.assertEqual(got["aliases"], ["kept"])                   # MODEL_ALIASES unset: not touched

    def test_expert_cache_is_added_when_the_config_has_none(self):
        out = self.run_entry("0", EXPERT_CACHE="AUTO")                # first setup: the stub's args have no cache flag
        self.assertIn("Expert cache: auto", out)
        got = json.loads((self.data / "config" / "strata-iq3_s.json").read_text())
        self.assertEqual(got["args"], ["from-setup", "--expert-cache", "auto"])

    def test_bad_expert_cache_stops_the_start_and_changes_nothing(self):
        cfg = self.data / "config" / "strata-iq3_s.json"
        text = '{"args": ["--expert-cache", "auto"]}\n'
        cfg.write_text(text)
        for bad in ("0", "-5", "8G", "lots"):
            out = self.run_entry("0", rc=1, EXPERT_CACHE=bad)
            self.assertIn("EXPERT_CACHE", out)
            self.assertNotIn("STARTED WITH", out)
            self.assertEqual(cfg.read_text(), text)

    def test_unset_model_aliases_leave_the_config_alone(self):
        cfg = self.data / "config" / "strata-iq3_s.json"
        text = '{"args": ["on-volume"], "aliases": ["from-web-page"]}\n'
        cfg.write_text(text)
        self.run_entry("0")
        self.assertEqual(cfg.read_text(), text)


    def test_bind_beyond_this_machine_needs_an_api_key(self):
        for bind in ("0.0.0.0", "192.168.1.210", "::"):
            for key in ("", "   "):
                out = self.run_entry("0", rc=1, STRATA_BIND=bind, API_KEY=key)
                self.assertIn("set API_KEY", out)
                self.assertNotIn("Setting up", out)                  # stopped before any setup pass or download
                self.assertNotIn("STARTED WITH", out)

    def test_bind_with_a_key_starts_and_hands_the_key_to_the_server(self):
        (self.data / "config" / "strata-iq3_s.json").write_text('{"args": ["on-volume"]}\n')
        out = self.run_entry("0", STRATA_BIND="0.0.0.0", API_KEY="s3cret")
        self.assertIn("KEY=s3cret", out)

    def test_localhost_bind_needs_no_key_and_sets_none(self):
        (self.data / "config" / "strata-iq3_s.json").write_text('{"args": ["on-volume"]}\n')
        for bind in (None, "127.0.0.1", "localhost"):
            out = self.run_entry("0", **({} if bind is None else {"STRATA_BIND": bind}))
            self.assertIn("KEY=unset", out)                          # an empty STRATA_API_KEY would stop the server


if __name__ == "__main__":
    unittest.main()
