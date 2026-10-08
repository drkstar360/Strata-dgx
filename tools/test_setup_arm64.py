"""Tests for setup.py on an ARM CPU (the DGX Spark, docs/DGX_SPARK.md): the CPU is named and never sent to the x86
floor checks, the GB10's "[N/A]" memory makes it a unified-memory GPU sized from the RAM, no ready-made (x86) engine is
used, and on x86 nothing changes.  No compiler, no GPU.

    python -m unittest tools.test_setup_arm64
"""
from __future__ import annotations

import io
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

# what a DGX Spark reports (measured, docs/DGX_SPARK.md)
SPARK_SMI = "0, NVIDIA GB10, [N/A], 12.1, 580.178.04\n"
SPARK_CPUINFO = ("processor\t: 0\nBogoMIPS\t: 2000.00\nFeatures\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics "
                 "fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit "
                 "uscat ilrcpc flagm sb paca pacg dcpodp sve2 i8mm bf16\nCPU implementer\t: 0x41\nCPU part\t: 0xd85\n")
SPARK_LSCPU = ("Architecture:             aarch64\n  Model name:             Cortex-X925\n"
               "  Model name:             Cortex-A725\n")
SPARK_RAM = 121.7


def spark():
    """setup as it runs on the Spark: ARM64, Linux, its nvidia-smi / lscpu / cpuinfo / RAM."""
    return [mock.patch.object(setup, "ARM64", True), mock.patch.object(setup, "WIN", False),
            mock.patch.object(setup, "ram_gb", return_value=SPARK_RAM),
            mock.patch.object(setup, "out", lambda cmd: SPARK_LSCPU if cmd[0] == "lscpu" else SPARK_SMI),
            mock.patch("builtins.open", lambda *a, **k: io.StringIO(SPARK_CPUINFO))]


class Arm64Test(unittest.TestCase):
    def setUp(self):
        self.patches = spark()
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def test_cpu_is_named_without_x86_flags(self):
        name, avx2, avx512 = setup.cpu_info()
        self.assertEqual(name, "Cortex-X925 + Cortex-A725")
        self.assertFalse(avx2)
        self.assertFalse(avx512)

    def test_no_x86_floor(self):
        self.assertEqual(setup.cpu_floor(False), "")
        with mock.patch.dict(os.environ, {"STRATA_ISA_FLOOR": "none"}):
            self.assertEqual(setup.cpu_floor(False), "")      # an x86 build CMake refuses on ARM

    def test_gb10_is_unified_memory(self):
        found = setup.gpus()
        self.assertEqual(len(found), 1)
        g = found[0]
        self.assertEqual((g["name"], g["arch"], setup.cc(g)), ("NVIDIA GB10", "121", "12.1"))
        self.assertTrue(g["uma"])
        self.assertAlmostEqual(g["vram_gb"], SPARK_RAM - setup.UMA_OS_LEFT_GB)
        self.assertEqual(setup.low_ram_vram(g), 0.0)          # the RAM is not counted twice
        self.assertTrue(setup.spark_recommends(g, SPARK_RAM))        # UD-IQ4_XS fits in the shared memory
        self.assertFalse(setup.strix_halo_recommends(g, SPARK_RAM))  # upstream keeps that one to real Strix Halo

    def test_no_ready_made_engine(self):
        self.assertIsNone(setup.get_prebuilt("https://example.invalid/", {"arch": "121"}, False))


class X86UnchangedTest(unittest.TestCase):
    def test_na_memory_still_skipped_on_x86(self):
        with mock.patch.object(setup, "ARM64", False), mock.patch.object(setup, "out", lambda cmd: SPARK_SMI):
            self.assertEqual(setup.gpus(), [])

    def test_discrete_card_unchanged(self):
        smi = "0, NVIDIA GeForce RTX 5090, 32607, 12.0, 580.88\n"
        with mock.patch.object(setup, "ARM64", False), mock.patch.object(setup, "out", lambda cmd: smi):
            g = setup.gpus()[0]
        self.assertNotIn("uma", g)
        self.assertAlmostEqual(g["vram_gb"], 32607 / 1024.0)

    def test_no_spark_recommendation_on_x86(self):
        g = {"index": 0, "name": "AMD Radeon 8060S", "uma": True, "dedicated_gb": 2.0, "vram_gb": 66.0, "arch": "gfx1151"}
        with mock.patch.object(setup, "ARM64", False):
            self.assertFalse(setup.spark_recommends(g, SPARK_RAM))

    def test_x86_floor_unchanged(self):
        with mock.patch.object(setup, "ARM64", False), mock.patch.dict(os.environ, {"STRATA_ISA_FLOOR": ""}):
            self.assertEqual(setup.cpu_floor(True), "")


if __name__ == "__main__":
    unittest.main()
