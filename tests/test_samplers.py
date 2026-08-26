#!/usr/bin/env python3
"""Tests for src/samplers.c (production sampler stack).

Drives a tiny CLI binary compiled from the #ifdef SAMPLERS_MAIN block in
src/samplers.c -- keeps Python free of ctypes struct-layout coupling.

Compile flags (standalone, no Makefile changes):
    cc -std=c99 -O2 -DSAMPLERS_MAIN -o <tmp>/tt_sampler_cli src/samplers.c -lm

Run:
    python3 tests/test_samplers.py
"""
import math
import shutil
import subprocess
import sys
import tempfile
import os
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src", "samplers.c")


def compile_cli():
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise RuntimeError("no C compiler found (cc/gcc/clang)")
    exe = os.path.join(tempfile.mkdtemp(prefix="tt_samplers_"), "tt_sampler_cli")
    cmd = [cc, "-std=c99", "-O2", "-Wall", "-Wextra",
           "-DSAMPLERS_MAIN", "-o", exe, SRC, "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("compile failed:\n" + r.stderr)
    return exe


class SamplerCli:
    """Persistent driver process; one command per line."""

    def __init__(self, exe):
        self.p = subprocess.Popen(
            [exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)

    def _cmd(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        out = self.p.stdout.readline().strip()
        if out.startswith("ERR"):
            raise RuntimeError(f"driver error: {out} (line: {line!r})")
        return out

    def run(self, cfg, seed, logits):
        csv = ",".join(repr(float(x)) for x in logits)
        return int(self._cmd(f"run {cfg} {seed} {csv}"))

    def seq(self, cfg, seed, ndraws, logits):
        csv = ",".join(repr(float(x)) for x in logits)
        return [int(t) for t in self._cmd(f"seq {cfg} {seed} {ndraws} {csv}").split()]

    def cand(self, cfg, logits):
        csv = ",".join(repr(float(x)) for x in logits)
        pairs = self._cmd(f"cand {cfg} {csv}").split()
        toks = [int(p.split(":")[0]) for p in pairs]
        probs = [float(p.split(":")[1]) for p in pairs]
        return toks, probs


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def entropy(probs):
    return -sum(p * math.log(p) for p in probs if p > 0)


class TestSamplers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = SamplerCli(compile_cli())

    # -- greedy ----------------------------------------------------------
    def test_greedy_is_argmax(self):
        cases = [
            ([3.0, 2.9, 1.0], 0),
            ([-5.0, -1.0, -2.0], 1),
            ([0.0] * 10, 0),
            ([1.0, 9.5, 9.49, 2.0], 1),
        ]
        for logits, want in cases:
            for seed in (0, 42, 2**63):
                self.assertEqual(self.cli.run("greedy", seed, logits),
                                 want, f"logits={logits}")

    def test_temp_to_zero_converges_argmax(self):
        logits = [1.0, 4.2, 4.0, 0.5]
        for t in (1e-6, 1e-4):
            # temp <= 0 or tiny -> greedy fallback path (llama.cpp convention)
            self.assertEqual(self.cli.run(f"T={t}", 123, logits), 1)

    def test_high_temp_flattens_entropy(self):
        logits = [3.0, 2.5, 2.0, 1.0]
        _, low = self.cli.cand("T=0.3", logits)
        _, high = self.cli.cand("T=4.0", logits)
        self.assertGreater(entropy(high), entropy(low))
        # high-temp empirical distribution should spread over more tokens
        draws_hi = set(self.cli.seq("T=4.0", 9, 200, logits))
        draws_lo = set(self.cli.seq("T=0.3", 9, 200, logits))
        self.assertGreater(len(draws_hi), len(draws_lo))

    # -- top-k -----------------------------------------------------------
    def test_topk1_equals_greedy(self):
        logits = [0.5, 2.0, 1.9, -1.0]
        for seed in range(20):
            self.assertEqual(self.cli.run("K=1,T=1.0", seed, logits),
                             self.cli.run("greedy", seed, logits))

    def test_topk5_subset_of_top5_set(self):
        logits = [4.0, 1.0, 3.5, 0.2, 2.9, 3.6, 0.1, 2.0, 3.1, -2.0]
        order = sorted(range(len(logits)), key=lambda i: -logits[i])
        top5 = set(order[:5])
        draws = set(self.cli.seq("K=5,T=1.0", 77, 500, logits))
        self.assertTrue(draws <= top5,
                        f"drew outside top-5: {draws - top5}")
        self.assertEqual(len(draws), 5)  # all five reachable given enough draws

    # -- top-p -----------------------------------------------------------
    def test_topp_minimal_set_and_mass(self):
        logits = [2.0, 1.5, 0.1, 0.05, -1.0]
        p = softmax(logits)
        toks, probs = self.cli.cand("P=0.5", logits)
        mass = sum(probs)
        self.assertGreaterequal = None  # guard against typo-shadows below
        self.assertGreaterEqual(mass, 0.5 - 1e-5)
        # minimality: dropping the last candidate must fall under 0.5
        orig = sorted(p, reverse=True)
        kept_mass_unnorm = sum(orig[:len(toks)])
        dropped = kept_mass_unnorm - orig[len(toks) - 1]
        self.assertLess(dropped, 0.5)
        # candidate list is exactly the top-k prefix of the true distribution
        self.assertEqual(toks, sorted(range(len(p)), key=lambda i: -p[i])[:len(toks)])

    # -- min-p -----------------------------------------------------------
    def test_minp_relative_threshold(self):
        logits = [5.0, 2.0, 0.0, -1.0]     # p_top >> others
        base = softmax(logits)
        toks, _ = self.cli.cand("M=0.5", logits)
        # keep only tokens with p >= 0.5 * p_top
        want = [i for i, v in enumerate(base) if v >= 0.5 * base[0]]
        self.assertEqual(sorted(toks), want)
        # min_p off keeps everything
        toks_all, _ = self.cli.cand("", logits)
        self.assertEqual(toks_all, list(range(len(logits))))

    # -- repetition penalty ----------------------------------------------
    def test_repeat_penalty_suppresses_repeats(self):
        # token 0 leads by 0.5; one prior occurrence + penalty 2.0 flips it:
        #   3.0/2.0 = 1.5 < 2.5 -> token 1 wins deterministically (temp->tiny)
        logits = [3.0, 2.5, 0.0]
        hist = [0]
        cfg = "rp=2.0,rln=8,hist=0,T=0.000001"
        self.assertEqual(self.cli.run(cfg, 1, logits), 1)
        # without penalty token 0 still wins
        self.assertEqual(self.cli.run("T=0.000001", 1, logits), 0)
        # window excludes older tokens: history outside last-n is ignored
        far_hist = "rp=2.0,rln=1,hist=0;0;0;0"
        # trailing window of size 1 sees only the final 0 -> still suppressed
        self.assertEqual(self.cli.run(far_hist + ",T=0.000001", 1, logits), 1)
        no_hit = "rp=2.0,rln=1,hist=9;1;T=0.000001"
        self.assertEqual(self.cli.run(no_hit, 1, logits), 0)

    def test_frequency_presence_penalties(self):
        # freq penalty proportional to count: two occurrences hit harder than one
        logits = [3.0, 2.5, 0.0]
        once = self.cli.run("fp=0.4,hist=0,T=0.000001", 1, logits)      # 3.0-0.4=2.6>2.5
        twice = self.cli.run("fp=0.4,hist=0;0;T=0.000001", 1, logits)   # 3.0-0.8<2.5
        self.assertEqual(once, 0)
        self.assertEqual(twice, 1)
        # presence fires once regardless of count
        pres1 = self.cli.run("pp=0.6,hist=0,T=0.000001", 1, logits)     # 3.0-0.6=2.4<2.5
        pres2 = self.cli.run("pp=0.6,hist=0;0;0;T=0.000001", 1, logits)
        self.assertEqual(pres1, 1)
        self.assertEqual(pres2, 1)

    # -- determinism ------------------------------------------------------
    def test_determinism_same_seed_100_draws(self):
        logits = [2.2, 1.1, 3.3, 0.4, 1.7, 2.9]
        cfg = "T=0.8,K=40,P=0.95,rp=1.15,rln=64"
        a = self.cli.seq(cfg, 20240101, 100, logits)
        b = self.cli.seq(cfg, 20240101, 100, logits)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 100)

    def test_different_seeds_diverge(self):
        logits = [2.2, 1.1, 3.3, 0.4, 1.7, 2.9]
        cfg = "T=0.8,K=40,P=0.95"
        seen = {tuple(self.cli.seq(cfg, seed, 50, logits)) for seed in (1, 2, 3)}
        self.assertEqual(len(seen), 3, "distinct seeds produced identical streams")

    # -- speculative-decode API --------------------------------------------
    def test_candidates_sorted_renormalized(self):
        logits = [1.0, 3.0, 2.0, 0.5, 2.5]
        toks, probs = self.cli.cand("P=0.9,K=10", logits)
        self.assertEqual(len(toks), len(set(toks)))
        self.assertTrue(all(probs[i] >= probs[i + 1] - 1e-6 for i in range(len(probs) - 1)))
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
        expect = sorted(range(5), key=lambda i: -logits[i])
        self.assertEqual(toks, expect[: len(toks)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
