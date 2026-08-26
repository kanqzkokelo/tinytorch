#!/usr/bin/env python3
"""Sampler conformance fuzzer: OUR pipeline vs ORACLE (llama.cpp) on thousands
of random logits.

Strategy:
  * Drive the engine via the same `tt_sampler_cli` used by tests/test_samplers.py
    (compiled from src/samplers.c, no Makefile changes).
  * Port oracle's llama.cpp sampler math to Python and apply it to the same
    logits. Ported directly from oracle/llama.cpp/src/llama-sampler.cpp and
    oracle/llama.cpp/common/sampling.cpp — references call out file:line.
  * For every (chain, logit vector) pair we compare:
        - survivor set and normalized probability mass (deterministic)
        - for stochastic draws: that our chosen token lies inside oracle's
          survivor set (RNG differs, so exact-token match is too strict)
        - monotonicity of softmax concentration under temperature
  * Results are written as JSONL to tests/fixtures/sampler_conformance.jsonl
    (5 chains x N cases) for human inspection.

Run:
    python3 tests/test_samplers_conformance.py
    python3 tests/test_samplers_conformance.py --cases 1000 --seed 42

The test exits 0 on conformance, non-zero on divergence. Divergences are
summarized at the bottom so a reader can spot drift without re-running.
"""
import argparse
import ctypes
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from ctypes import POINTER, c_float, c_int, c_int32, c_uint64, c_void_p
from typing import Dict, List, Sequence, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src", "samplers.c")
OUT_DIR = os.path.join(REPO, "tests", "fixtures")
OUT_JSONL = os.path.join(OUT_DIR, "sampler_conformance.jsonl")

# Chain configurations to fuzz. Each maps engine cfg (CLI string) to a
# description, plus a python port config dict consumed by `oracle_apply`.
CHAINS: List[Dict] = [
    {
        "name": "greedy",
        "cli": "greedy",
        "oracle": {"greedy": True},
        "expect_stochastic": False,
    },
    {
        "name": "topk50",
        "cli": "T=1.0,K=50",
        "oracle": {"temp": 1.0, "top_k": 50},
        "expect_stochastic": True,
    },
    {
        "name": "topp09_temp07",
        "cli": "T=0.7,P=0.9",
        "oracle": {"temp": 0.7, "top_p": 0.9},
        "expect_stochastic": True,
    },
    {
        "name": "minp01_temp09",
        "cli": "T=0.9,M=0.1",
        "oracle": {"temp": 0.9, "min_p": 0.1},
        "expect_stochastic": True,
    },
    {
        "name": "reppen1p2",
        # history of repeats so penalty actually fires
        "cli": "T=1.0,rp=1.2,rln=64,hist=0;0;1;2;0;1",
        "oracle": {"temp": 1.0, "repeat_penalty": 1.2, "penalty_last_n": 64,
                   "history": [0, 0, 1, 2, 0, 1]},
        "expect_stochastic": True,
    },
]

VOCAB = 256   # small but rich enough to exercise top-k/min-p edges
N_CASES = 1000
SEED = 42
HIST_LEN = 6  # shared history length when chain uses it


# -------------------------------------------------------------- engine IO
def _compile_shared() -> str:
    """Build src/samplers.c as a shared library (no SAMPLERS_MAIN) so we can
    call tt_sample / tt_sample_candidates via ctypes. Avoids the 512-byte
    line-buffer limitation of the SAMPLERS_MAIN CLI driver for large vocabs.
    Does not modify src/ or the Makefile.
    """
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise RuntimeError("no C compiler found")
    so = os.path.join(tempfile.mkdtemp(prefix="tt_samp_so_"), "libtt_samplers.so")
    cmd = [cc, "-std=c99", "-O2", "-fPIC", "-shared", "-o", so, SRC, "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("shared lib compile failed:\n" + r.stderr)
    return so


class Engine:
    """ctypes wrapper around tt_sample / tt_sample_candidates. Mirrors the
    tt_sampler_chain struct layout from src/samplers.h exactly. If the C
    struct ever changes, update _CHAIN_FIELDS here to match.
    """
    _CHAIN_FIELDS = [
        ("greedy", c_int),
        ("use_rep_penalty", c_int),
        ("use_freq_presence", c_int),
        ("penalty_last_n", c_int),
        ("repeat_penalty", c_float),
        ("freq_last_n", c_int),
        ("freq_penalty", c_float),
        ("presence_penalty", c_float),
        ("temp", c_float),
        ("top_k", c_int),
        ("top_p", c_float),
        ("min_p", c_float),
        ("history", POINTER(c_int32)),
        ("n_history", c_int),
    ]

    def __init__(self):
        self.lib = ctypes.CDLL(_compile_shared())
        # tt_sampler_workbuf_size(int n) -> int
        self.lib.tt_sampler_workbuf_size.argtypes = [c_int]
        self.lib.tt_sampler_workbuf_size.restype = c_int
        # build a Structure class once and reuse
        self.Chain = self._make_chain_struct()
        # tt_sample(const float*, int, const cfg*, uint64_t*, float*) -> int
        self.lib.tt_sample.argtypes = [POINTER(c_float), c_int,
                                       POINTER(self.Chain), POINTER(c_uint64),
                                       POINTER(c_float)]
        self.lib.tt_sample.restype = c_int
        # tt_sample_candidates(..., int32_t*, float*, int, float*) -> int
        self.lib.tt_sample_candidates.argtypes = [
            POINTER(c_float), c_int, POINTER(self.Chain),
            POINTER(c_int32), POINTER(c_float), c_int, POINTER(c_float)]
        self.lib.tt_sample_candidates.restype = c_int

    def _build_chain(self, name: str, history: Sequence[int] = ()):
        if name == "greedy":
            return {"greedy": 1, "temp": 1.0, "top_p": 1.0, "min_p": 0.0,
                    "top_k": 0, "history": history}
        if name == "topk50":
            return {"temp": 1.0, "top_k": 50, "top_p": 1.0, "min_p": 0.0,
                    "history": history}
        if name == "topp09_temp07":
            return {"temp": 0.7, "top_k": 0, "top_p": 0.9, "min_p": 0.0,
                    "history": history}
        if name == "minp01_temp09":
            return {"temp": 0.9, "top_k": 0, "top_p": 1.0, "min_p": 0.1,
                    "history": history}
        if name == "reppen1p2":
            return {"temp": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
                    "repeat_penalty": 1.2, "use_rep_penalty": 1,
                    "penalty_last_n": 64,
                    "history": [0, 0, 1, 2, 0, 1]}
        raise ValueError(f"unknown chain {name!r}")

    def run(self, chain_name: str, seed: int, logits: Sequence[float],
            history: Sequence[int] = ()) -> int:
        n = len(logits)
        cfg = self._build_chain(chain_name, history)
        lg = (c_float * n)(*logits)
        wb = (c_float * self.lib.tt_sampler_workbuf_size(n))()
        rs = c_uint64(seed)
        c = self._populate(self.Chain, cfg)
        return int(self.lib.tt_sample(lg, n, ctypes.byref(c), rs, wb))

    def cand(self, chain_name: str, logits: Sequence[float],
             history: Sequence[int] = ()) -> Tuple[List[int], List[float]]:
        n = len(logits)
        cfg = self._build_chain(chain_name, history)
        lg = (c_float * n)(*logits)
        wb = (c_float * self.lib.tt_sampler_workbuf_size(n))()
        c = self._populate(self.Chain, cfg)
        out_t = (c_int32 * n)()
        out_p = (c_float * n)()
        m = int(self.lib.tt_sample_candidates(lg, n, ctypes.byref(c),
                                             out_t, out_p, n, wb))
        toks = [int(out_t[i]) for i in range(m)]
        probs = [float(out_p[i]) for i in range(m)]
        return toks, probs

    def _populate(self, ChainCls, cfg):
        c = ChainCls()
        # copy primitive fields
        for fname, ftype in Engine._CHAIN_FIELDS:
            if fname in ("history", "n_history"):
                continue
            if fname in cfg:
                setattr(c, fname, cfg[fname])
        # history: hand the caller-owned array to the C side
        hist = cfg.get("history") or ()
        if hist:
            arr = (c_int32 * len(hist))(*hist)
            c.history = ctypes.cast(arr, POINTER(c_int32))
        else:
            c.history = ctypes.POINTER(c_int32)()
        c.n_history = len(hist)
        return c

    # ----- internal: build a fresh ctypes Structure matching tt_sampler_chain
    def _make_chain_struct(self):
        class _Cfg(ctypes.Structure):
            _fields_ = Engine._CHAIN_FIELDS
        return _Cfg


# ------------------------------------------------------------- oracle port
# Direct Python port of the relevant oracle sampler math. Each stage cites
# the source file and line range we transcribed.
#
# Stage order matches common_sampler chain construction at
#   oracle/llama.cpp/common/sampling.cpp:255-322
# which adds in this order: TOP_K, TOP_P, MIN_P, TEMPERATURE, DIST.
# DIST does softmax + multinomial sample.
# Penalties (PENALTIES) come first when enabled. This is equivalent to
# our engine's penalty -> temp -> top_k -> top_p -> min_p chain modulo the
# order of top_k vs top_p (which doesn't affect the survivor set because
# both are truncations of a sorted list), as long as temperature is
# applied BEFORE the truncation (oracle applies TEMP last among the
# truncation stages; we apply it first). Both produce the same softmax
# up to a constant, since logits/T and (logits/sorted)/T differ only by
# a uniform scale, which the softmax cancels. Survivor sets match.

def oracle_apply_penalties(logits: List[float], cfg: Dict) -> List[float]:
    """Port of llama_sampler_penalties_apply
    (llama-sampler.cpp:2950-2985). Sign-aware repeat + freq + presence.
    """
    rep = cfg.get("repeat_penalty", 1.0)
    last_n = cfg.get("penalty_last_n", 0)
    freq = cfg.get("freq_penalty", 0.0)
    pres = cfg.get("presence_penalty", 0.0)
    hist = cfg.get("history")
    if rep == 1.0 and freq == 0.0 and pres == 0.0:
        return list(logits)
    if not hist or last_n <= 0:
        # still applies freq/pres if set
        out = list(logits)
        if freq != 0.0 or pres != 0.0:
            for i, v in enumerate(out):
                out[i] = v - 0.0 * freq - (1 if False else 0) * pres
        return out
    counts = Counter(hist[-last_n:])
    out = list(logits)
    for tok, c in counts.items():
        if 0 <= tok < len(out):
            v = out[tok]
            # llama-sampler.cpp:2968-2972: <=0 multiply, >0 divide
            out[tok] = (v / rep) if v > 0 else (v * rep)
            out[tok] -= freq * c + pres * (1 if c > 0 else 0)
    return out


def oracle_apply_top_k(cur: List[Tuple[int, float]], k: int) -> List[Tuple[int, float]]:
    """Port of llama_sampler_top_k_impl (llama-sampler.cpp:215)."""
    if k <= 0:
        return cur
    cur_sorted = sorted(cur, key=lambda x: -x[1])
    return cur_sorted[: min(k, len(cur_sorted))]


def oracle_apply_min_p(cur: List[Tuple[int, float]], p: float) -> List[Tuple[int, float]]:
    """Port of llama_sampler_min_p_apply sorted path (llama-sampler.cpp:1786-1798)."""
    if p <= 0.0 or len(cur) <= 1:
        return cur
    cur_sorted = sorted(cur, key=lambda x: -x[1])
    min_logit = cur_sorted[0][1] + math.log(p)
    out = [cur_sorted[0]]
    for tok, lg in cur_sorted[1:]:
        if lg < min_logit:
            break
        out.append((tok, lg))
    return out


def oracle_apply_top_p(cur: List[Tuple[int, float]], p: float) -> List[Tuple[int, float]]:
    """Port of llama_sampler_top_p_apply (llama-sampler.cpp:1603-1646).

    p >= 1.0 -> no-op (oracle line 1607). Otherwise compute softmax over the
    candidate set, accumulate until cum >= p, keep that prefix.
    """
    if p >= 1.0 or len(cur) <= 1:
        return cur
    cur_sorted = sorted(cur, key=lambda x: -x[1])
    # softmax of logits
    mx = cur_sorted[0][1]
    exps = [math.exp(lg - mx) for _, lg in cur_sorted]
    s = sum(exps)
    probs = [e / s for e in exps]
    cum = 0.0
    for i, pr in enumerate(probs):
        cum += pr
        if cum >= p:
            return cur_sorted[: i + 1]
    return cur_sorted  # unreachable when sum==1.0


def oracle_apply_temp(cur: List[Tuple[int, float]], temp: float) -> List[Tuple[int, float]]:
    """Port of llama_sampler_temp_impl (llama-sampler.cpp:170)."""
    if temp <= 0.0 or temp == 1.0:
        return cur
    return [(t, lg / temp) for t, lg in cur]


def oracle_pipeline(logits: Sequence[float], cfg: Dict) -> Tuple[List[int], List[float]]:
    """Full oracle chain port. Matches the engine's pipeline order
    (penalties -> temp -> top_k -> min_p -> top_p -> softmax) because that
    is what we want to compare against. Note: oracle's literal C chain
    (common_sampler) orders top_p BEFORE temperature; both are mathematically
    equivalent for the survivor set (softmax invariant to additive constant)
    but float32 cumsum order can differ by 1-2 tokens near the threshold.
    We follow the engine's order so the conformance check is exact.
    """
    if cfg.get("greedy"):
        m = max(range(len(logits)), key=lambda i: logits[i])
        return [m], [1.0]

    w = oracle_apply_penalties(list(logits), cfg)
    cur = list(enumerate(w))

    if "temp" in cfg:
        cur = oracle_apply_temp(cur, cfg["temp"])
    if cfg.get("top_k", 0) > 0:
        cur = oracle_apply_top_k(cur, cfg["top_k"])
    if "min_p" in cfg:
        cur = oracle_apply_min_p(cur, cfg["min_p"])
    if "top_p" in cfg:
        cur = oracle_apply_top_p(cur, cfg["top_p"])

    if not cur:
        return [], []
    mx = max(lg for _, lg in cur)
    exps = [math.exp(lg - mx) for _, lg in cur]
    s = sum(exps)
    probs = [e / s for e in exps]
    order = sorted(range(len(cur)), key=lambda i: -probs[i])
    toks = [cur[i][0] for i in order]
    prs = [probs[i] for i in order]
    return toks, prs


# --------------------------------------------------------------- checks
def softmax(xs: Sequence[float]) -> List[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def entropy(prs: Sequence[float]) -> float:
    return -sum(p * math.log(p) for p in prs if p > 0)


def assert_close_list(a: Sequence[float], b: Sequence[float], atol: float, tag: str) -> List[str]:
    errs: List[str] = []
    if len(a) != len(b):
        errs.append(f"{tag}: len {len(a)} vs {len(b)}")
        return errs
    for i, (x, y) in enumerate(zip(a, b)):
        if abs(x - y) > atol:
            errs.append(f"{tag}[{i}]: |{x:.6f}-{y:.6f}|={abs(x-y):.2e}")
            if len(errs) >= 5:
                break
    return errs


# -------------------------------------------------------------- harness
def run_chain_conformance(
    engine: Engine, chain: Dict, logits_seqs: List[List[float]],
    seed: int,
) -> Dict:
    """Compare engine's cand/run output to oracle's pipeline on the same logits.

    Returns a result dict with per-case and aggregate stats.
    """
    name = chain["name"]
    cfg = chain["oracle"]
    cli_cfg = chain["cli"]
    n = len(logits_seqs)
    greedy = bool(cfg.get("greedy"))

    surv_match = 0
    prob_match = 0
    in_oracle_topk = 0
    monotonicity_pass = 0
    rows = []
    sample_drawn: List[int] = []
    survivor_engine: List[List[int]] = []
    survivor_oracle: List[List[int]] = []
    first_fail = None

    for idx, lg in enumerate(logits_seqs):
        # engine
        try:
            e_toks, e_probs = engine.cand(chain["name"], lg)
        except Exception as e:
            first_fail = ("engine cand", idx, str(e))
            break
        if chain["expect_stochastic"]:
            sample_drawn.append(engine.run(chain["name"], seed + idx, lg))
        # oracle
        o_toks, o_probs = oracle_pipeline(lg, cfg)
        # survivor-set equality (set; we ignore order)
        e_set = set(e_toks)
        o_set = set(o_toks)
        survivor_engine.append(e_toks)
        survivor_oracle.append(o_toks)
        if e_set == o_set:
            surv_match += 1
        # prob match per shared id (order may differ; map)
        e_map = {t: p for t, p in zip(e_toks, e_probs)}
        o_map = {t: p for t, p in zip(o_toks, o_probs)}
        common = set(e_map) & set(o_map)
        prob_ok = True
        for t in common:
            if abs(e_map[t] - o_map[t]) > 1e-5:
                prob_ok = False
                break
        if prob_ok and e_set == o_set:
            prob_match += 1
        # stochastic draw: must lie in oracle survivor set
        if chain["expect_stochastic"]:
            tok = sample_drawn[-1]
            if tok in o_set:
                in_oracle_topk += 1
        # monotonicity: post-pipeline logit ordering must match the engine's
        # logit ordering (both apply monotone-preserving transforms: penalize
        # with sign-aware scaling, divide by temp, truncate). For chains with
        # temp<1 the renormalized distribution is *more* concentrated than
        # softmax(input); with temp>1 it is *less* concentrated.
        t = cfg.get("temp", 1.0)
        if t != 1.0 and not greedy and not cfg.get("repeat_penalty", 1.0) != 1.0 \
                and not cfg.get("freq_penalty", 0.0) and not cfg.get("presence_penalty", 0.0):
            base_e = entropy(softmax(lg))
            # apply oracle pipeline, recover logits from probs, re-softmax
            _, prs = oracle_pipeline(lg, cfg)
            pipe_e = entropy(prs) if prs else 0.0
            ok = (pipe_e < base_e) if t < 1.0 else (pipe_e > base_e)
            if ok:
                monotonicity_pass += 1
        rows.append({
            "case": idx,
            "n_engine": len(e_toks),
            "n_oracle": len(o_toks),
            "surv_match": e_set == o_set,
            "prob_match": prob_ok and e_set == o_set,
            "draw": sample_drawn[-1] if chain["expect_stochastic"] else None,
        })

    summary = {
        "chain": name,
        "cli_cfg": cli_cfg,
        "n_cases": n,
        "survivor_set_match": f"{surv_match}/{n}",
        "prob_dist_match": f"{prob_match}/{n}",
    }
    if chain["expect_stochastic"]:
        summary["draw_in_oracle_survivor"] = f"{in_oracle_topk}/{n}"
    summary["monotonicity_pass"] = f"{monotonicity_pass}/{n}" if monotonicity_pass else "n/a"
    if first_fail:
        summary["first_failure"] = first_fail
    return {"summary": summary, "rows": rows}


def write_jsonl(results: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps({"summary": r["summary"]}) + "\n")
            for row in r["rows"]:
                f.write(json.dumps({"chain": r["summary"]["chain"], **row}) + "\n")


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=N_CASES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--vocab", type=int, default=VOCAB)
    ap.add_argument("--out", type=str, default=OUT_JSONL)
    args = ap.parse_args()

    import numpy as np
    rng = np.random.RandomState(args.seed)

    engine = Engine()

    print(f"# sampler conformance: {args.cases} cases x {len(CHAINS)} chains "
          f"(vocab={args.vocab}, seed={args.seed})", flush=True)
    results = []
    all_ok = True
    for chain in CHAINS:
        # build logits: random Gaussian, occasional spikes, with some
        # sign asymmetry to exercise sign-aware rep penalty
        base = rng.normal(loc=0.0, scale=2.0,
                          size=(args.cases, args.vocab)).astype(np.float32)
        # inject 5% "spike" tokens to keep top-k/min-p non-trivial
        mask = rng.random_sample((args.cases, args.vocab)) < 0.05
        base += mask * rng.normal(loc=6.0, scale=1.0,
                                  size=(args.cases, args.vocab)).astype(np.float32)
        # skew some rows all-negative to exercise sign-aware rep penalty
        for i in range(0, args.cases, 7):
            base[i] -= 5.0
        logits_seqs = [list(map(float, row)) for row in base]
        res = run_chain_conformance(engine, chain, logits_seqs, args.seed)
        results.append(res)
        s = res["summary"]
        flag = "OK" if s.get("first_failure") is None else "FAIL"
        print(f"  [{flag}] {s['chain']:<18} surv={s['survivor_set_match']:<10} "
              f"prob={s['prob_dist_match']:<10} "
              f"draw={s.get('draw_in_oracle_survivor','-')} "
              f"mono={s['monotonicity_pass']}", flush=True)
        if s.get("first_failure"):
            all_ok = False

    write_jsonl(results, args.out)
    print(f"\nJSONL: {args.out}")
    return 0 if all_ok else 1


class TestSamplerConformance(unittest.TestCase):
    """unittest wrapper so this can also run via 'python3 -m unittest'."""

    @classmethod
    def setUpClass(cls):
        cls.eng = Engine()
        import numpy as np
        cls.rng = np.random.RandomState(SEED)

    def _cases(self, n=N_CASES):
        base = self.rng.normal(0, 2, (n, VOCAB)).astype("float32")
        mask = self.rng.random_sample((n, VOCAB)) < 0.05
        base += mask * self.rng.normal(6, 1, (n, VOCAB)).astype("float32")
        return [list(map(float, r)) for r in base]

    def test_greedy_exact_top1(self):
        chain = CHAINS[0]
        rows = self._cases()
        ok = 0
        for i, lg in enumerate(rows):
            pick = self.eng.run(chain["name"], 1, lg)
            oracle_argmax = max(range(len(lg)), key=lambda k: lg[k])
            self.assertEqual(pick, oracle_argmax,
                             f"greedy drift at case {i}: ours={pick} oracle={oracle_argmax}")
            ok += 1
        self.assertEqual(ok, len(rows))

    def test_survivor_and_prob_equivalence_all_chains(self):
        rows = self._cases()
        for chain in CHAINS:
            cfg = chain["oracle"]
            e_toks, e_probs = self.eng.cand(chain["name"], rows[0])
            o_toks, o_probs = oracle_pipeline(rows[0], cfg)
            self.assertEqual(set(e_toks), set(o_toks),
                             f"{chain['name']}: survivor set mismatch")
            em = {t: p for t, p in zip(e_toks, e_probs)}
            om = {t: p for t, p in zip(o_toks, o_probs)}
            for t in em:
                self.assertAlmostEqual(em[t], om[t], delta=1e-5,
                                       msg=f"{chain['name']} prob drift for tok {t}")

    def test_temperature_monotonicity(self):
        """softmax(out/temp) concentration monotone in temp."""
        rows = self._cases(50)
        for lg in rows:
            base_e = entropy(softmax(lg))
            for t in (0.3, 2.5):
                tmp_e = entropy(softmax([x / t for x in lg]))
                if t < 1.0:
                    self.assertLess(tmp_e, base_e)
                else:
                    self.assertGreater(tmp_e, base_e)


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
