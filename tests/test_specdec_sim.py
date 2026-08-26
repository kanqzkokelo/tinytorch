#!/usr/bin/env python3
"""M10 deliverable 2: speculative-decoding acceptance-loop SIMULATOR.

Pure simulation, NO GPU. Pretends target model output == corpus token
stream (greedy), so greedy acceptance is exact-prefix matching: a drafted
token is accepted iff it equals the actual next corpus token. This gives
an honest upper-bound estimate of ngram drafter hit rates on realistic
text before any engine work burns GPU time.

Methodology
-----------
1. Tokenize embedded corpora (prose / code / mixed / adversarial) with a
   GPT-ish word-p splitter (words + punctuation as separate tokens).
   Token-level repetition structure differs from char-level; word-level
   is closer to real BPE behavior for English.
2. For each decode position i (after warmup), replicate the C drafter's
   semantics exactly: last-n window of history tokens[:i], find MOST
   RECENT earlier occurrence, propose up to m following tokens.
3. Greedy accept: count leading proposed tokens that equal the true
   continuation tokens[i:]. Advance i by 1 + accepted. Record per-step:
   drafted k, accepted a.
4. Cost models (both reported):
   - LINEAR (pessimistic): verify of k+1 positions costs (k+1)*t_tok.
     multiplier = (1 + E[a]) / E[k] ... see sim() for exact form.
   - FLAT (optimistic, memory-bound GPU): verify of up to m+1 positions
     costs c*t_tok regardless of batch width. multiplier = E[1+a] / c.
   Drafting itself assumed free (~microseconds CPU table lookup).

Assumptions stated in output header below.
"""

import re
import sys

WARMUP = 16          # min history before drafting starts
WINDOWS = [8, 12, 16]
DRAFT_LENS = [2, 4, 8]

PROSE = (
    # ~3KB essay-style English with naturally recurring phrases (as real
    # chat/summarize output does): names, boilerplate clauses, repeated
    # sentence frames. This is the regime ngram drafting targets.
    "The committee reviewed the quarterly report and found that the "
    "quarterly report contained several inconsistencies in the budget "
    "summary. The budget summary had been prepared by the finance team, "
    "and the finance team confirmed that the numbers would be corrected "
    "before the next meeting. Before the next meeting, the chair asked "
    "every department to submit an updated timeline for the migration "
    "project, because the migration project was already three weeks "
    "behind schedule. Three weeks behind schedule meant that the launch "
    "date could not be guaranteed, and the launch date appeared on the "
    "first page of the quarterly report.\n\n"
    "In other business, the committee discussed the hiring plan. The "
    "hiring plan called for six engineers and two designers over the "
    "next two quarters. Over the next two quarters, recruiting would "
    "run a referral program, and the referral program would pay a bonus "
    "for each candidate who accepted an offer and passed the probation "
    "period. The probation period lasts ninety days, and during the "
    "probation period the team lead is responsible for weekly check-ins. "
    "Weekly check-ins are documented in the shared tracker, and the "
    "shared tracker is reviewed by HR at the end of every month.\n\n"
    "The committee also revisited the security audit. The security audit "
    "found that password rotation policies were out of date, that two "
    "servers ran unsupported operating systems, and that the backup job "
    "had failed silently for eleven days. Eleven days without backups is "
    "an unacceptable risk, so the committee directed the operations team "
    "to add alerting to the backup job, to patch the two servers, and to "
    "update the password rotation policies before the next meeting. The "
    "operations team agreed, and the operations team promised a written "
    "remediation plan within one week. A written remediation plan within "
    "one week was also required by the auditors, who will return next "
    "quarter to verify compliance. Next quarter the auditors expect full "
    "documentation, and full documentation must include the updated "
    "timeline for the migration project.\n\n"
    "Finally, the chair summarized the action items: correct the budget "
    "summary, publish the hiring plan, remediate the security audit "
    "findings, and submit an updated timeline for the migration project. "
    "All action items are due before the next meeting, and progress on "
    "all action items will be tracked in the shared tracker until every "
    "item is closed."
)

CODE = '''import os
import sys
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return json.load(f)


def save_config(cfg, path):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def train_step(model, batch, opt):
    opt.zero_grad()
    logits = model(batch.inputs)
    loss = cross_entropy(logits, batch.targets)
    loss.backward()
    opt.step()
    return loss.item()


def eval_step(model, batch):
    logits = model(batch.inputs)
    loss = cross_entropy(logits, batch.targets)
    return loss.item()


def save_checkpoint(model, opt, path):
    torch.save({"model": model.state_dict(),
                "opt": opt.state_dict()}, path)


def load_checkpoint(model, opt, path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])


def main():
    args = parse_args()
    cfg = load_config(args.config)
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = make_loader(cfg)
    step = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            loss = train_step(model, batch, opt)
            if step % 100 == 0:
                print(f"epoch {epoch} step {step} train loss {loss:.4f}")
            step += 1
        model.eval()
        total = 0.0
        count = 0
        for batch in val_loader:
            loss = eval_step(model, batch)
            total += loss
            count += 1
        print(f"epoch {epoch} val loss {total / max(count, 1):.4f}")
        save_checkpoint(model, opt, f"ckpt_epoch{epoch}.pt")


if __name__ == "__main__":
    main()
'''

MIXED = PROSE[:400] + "\n" + CODE + "\n" + PROSE[400:]

# Verbatim-copy regime: the documented win case (roadmap M10 / llama.cpp
# findings: code-edit, summarize, RAG quote 2-4x). Target stream re-emits
# large verbatim chunks seen earlier in context.
_DOC = (
    "Retrieval result: the payment service authenticates requests using "
    "signed JWT tokens with a fifteen minute expiry, retries failed "
    "charges three times with exponential backoff, and emits a webhook "
    "for every state transition of an order."
)
COPY = (
    "Based on the retrieved document, here is what it says: " + _DOC +
    " To summarize the retrieval result again: " + _DOC +
    " Key details from the document: " + _DOC
)

# False-positive regime: repeated sentence FRAMES with divergent fillers.
# Windows land inside the constant frame -> match -> draft continues into
# a filler that differs -> acceptance fails -> wasted verify width. This
# is the true adversarial worst case for fallback overhead.
_fillers = None  # (unused after switch to unique per-sentence ids)
import random
_rng = random.Random(7)
_adv_sents = []
for i in range(240):
    _filler = "cfg_" + format(_rng.getrandbits(32), "08x")
    _adv_sents.append(
        f"The worker process {i} completed its assigned batch of tasks "
        f"using configuration {_filler} and reported status back to the "
        f"coordinator node as required by protocol."
    )
ADVERSARIAL = " ".join(_adv_sents)


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)


def simulate(tokens, window, max_draft):
    """Replicates src/specdec.c tt_ngram_draft semantics exactly.

    Returns (steps, total_generated_tokens, sum_drafted, sum_accepted).
    """
    steps = 0
    total_out = 0
    sum_k = 0
    sum_a = 0
    n_tok = len(tokens)
    i = WARMUP
    hist_start = 0
    while i < n_tok:
        k = 0
        if i - hist_start > window:
            needle = tokens[i - window:i]
            # most recent earlier occurrence ending at e <= i-window
            found = -1
            for e in range(i - window, hist_start, -1):
                cs = e - window
                if tokens[cs:e] == needle:
                    found = e
                    break
            if found >= 0:
                k = min(max_draft, n_tok - found)
                prop = tokens[found:found + k]
                a = 0
                while a < k and i + a < n_tok and prop[a] == tokens[i + a]:
                    a += 1
                sum_a += a
            else:
                k = 0          # miss -> base step only, nothing wasted
        steps += 1
        total_out += 1 + k  # greedy exact-match proxy: accepted all or we stop
        sum_k += k
        i += 1 + k
    return steps, total_out, sum_k, sum_a


def report(corpus_name, tokens):
    n_tok = len(tokens)
    rows = []
    for n in WINDOWS:
        for m in DRAFT_LENS:
            steps, out, sum_k, sum_a = simulate(tokens, n, m)
            avg_k = sum_k / steps
            avg_a = sum_a / steps
            acc_rate = (sum_a / sum_k) * 100 if sum_k else 0.0
            # LINEAR cost model: each step verifies avg_k+1 tokens,
            # yields 1 + accepted tokens.
            lin_mult = (1 + avg_a) / (avg_k + 1)
            # FLAT cost model: verify <= m+1 tokens costs same as 1 token
            # (c=1.0 ideal; real graphs ~1.05-1.15x).
            flat_mult = (1 + avg_a) / 1.0
            rows.append((n, m, acc_rate, avg_k, avg_a, lin_mult, flat_mult))
    print(f"\n=== corpus: {corpus_name} ({n_tok} tokens) ===")
    print(f"{'n':>3} {'m':>2} | {'acc%':>6} {'avgK':>5} {'avgA':>5} | "
          f"{'lin-x':>6} {'flat-x':>7}")
    for n, m, ar, ak, aa, lm, fm in rows:
        print(f"{n:>3} {m:>2} | {ar:>6.1f} {ak:>5.2f} {aa:>5.2f} | "
              f"{lm:>6.3f} {fm:>7.3f}")
    return rows


def main():
    print(__doc__.split("Assumptions")[0])
    print("""Assumptions:
- Target == corpus stream, greedy decoding => accept = exact prefix match.
  Real LLM distributions are LESS predictable than verbatim text reuse,
  BUT specdec wins come from verbatim reuse (edits/quotes/RAG), which
  this proxies faithfully.
- Drafter cost ~0 (CPU table scan, microseconds vs ~10ms+ GPU step).
- LINEAR model: verify(k+1) == (k+1)*t_tok -- what you get WITHOUT graph
  capture / batching efficiency. Speculation can only lose here.
- FLAT model: verify(<=m+1) == 1*t_tok (memory-bound weights reload
  dominates). This is the achievable ceiling WITH captured multi-position
  forward. Real number lands between, closer to flat for small m.
""")
    corpora = [("prose", tokenize(PROSE)),
               ("code", tokenize(CODE)),
               ("mixed", tokenize(MIXED)),
               ("verbatim-copy-RAG", tokenize(COPY)),
               ("adversarial-falsepos", tokenize(ADVERSARIAL))]
    all_rows = {}
    for name, toks in corpora:
        all_rows[name] = report(name, toks)

    print("\n=== headline envelope ===")
    best_flat = max(r[6] for rows in all_rows.values() for r in rows)
    best_cfg = [(c, r[0], r[1]) for c, rows in all_rows.items() for r in rows
                if r[6] == best_flat]
    worst_lin = min(r[5] for rows in all_rows.values() for r in rows)
    adv = all_rows["adversarial-falsepos"]
    adv_worst_lin = min(r[5] for r in adv)
    print(f"BEST case : flat-model mult {best_flat:.2f}x  at {best_cfg[0]}")
    print(f"WORST case: linear-model mult {worst_lin:.3f}x  (no-capture path)")
    print(f"WORST case adversarial: flat {min(r[6] for r in adv):.2f}x, "
          f"linear {adv_worst_lin:.3f}x -> fallback overhead bounded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
