"""
occlusion_analysis.py
=====================

Occlusion: remove a word and measure how much the score drops.
Occlusion: 단어를 지우고 점수가 얼마나 떨어지는지 직접 잰다.

No gradients: only forward passes, so it is exact for the change it makes,
and it works on any model, including black-box APIs.
기울기 없이 forward만 쓴다. 그래서 바꾼 만큼은 정확하고, 내부를 못 보는 모델에도 쓸 수 있다.

Score  S(X) = Z[<call>, y]   (the logit of tool y, same as tool_attribution's default)

Three ways to "remove" token i / 단어를 지우는 세 가지 방법
--------------------------------------------------------
  zero     X_i -> 0                         (same baseline as IG's X')
  mean     X_i -> mean of all embeddings    (a "typical" token)
  delete   drop the token, the sentence gets shorter (positions shift)

    occ_i = S(X) - S(X with token i removed)      (> 0: the token supported y)

Interactions / 상호작용 (pairs)
    I_ij = occ_{ij} - occ_i - occ_j
    occ_{ij} = drop when BOTH are removed. I_ij != 0 means the two tokens do not
    act independently; single-token occlusion (and any per-token score) misses that.

Agreement with IG / IG와의 일치
    Spearman rank correlation between occ_i (zero) and IG_i over the request words.

Usage:
    python occlusion_analysis.py                     # the draft sentence (wrong tool)
    python occlusion_analysis.py --text "the server shows an error so help me read the log file"
    python occlusion_analysis.py --summary           # all flipping edits
    python occlusion_analysis.py --augment --summary
"""

import argparse
import itertools
import sys

import numpy as np

from tool_data import TOOL_NAMES
from tool_attribution import attribute, encode_request
import long_sentence_study as L

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODES = ("zero", "mean", "delete")
DRAFT_B = "the draft shows an error so help me read the log file"


# ====================================================================
# Score with tokens removed
# ====================================================================

def score(model, ids, tool_id, remove=(), mode="zero"):
    """Return (S = Z[<call>, tool], predicted tool) with tokens in `remove` removed."""
    ids = list(ids)
    if mode == "delete":
        keep = [t for i, t in enumerate(ids) if i not in remove]
        logits, _, _ = model.forward(keep)
    else:
        X = model.p["E"][ids].copy()
        fill = 0.0 if mode == "zero" else model.p["E"].mean(0)
        for i in remove:
            X[i] = fill
        logits, _, _ = model.forward_embedded(X)
    z = logits[-1]
    return float(z[tool_id]), z


def predicted(z, token_to_id):
    tids = [token_to_id[t] for t in TOOL_NAMES]
    return TOOL_NAMES[int(np.argmax(z[tids]))]


def occlusion(model, token_to_id, ids, tool_id, mode):
    """Per request word i (1..k-1): drop in S and the tool predicted after removal."""
    s0, _ = score(model, ids, tool_id)
    k = len(ids) - 1
    drop, after = np.zeros(len(ids)), [None] * len(ids)
    for i in range(1, k):
        s, z = score(model, ids, tool_id, (i,), mode)
        drop[i], after[i] = s0 - s, predicted(z, token_to_id)
    return s0, drop, after


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ====================================================================
# One sentence
# ====================================================================

def explain(model, token_to_id, text, top_pairs=4):
    ids, toks = encode_request(token_to_id, text)
    k = len(ids) - 1
    s0, z0 = score(model, ids, 0)
    tool = predicted(z0, token_to_id)
    tid = token_to_id[tool]
    s0 = float(z0[tid])
    print(f"\nrequest : {text}\nselected: {tool}      S = Z[<call>, {tool}] = {s0:.3f}")

    res = {m: occlusion(model, token_to_id, ids, tid, m) for m in MODES}
    ig = attribute(model, ids, tid, "logit", None, 128)["ig"]

    print(f"\n  {'token':<10}" + "".join(f"{m:>9}" for m in MODES) + f"{'IG':>9}   tool if removed (zero)")
    for i in range(1, k):
        cells = "".join(f"{res[m][1][i]:>9.2f}" for m in MODES)
        changed = res["zero"][2][i]
        mark = changed if changed != tool else ""
        print(f"  {toks[i]:<10}{cells}{ig[i]:>9.2f}   {mark}")

    words = list(range(1, k))
    rho = {m: spearman(res[m][1][words], ig[words]) for m in MODES}
    print("\n  rank agreement with IG (Spearman): " + "  ".join(f"{m} {rho[m]:+.2f}" for m in MODES))

    # pairwise interactions among the strongest words (zero mode)
    d = res["zero"][1]
    top = sorted(words, key=lambda i: -abs(d[i]))[:top_pairs]
    print(f"\n  interactions among the top {len(top)} words (zero):  I_ij = occ_ij - occ_i - occ_j")
    for i, j in itertools.combinations(sorted(top), 2):
        s_ij, _ = score(model, ids, tid, (i, j), "zero")
        both = s0 - s_ij
        print(f"    {toks[i]:>8} + {toks[j]:<8} occ_ij {both:>8.2f}   sum of singles {d[i] + d[j]:>8.2f}"
              f"   I_ij {both - d[i] - d[j]:>+8.2f}")


# ====================================================================
# All flips
# ====================================================================

def summary(model, token_to_id, test):
    flips, n_checked, n_fragile = L.audit(model, token_to_id, test)
    restore = {m: 0 for m in MODES}
    top1 = {m: 0 for m in MODES}
    top1_ig, rhos = 0, []
    for x, pos, old, new, pred in flips:
        ids, _ = encode_request(token_to_id, L.edited_text(x, pos, new))
        tid, e = token_to_id[pred], pos + 1
        words = list(range(1, len(ids) - 1))
        for m in MODES:
            _, drop, after = occlusion(model, token_to_id, ids, tid, m)
            restore[m] += after[e] == x["tool"]
            top1[m] += int(max(words, key=lambda i: drop[i]) == e)
            if m == "zero":
                d_zero = drop
        ig = attribute(model, ids, tid, "logit", None, 32)["ig"]
        top1_ig += int(max(words, key=lambda i: ig[i]) == e)
        rhos.append(spearman(d_zero[words], ig[words]))

    n = len(flips)
    print(f"\n{n} flipping one-noun edits ({n_fragile}/{n_checked} requests); "
          f"score S = Z[<call>, wrong tool]")
    print(f"\n  {'':<34}" + "".join(f"{m:>9}" for m in MODES) + f"{'IG':>9}")
    print(f"  {'edited word is the top token':<34}" + "".join(f"{top1[m] / n:>9.0%}" for m in MODES)
          + f"{top1_ig / n:>9.0%}")
    print(f"  {'removing it restores the gold tool':<34}" + "".join(f"{restore[m] / n:>9.0%}" for m in MODES)
          + f"{'-':>9}")
    print(f"\n  Spearman(occlusion zero, IG) over request words: mean {np.mean(rhos):+.2f}, "
          f"median {np.median(rhos):+.2f}")


def main():
    ap = argparse.ArgumentParser(description="Occlusion analysis")
    ap.add_argument("--text")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--augment", action="store_true", help="use the augmented model")
    args = ap.parse_args()

    train, test = L.make_long_split(args.augment)
    path = L.MODEL_PATHS[args.augment]
    if not path.exists():
        sys.exit(f"{path.name} not found. Run: python long_sentence_study.py"
                 f"{' --augment' if args.augment else ''} --retrain")
    model, token_to_id = L.get_model(train, 20, False, path)

    if args.summary:
        summary(model, token_to_id, test)
    else:
        explain(model, token_to_id, (args.text or DRAFT_B).lower())


if __name__ == "__main__":
    main()
