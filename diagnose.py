"""
diagnose.py
===========

Combined diagnosis of one wrong decision: IG, occlusion, layer-wise
hidden-state diff and attention on the SAME pair, then the training data.
오판 하나를 종합 진단: 같은 사례에 IG, occlusion, 층별 hidden state 비교,
attention을 모두 적용하고, 마지막으로 학습 데이터를 본다.

    A: correct input      B: the same input with one context noun changed (wrong)

  1. IG          S = Z[<call>, wrong] on B      which token raised the wrong tool?
  2. occlusion   remove each token of B         which removal brings the right tool back?
  3. hidden      A vs B, stage by stage         from which layer does <call> stay wrong?
  4. attention   that layer, row <call>         does <call> take its information from the token?
  5. data        training co-occurrence         why did the model learn that link?

One method gives a hypothesis; when several point at the same token through
a consistent path, that is evidence. When they disagree, that is a finding too.
기법 하나는 가설이고, 여러 기법이 같은 토큰과 같은 경로를 가리키면 근거가 된다.
어긋나면 그것도 발견이다.

Usage:
    python diagnose.py                  # the draft flip
    python diagnose.py --index 5        # the 5th flip found by the audit
    python diagnose.py --all            # agreement of the methods over all flips
    python diagnose.py --augment --all
"""

import argparse
import sys
from collections import Counter

import numpy as np

from tool_attribution import attribute, encode_request
import long_sentence_study as L
import hidden_state_diff as HSD
import attention_analysis as ATT
import occlusion_analysis as OCC

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DRAFT = (ATT.DRAFT_A, ATT.DRAFT_B)


def mark(ok):
    return "✓" if ok else "✗"


# ====================================================================
# One pair -> all four methods
# ====================================================================

def run(model, token_to_id, text_a, text_b, gold, train=None, quiet=False):
    ids_a, toks_a = encode_request(token_to_id, text_a)
    ids_b, toks_b = encode_request(token_to_id, text_b)
    edited = [i for i in range(len(ids_a)) if ids_a[i] != ids_b[i]]
    e = edited[0]
    k = len(ids_b) - 1
    words = list(range(1, k))
    wrong = L.predict(model, token_to_id, ids_b)
    tid = token_to_id[wrong]

    # 1. IG on B
    ig = attribute(model, ids_b, tid, "logit", None, 64)["ig"]
    ig_top = max(words, key=lambda i: ig[i])

    # 2. occlusion on B (zero)
    _, drop, after = OCC.occlusion(model, token_to_id, ids_b, tid, "zero")
    occ_top = max(words, key=lambda i: drop[i])
    restores = after[e] == gold

    # 3. hidden-state diff
    stage, names, SA, SB = HSD.first_wrong_stage(model, token_to_id, ids_a, ids_b, gold, wrong)
    call_diff = [HSD.rel_diff(a, b)[k] for a, b in zip(SA, SB)]

    # 4. attention at the layer where B settles on the wrong tool
    layer = int(stage[1]) - 1 if stage != "never" else model.N - 1
    _, mA, nA = ATT.row_views(ATT.attention(model, ids_a), k)
    _, mB, nB = ATT.row_views(ATT.attention(model, ids_b), k)
    att_top = int(np.argmax(nB[layer][:k]))
    att_top_a = int(np.argmax(nA[layer][:k]))

    r = dict(e=e, word=toks_b[e], wrong=wrong, stage=stage, layer=layer,
             ig=ig_top == e, occ=occ_top == e, restores=restores, att=att_top == e)
    if quiet:
        return r

    def top3(v):
        return ", ".join(f"{toks_b[i]} {v[i]:.1f}" for i in sorted(words, key=lambda i: -v[i])[:3])

    print(f"\nA: {text_a}\n   -> {gold}  (correct)")
    print(f"B: {text_b}\n   -> {wrong}  (wrong)")
    print(f"edited: '{toks_a[e]}' -> '{toks_b[e]}'")
    print("\n" + "-" * 78)
    print(f" 1. IG           S = Z[<call>, {wrong}] on B")
    print(f"                 top: {top3(ig)}")
    print(f"                 -> '{toks_b[ig_top]}'  {mark(ig_top == e)} edited word")
    print(f" 2. occlusion    remove one token of B (zero)")
    print(f"                 top drop: {top3(drop)}")
    print(f"                 removing '{toks_b[e]}' -> {after[e]}  {mark(restores)} back to the correct tool")
    print(f" 3. hidden state <call> relative diff A vs B: "
          + "  ".join(f"{n} {d:.2f}" for n, d in zip(names, call_diff)))
    print(f"                 <call> stays on the wrong tool from {stage}")
    print(f" 4. attention    layer {layer + 1}, what <call> takes the most from (norm)")
    print(f"                 A: '{toks_a[att_top_a]}' {nA[layer][att_top_a]:.1f}    "
          f"B: '{toks_b[att_top]}' {nB[layer][att_top]:.1f}   {mark(att_top == e)} edited word")
    print(f"                 weight to '{toks_b[e]}': A {mA[layer][e]:.2f} -> B {mB[layer][e]:.2f}")
    if train is not None:
        co = Counter(x["tool"] for x in train if toks_b[e] in x["text"].split())
        print(f" 5. data         '{toks_b[e]}' in training: "
              + ", ".join(f"{t} {c}" for t, c in co.most_common()))

    agree = sum([r["ig"], r["occ"], r["att"]])
    print("\n" + "=" * 78)
    if agree == 3 and restores:
        print(f" VERDICT: '{toks_b[e]}' caused the wrong decision.")
        print(f"   IG, occlusion and attention all point at it; removing it restores {gold};")
        print(f"   it reaches <call> through the layer-{layer + 1} attention and <call> stays on "
              f"{wrong} from {stage}.")
    else:
        print(f" VERDICT: mixed — IG {mark(r['ig'])}  occlusion {mark(r['occ'])}  "
              f"attention {mark(r['att'])}  restore {mark(restores)}")
        print("   the methods disagree; look at the tables above before concluding.")
    print("=" * 78)
    return r


# ====================================================================
# All flips
# ====================================================================

def run_all(model, token_to_id, test):
    flips, n_checked, n_fragile = L.audit(model, token_to_id, test)
    R = [run(model, token_to_id, x["text"], L.edited_text(x, pos, new), x["tool"], quiet=True)
         for x, pos, old, new, pred in flips]
    n = len(R)
    print(f"\n{n} flipping one-noun edits ({n_fragile}/{n_checked} requests)")
    print("\n  does the method point at the edited word?")
    for key, label in (("ig", "IG top token"), ("occ", "occlusion top token"),
                       ("restores", "removing it restores the tool"),
                       ("att", "attention top source (that layer)")):
        c = sum(r[key] for r in R)
        print(f"    {label:<34} {c:>4}/{n}  {c / n:>5.0%}")
    allk = sum(r["ig"] and r["occ"] and r["att"] and r["restores"] for r in R)
    print(f"    {'all four agree':<34} {allk:>4}/{n}  {allk / n:>5.0%}")

    print("\n  by the stage where <call> settles on the wrong tool:")
    print(f"    {'stage':<10}{'flips':>7}{'all agree':>11}{'attention ✓':>13}")
    for stage in ("L1.attn", "L1.ffn", "L2.attn", "L2.ffn", "never"):
        sub = [r for r in R if r["stage"] == stage]
        if not sub:
            continue
        a = sum(r["ig"] and r["occ"] and r["att"] and r["restores"] for r in sub)
        t = sum(r["att"] for r in sub)
        print(f"    {stage:<10}{len(sub):>7}{a / len(sub):>11.0%}{t / len(sub):>13.0%}")

    mixed = [r for r in R if not (r["ig"] and r["occ"] and r["att"])]
    if mixed:
        print("\n  most common edited words where the methods disagree:")
        for w, c in Counter(r["word"] for r in mixed).most_common(5):
            print(f"    {w:<12} {c}")


def main():
    ap = argparse.ArgumentParser(description="Combined diagnosis of a wrong decision")
    ap.add_argument("--index", type=int, help="diagnose the N-th flip found by the audit")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--augment", action="store_true", help="use the augmented model")
    args = ap.parse_args()

    train, test = L.make_long_split(args.augment)
    path = L.MODEL_PATHS[args.augment]
    if not path.exists():
        sys.exit(f"{path.name} not found. Run: python long_sentence_study.py"
                 f"{' --augment' if args.augment else ''} --retrain")
    model, token_to_id = L.get_model(train, 20, False, path)

    if args.all:
        run_all(model, token_to_id, test)
        return
    if args.index is not None:
        flips, _, _ = L.audit(model, token_to_id, test)
        x, pos, old, new, pred = flips[args.index % len(flips)]
        run(model, token_to_id, x["text"], L.edited_text(x, pos, new), x["tool"], train)
    else:
        a_ids, _ = encode_request(token_to_id, DRAFT[0])
        run(model, token_to_id, DRAFT[0], DRAFT[1], L.predict(model, token_to_id, a_ids), train)


if __name__ == "__main__":
    main()
