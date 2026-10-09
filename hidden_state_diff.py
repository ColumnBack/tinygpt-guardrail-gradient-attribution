"""
hidden_state_diff.py
====================

Layer-wise hidden-state diff: AT WHICH LAYER do a correct input and a
wrong input part ways?
층별 hidden state 비교: 정답 입력과 오답 입력이 몇 번째 층에서 갈라지나?

Inputs / 입력
-------------
A pair from long_sentence_study.py that differs in ONE context noun
(same length, same positions), e.g.

    A: the server shows an error so help me read the log file   -> read_file (correct)
    B: the draft  shows an error so help me read the log file   -> write_file (wrong)

Stages / 단계  (tinygpt.block_forward, read-only; tinygpt.py is unchanged)
--------------
    embed    h = E[ids] + P
    Ln.attn  Y   = LN1(X + Attn(X))          after the attention sublayer
    Ln.ffn   X'  = LN2(Y + FFN(Y))           after the FFN sublayer (= block output)
The last stage is H, the hidden state the LM head reads: Z = H E^T.

What is printed / 출력
---------------------
  1. relative diff per stage and position:  ||h_B[i] - h_A[i]|| / ||h_A[i]||
       where the edit starts (the edited word) and when it reaches <call>
  2. logit lens at <call>: read each stage through the LM head,
       z(h) = h E^T,  margin = z[gold] - z[wrong]
       -> the stage from which the wrong tool stays ahead to the end
       (intermediate stages are not trained to be read out, so this is an
        approximate view; the last stage is the real output)
  3. which sublayer carried the change at <call>:
       ||Attn_B - Attn_A||  vs  ||FFN_B - FFN_A||   per block
  --summary: over every flipping edit, the stage from which <call> keeps
       preferring the wrong tool, and the <call> diff for flipping vs
       non-flipping edits

Usage:
    python hidden_state_diff.py                       # the draft example
    python hidden_state_diff.py --summary             # all flips
    python hidden_state_diff.py --augment             # same, on the augmented model
    python hidden_state_diff.py --pair "the server shows an error so help me read the log file" \\
                                       "the draft shows an error so help me read the log file"
"""

import argparse
import sys
from collections import Counter

import numpy as np

from tool_data import TOOL_NAMES
from tool_attribution import encode_request
import long_sentence_study as L

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SHADES = " ░▒▓█"


# ====================================================================
# Hidden states at every stage
# ====================================================================

def hidden_states(model, ids):
    """Return (names, states, sub).

    states[s] : (T x d) hidden state after stage s
    sub[n]    : dict(attn=O, ffn=F) residual updates of block n (T x d)
    """
    ids = np.asarray(ids, dtype=np.int64)
    X = model.p["E"][ids] + model.p["P"][:len(ids)]
    names, states, sub = ["embed"], [X], []
    for n in range(model.N):
        X, cache = model.block_forward(X, n)
        O = cache["Ocat"] @ model.p[f"WO_{n}"]          # attention output
        F = cache["Z"] - cache["Y"]                       # FFN output
        names += [f"L{n + 1}.attn", f"L{n + 1}.ffn"]
        states += [cache["Y"], X]
        sub.append(dict(attn=O, ffn=F))
    return names, states, sub


def rel_diff(hA, hB):
    """Per-position relative L2 difference."""
    return np.linalg.norm(hB - hA, axis=1) / (np.linalg.norm(hA, axis=1) + 1e-12)


def layer_score(names, SA, SB):
    """Layer-level score (Interpretability math.pdf p.2), on the block outputs H^(l):
        D_l = (1/T) sum_i || H_i^{F,(l)} - H_i^{N,(l)} ||_2,     l* = argmax_l D_l
    N = normal (correct) input A, F = wrong input B. Returns ({layer: D_l}, l*)."""
    D = {name.split(".")[0]: float(np.linalg.norm(b - a, axis=1).mean())
         for name, a, b in zip(names, SA, SB) if name.endswith(".ffn")}
    return D, max(D, key=D.get)


def layer_increment(names, SA, SB):
    """New divergence added by each block (D_l is cumulative, so l* drifts to the last layer):
        dD_l = D_l - D_{l-1},   D_0 = the same score at the embedding (block 1's input)
    Returns ({layer: dD_l}, argmax_l dD_l)."""
    mean_diff = lambda a, b: float(np.linalg.norm(b - a, axis=1).mean())
    prev = mean_diff(SA[0], SB[0])                          # embed
    dD = {}
    for name, a, b in zip(names, SA, SB):
        if name.endswith(".ffn"):
            cur = mean_diff(a, b)
            dD[name.split(".")[0]] = cur - prev
            prev = cur
    return dD, max(dD, key=dD.get)


def settle_stage(margins, names):
    """First stage s >= 1 from which every later margin is < 0 (wrong tool ahead)."""
    for s in range(1, len(margins)):
        if all(m < 0 for m in margins[s:]):
            return names[s]
    return "never"


def lens_margin(model, h_k, gold_id, wrong_id):
    """Logit lens at one position: z = h E^T, return z[gold] - z[wrong]."""
    z = h_k @ model.p["E"].T
    return float(z[gold_id] - z[wrong_id])


def shade(v, vmax):
    if vmax <= 0:
        return SHADES[0]
    return SHADES[min(len(SHADES) - 1, int(round(v / vmax * (len(SHADES) - 1))))]


# ====================================================================
# One pair
# ====================================================================

def compare(model, token_to_id, text_a, text_b, gold=None):
    ids_a, toks_a = encode_request(token_to_id, text_a)
    ids_b, toks_b = encode_request(token_to_id, text_b)
    if len(ids_a) != len(ids_b):
        sys.exit("the two sentences must have the same number of words")
    pred_a = L.predict(model, token_to_id, ids_a)
    pred_b = L.predict(model, token_to_id, ids_b)
    gold = gold or pred_a
    wrong = pred_b if pred_b != gold else pred_a
    g_id, w_id = token_to_id[gold], token_to_id[wrong]
    k = len(ids_a) - 1                                    # <call>
    edited = [i for i in range(len(ids_a)) if ids_a[i] != ids_b[i]]

    names, SA, subA = hidden_states(model, ids_a)
    _, SB, subB = hidden_states(model, ids_b)

    print(f"\nA: {text_a}\n   -> {pred_a}")
    print(f"B: {text_b}\n   -> {pred_b}")
    print(f"edited position(s): {[toks_a[i] + '->' + toks_b[i] for i in edited]}")

    # 1 ----------------------------------------------------------------
    D = np.array([rel_diff(a, b) for a, b in zip(SA, SB)])   # stages x T
    vmax = D.max()
    toks = [toks_b[i] if i in edited else toks_a[i] for i in range(len(toks_a))]
    print("\n 1. relative diff ||h_B - h_A|| / ||h_A||  per stage and position")
    print("    " + " " * 10 + "".join(f"{t[:6]:>7}" for t in toks))
    for s, name in enumerate(names):
        cells = "".join(f"{D[s, i]:>7.2f}" for i in range(len(toks)))
        print(f"    {name:<10}{cells}")
    print("    " + " " * 10 + "".join(f"{shade(D[-1, i], vmax) * 3:>7}" for i in range(len(toks)))
          + "   <- last stage, shaded")
    print(f"    at <call>: " + "  ".join(f"{n} {D[s, k]:.2f}" for s, n in enumerate(names)))
    Dl, lstar = layer_score(names, SA, SB)
    print("    layer score D_l = (1/T) sum_i ||H_B - H_A||_2 on block outputs:  "
          + "  ".join(f"{l} {v:.2f}" for l, v in Dl.items()) + f"   -> l* = {lstar}")
    dD, dstar = layer_increment(names, SA, SB)
    print("    new divergence per block dD_l = D_l - D_(l-1) (D_0 = embed):        "
          + "  ".join(f"{l} {v:+.2f}" for l, v in dD.items()) + f"   -> argmax = {dstar}")

    # 2 ----------------------------------------------------------------
    print(f"\n 2. logit lens at <call>:  z[{gold}] - z[{wrong}]   (> 0: correct tool ahead)")
    MA = [lens_margin(model, h[k], g_id, w_id) for h in SA]
    MB = [lens_margin(model, h[k], g_id, w_id) for h in SB]
    first = settle_stage(MB, names)
    for s, name in enumerate(names):
        flag = "   <- from here B stays on the wrong tool" if name == first else ""
        print(f"    {name:<10} A {MA[s]:>8.2f}   B {MB[s]:>8.2f}{flag}")

    # 3 ----------------------------------------------------------------
    print("\n 3. change of each sublayer's output at <call>  (||B - A||)")
    for n in range(model.N):
        da = np.linalg.norm(subB[n]["attn"][k] - subA[n]["attn"][k])
        df = np.linalg.norm(subB[n]["ffn"][k] - subA[n]["ffn"][k])
        big = "attention" if da > df else "FFN"
        print(f"    block {n + 1}:  attention {da:>7.2f}   FFN {df:>7.2f}   -> mostly {big}")
    return first


# ====================================================================
# All flips
# ====================================================================

def first_wrong_stage(model, token_to_id, ids_a, ids_b, gold, wrong):
    names, SA, _ = hidden_states(model, ids_a)
    _, SB, _ = hidden_states(model, ids_b)
    k = len(ids_a) - 1
    MB = [lens_margin(model, h[k], token_to_id[gold], token_to_id[wrong]) for h in SB]
    return settle_stage(MB, names), names, SA, SB


def summary(model, token_to_id, test):
    flips, n_checked, n_fragile = L.audit(model, token_to_id, test)
    stages, lstars, dstars = Counter(), Counter(), Counter()
    d_flip, d_keep = [], []
    seen = set()
    for x, pos, old, new, pred in flips:
        ids_a, _ = encode_request(token_to_id, x["text"])
        ids_b, _ = encode_request(token_to_id, L.edited_text(x, pos, new))
        st, names, SA, SB = first_wrong_stage(model, token_to_id, ids_a, ids_b, x["tool"], pred)
        stages[st] += 1
        lstars[layer_score(names, SA, SB)[1]] += 1
        dstars[layer_increment(names, SA, SB)[1]] += 1
        k = len(ids_a) - 1
        d_flip.append([rel_diff(a, b)[k] for a, b in zip(SA, SB)])
        # a non-flipping noun swap at the same position, once per (sentence, position)
        if (x["text"], pos) in seen:
            continue
        seen.add((x["text"], pos))
        for alt in L.NOUNS:
            if alt in (old, new) or alt not in token_to_id:
                continue
            ids_c, _ = encode_request(token_to_id, L.edited_text(x, pos, alt))
            if L.predict(model, token_to_id, ids_c) == x["tool"]:
                _, SC_states, _ = hidden_states(model, ids_c)
                d_keep.append([rel_diff(a, c)[k] for a, c in zip(SA, SC_states)])
                break

    print(f"\n{len(flips)} flipping one-noun edits ({n_fragile}/{n_checked} requests)")
    print("\n stage from which <call> keeps preferring the wrong tool (logit lens):")
    for name in names + ["never"]:
        c = stages.get(name, 0)
        if c:
            print(f"    {name:<10} {c:>4}  {'#' * max(1, round(c / len(flips) * 40))}")
    print("\n mean relative diff at <call> per stage:")
    print(f"    {'stage':<10}{'flipping':>10}{'non-flipping':>14}")
    F, K = np.mean(d_flip, 0), np.mean(d_keep, 0) if d_keep else None
    for s, name in enumerate(names):
        kv = f"{K[s]:>14.3f}" if K is not None else f"{'-':>14}"
        print(f"    {name:<10}{F[s]:>10.3f}{kv}")
    print(f"    (non-flipping: {len(d_keep)} swaps of the same noun position that kept the tool)")
    print("\n l* = argmax_l D_l  vs  argmax_l dD_l (block that ADDS the most divergence):")
    print(f"    {'layer':<10}{'l* (D_l)':>10}{'argmax dD_l':>14}")
    for l in sorted(set(lstars) | set(dstars)):
        print(f"    {l:<10}{lstars.get(l, 0):>10}{dstars.get(l, 0):>14}")


# ====================================================================
# Main
# ====================================================================

def main():
    ap = argparse.ArgumentParser(description="Layer-wise hidden-state diff")
    ap.add_argument("--pair", nargs=2, metavar=("TEXT_A", "TEXT_B"))
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
    elif args.pair:
        compare(model, token_to_id, args.pair[0].lower(), args.pair[1].lower())
    else:
        compare(model, token_to_id,
                "the server shows an error so help me read the log file",
                "the draft shows an error so help me read the log file")


if __name__ == "__main__":
    main()
