"""
attention_analysis.py
=====================

Attention analysis: WHERE does the decision position <call> look, and
does that change when one context word is edited?
Attention 분석: 결정 위치 <call>은 어디를 보고 있나, 단어 하나를 바꾸면 그게 달라지나?

Read from tinygpt.block_forward caches (tinygpt.py is unchanged):
    A^{(n,h)}  = softmax(Q K^T / sqrt(d_h) + causal mask)     (T x T), row k = <call>

Three views of the same row k / 같은 k행을 세 가지로 본다
-----------------------------------------------------------
  weight     A^{(n,h)}[k, j]                       raw attention weight
  norm       || sum_h A^{(n,h)}[k, j] * V^{(n,h)}[j] W_O^{(n,h)} ||
             how much token j actually adds to <call>'s attention output
             (weight x size of what j sends; Kobayashi et al., 2020)
  rollout    R = A~_N ... A~_1,  A~_n = 0.5 * mean_h A^{(n,h)} + 0.5 * I
             flow from the input tokens to <call> through all layers,
             with the residual path as the 0.5 * I term (Abnar & Zuidema, 2020).
             Approximate: it ignores the FFN and LayerNorm.

A large weight does not mean a large influence: a token can get attention
but send a small vector. That is why "norm" is printed next to "weight".
가중치가 커도 보내는 값이 작으면 영향은 작다. 그래서 weight 옆에 norm을 같이 본다.

Usage:
    python attention_analysis.py                     # the draft pair
    python attention_analysis.py --summary           # all flipping edits vs non-flipping ones
    python attention_analysis.py --augment --summary
    python attention_analysis.py --pair "TEXT_A" "TEXT_B"
"""

import argparse
import sys

import numpy as np

from tool_attribution import encode_request
import long_sentence_study as L

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DRAFT_A = "the server shows an error so help me read the log file"
DRAFT_B = "the draft shows an error so help me read the log file"


# ====================================================================
# Attention maps and per-source contributions
# ====================================================================

def attention(model, ids):
    """Return per layer: dict(A=(H,T,T) weights, contrib=(H,T,T,d) per-head,
    per-source vectors A[q,j] * V[j] W_O_h)."""
    ids = np.asarray(ids, dtype=np.int64)
    X = model.p["E"][ids] + model.p["P"][:len(ids)]
    layers = []
    for n in range(model.N):
        X_in = X
        X, cache = model.block_forward(X_in, n)
        WO = model.p[f"WO_{n}"]
        A_all, C_all = [], []
        for h, hc in enumerate(cache["head_caches"]):
            V, A = hc[3], hc[4]
            WO_h = WO[h * model.dh:(h + 1) * model.dh]          # this head's rows of W_O
            VW = V @ WO_h                                       # (T, d): what each source sends
            A_all.append(A)
            C_all.append(A[:, :, None] * VW[None, :, :])        # (T, T, d)
        layers.append(dict(A=np.array(A_all), contrib=np.array(C_all)))
    return layers


def row_views(layers, k):
    """For query row k: weight (N,H,T), head-mean weight (N,T), norm (N,T)."""
    weight = np.array([ly["A"][:, k, :] for ly in layers])                       # N,H,T
    norm = np.array([np.linalg.norm(ly["contrib"][:, k].sum(0), axis=-1)       # sum heads
                     for ly in layers])                                        # N,T
    return weight, weight.mean(1), norm


def rollout(layers):
    T = layers[0]["A"].shape[-1]
    R = np.eye(T)
    for ly in layers:
        At = 0.5 * ly["A"].mean(0) + 0.5 * np.eye(T)
        At /= At.sum(1, keepdims=True)
        R = At @ R
    return R


# ====================================================================
# One pair
# ====================================================================

def bar(v, vmax, width=14):
    n = int(round(v / vmax * width)) if vmax > 0 else 0
    return "█" * n


def show_sentence(model, token_to_id, text, label):
    ids, toks = encode_request(token_to_id, text)
    k = len(ids) - 1
    layers = attention(model, ids)
    weight, wmean, norm = row_views(layers, k)
    R = rollout(layers)[k]
    pred = L.predict(model, token_to_id, ids)
    print(f"\n{label}: {text}\n   -> {pred}")
    print(f"   attention FROM <call> TO each token")
    head = "".join(f"   L{n + 1} w   L{n + 1} norm" for n in range(model.N))
    print(f"   {'token':<10}{head}   rollout")
    for j, t in enumerate(toks):
        cells = "".join(f"{wmean[n, j]:>8.2f}{norm[n, j]:>10.2f}" for n in range(model.N))
        print(f"   {t:<10}{cells}   {R[j]:>6.2f} {bar(R[j], R.max())}")
    return ids, toks, k, layers, weight, wmean, norm, R


def compare(model, token_to_id, text_a, text_b):
    ids_a, toks_a, k, _, wA, mA, nA, RA = show_sentence(model, token_to_id, text_a, "A")
    ids_b, toks_b, _, _, wB, mB, nB, RB = show_sentence(model, token_to_id, text_b, "B")
    edited = [i for i in range(len(ids_a)) if ids_a[i] != ids_b[i]]
    if len(ids_a) != len(ids_b) or not edited:
        return
    e = edited[0]
    print(f"\n edited position: '{toks_a[e]}' -> '{toks_b[e]}'   (attention from <call> to it)")
    print(f"   {'layer':<7}{'head weights A':<28}{'head weights B':<28}{'norm A':>8}{'norm B':>8}"
          f"{'rank B (norm)':>15}")
    for n in range(model.N):
        ha = " ".join(f"{x:.2f}" for x in wA[n, :, e])
        hb = " ".join(f"{x:.2f}" for x in wB[n, :, e])
        rank = int((nB[n] > nB[n, e]).sum()) + 1
        print(f"   L{n + 1:<6}{ha:<28}{hb:<28}{nA[n, e]:>8.2f}{nB[n, e]:>8.2f}{rank:>12} of {len(toks_b)}")
    print(f"   rollout  A {RA[e]:.3f}   B {RB[e]:.3f}")

    # Interpretability math.pdf p.6:  j* = argmax_j A[t, :],  dA[t, :] = A_F[t, :] - A_N[t, :]
    print("\n j* = argmax_j A[t, :] (head-mean weight) and dA[t, :] = A_B - A_A, per layer")
    for n in range(len(mA)):
        ja, jb = int(np.argmax(mA[n])), int(np.argmax(mB[n]))
        dA = mB[n] - mA[n]
        up = int(np.argmax(dA))
        print(f"   L{n + 1}:  j* A = '{toks_a[ja]}' ({mA[n][ja]:.2f})   j* B = '{toks_b[jb]}' ({mB[n][jb]:.2f})"
              f"   largest dA: '{toks_b[up]}' {dA[up]:+.2f}")
    top_b = [toks_b[j] for j in np.argsort(-nB[-1])[:3]]
    print(f"\n   tokens <call> takes the most from in the last layer (norm), B: {top_b}")


# ====================================================================
# All flips vs non-flipping swaps
# ====================================================================

def summary(model, token_to_id, test):
    flips, n_checked, n_fragile = L.audit(model, token_to_id, test)
    rows = {"flip": [], "keep": []}
    seen = set()

    def measure(text_a, text_b, pos):
        ids_a, _ = encode_request(token_to_id, text_a)
        ids_b, _ = encode_request(token_to_id, text_b)
        k, e = len(ids_a) - 1, pos + 1                     # +1 for <BOS>
        _, mA, nA = row_views(attention(model, ids_a), k)
        _, mB, nB = row_views(attention(model, ids_b), k)
        top = [int(np.argmax(nB[n]) == e) for n in range(model.N)]
        return np.concatenate([mB[:, e] - mA[:, e], nB[:, e] - nA[:, e], nB[:, e], top])

    for x, pos, old, new, pred in flips:
        rows["flip"].append(measure(x["text"], L.edited_text(x, pos, new), pos))
        if (x["text"], pos) in seen:
            continue
        seen.add((x["text"], pos))
        for alt in L.NOUNS:
            if alt in (old, new) or alt not in token_to_id:
                continue
            ids_c, _ = encode_request(token_to_id, L.edited_text(x, pos, alt))
            if L.predict(model, token_to_id, ids_c) == x["tool"]:
                rows["keep"].append(measure(x["text"], L.edited_text(x, pos, alt), pos))
                break

    N = model.N
    F, K = np.mean(rows["flip"], 0), np.mean(rows["keep"], 0)
    print(f"\n{len(rows['flip'])} flipping edits ({n_fragile}/{n_checked} requests), "
          f"{len(rows['keep'])} non-flipping swaps of the same noun position")
    print("\n attention from <call> to the EDITED word (B; change = B - A)")
    print(f"   {'':<28}{'flipping':>10}{'non-flipping':>14}")
    for n in range(N):
        print(f"   L{n + 1} weight change       {F[n]:>10.3f}{K[n]:>14.3f}")
    for n in range(N):
        print(f"   L{n + 1} norm change         {F[N + n]:>10.2f}{K[N + n]:>14.2f}")
    for n in range(N):
        print(f"   L{n + 1} norm (B)            {F[2 * N + n]:>10.2f}{K[2 * N + n]:>14.2f}")
    for n in range(N):
        print(f"   L{n + 1} edited word is top  {F[3 * N + n]:>9.0%}{K[3 * N + n]:>13.0%}")


def main():
    ap = argparse.ArgumentParser(description="Attention analysis at the decision position")
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
        compare(model, token_to_id, DRAFT_A, DRAFT_B)


if __name__ == "__main__":
    main()
