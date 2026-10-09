"""
tool_attribution.py
===================

Gradient attribution for the MCP tool selector:
"WHICH input tokens made TinyGPT call this tool?"
MCP 도구 선택기에 대한 Gradient Attribution:
"어떤 입력 토큰이 이 도구를 고르게 만들었나?"

Setup / 설정
------------
    u = <BOS> w_1 .. w_n <CALL>          (k = index of <CALL>)
    X_token = E[u]          (T x d)       token embeddings (input to attribute)
    X^(0)   = X_token + P                 positional part is NOT attributed

Score s(X_token) at the <CALL> row, two choices (--score):
    logp   : s = log softmax(z_k)[tool]                      "why this tool"
    margin : s = z_k[tool] - z_k[rival]                      "why A rather than B"
             (softmax normaliser cancels -> no saturation)

Backward (TinyGPT's own chain, run for a score instead of a loss and
stopped at the input embeddings):
    G_logits[k] = onehot(tool) - softmax(z_k)        (logp)
                = onehot(tool) - onehot(rival)       (margin)
    G_X_final   = G_logits E           (tied head)
    blocks N..1 -> g = dS/dX_token     (T x d)

Per-token attributions / 토큰별 기여도
(X = X_token, X' = baseline, g(t) = dS(t)/dt; same notation as docs/ig_formula.md):
    grad.norm  ||g_i(X)||                         sensitivity only (no sign)
    grad*in    g_i(X) . X_i                       1st-order Taylor vs X_i = 0
    IG         IG_j = (X_j - X'_j) int_0^1 dS(t)/dt_j |_{t = X' + alpha (X - X')} d alpha
               Integrated Gradients, midpoint Riemann sum over alpha;
               per token i: sum of IG_j over its d embedding dims.
               baseline X'_i = 0 for request words (BOS, <CALL> kept)
               completeness:  sum_i IG_i = S(X) - S(X')   (checked below)
    occlusion  S(X) - S(X with X_i = 0)           exact, one token at a time

Usage:
    python tool_attribution.py --text "could you open the config file"
    python tool_attribution.py --text "show the folder" --score margin
    python tool_attribution.py --text "save my changes" --vs filesystem.write_file
    python tool_attribution.py --summary      # prefix/verb/object share on test set
    python tool_attribution.py --check        # finite-difference check of dS/dX
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

from tinygpt import load_model
from tool_data import CALL, PREFIXES, TOOLS, TOOL_NAMES, make_split
from train_tool_gpt import MODEL_PATH

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ====================================================================
# Score and its gradient w.r.t. the token embeddings
# ====================================================================

def score_and_grad(model, X_token, k, tool_id, rival_id=None):
    """Return (s, dS/dX_token).  rival_id=None -> logp, else margin."""
    logits, _, caches = model.forward_embedded(X_token)
    z = logits[k]

    G = np.zeros_like(logits)
    if rival_id is None:
        p = np.exp(z - z.max())
        p /= p.sum()
        s = float(np.log(p[tool_id] + 1e-300))
        G[k] = -p
        G[k, tool_id] += 1.0
    else:
        s = float(z[tool_id] - z[rival_id])
        G[k, tool_id] += 1.0
        G[k, rival_id] -= 1.0

    GX = G @ model.p["E"]
    for n in reversed(range(model.N)):
        GX, _ = model.block_backward(GX, caches[n], n)
    return s, GX


def attribute(model, ids, tool_id, rival_id=None, steps=64):
    ids = np.asarray(ids, dtype=np.int64)
    k = len(ids) - 1                       # <CALL> is the last input token
    X = model.p["E"][ids].copy()

    # baseline X' (= B): zero embedding for request words, keep <BOS> and <CALL>
    B = X.copy()
    B[1:k] = 0.0

    s, g = score_and_grad(model, X, k, tool_id, rival_id)
    s_base, _ = score_and_grad(model, B, k, tool_id, rival_id)

    # Integrated Gradients, midpoint Riemann sum:
    #   IG_j ~ (X_j - X'_j) * mean_k dS(t)/dt_j at t = X' + alpha_k (X - X'),
    #   alpha_k = (k - 1/2) / steps
    acc = np.zeros_like(X)
    for a in (np.arange(steps) + 0.5) / steps:
        _, ga = score_and_grad(model, B + a * (X - B), k, tool_id, rival_id)
        acc += ga
    ig = ((X - B) * (acc / steps)).sum(1)

    occ = np.zeros(len(ids))
    for i in range(1, k):
        Xo = X.copy()
        Xo[i] = 0.0
        occ[i] = s - score_and_grad(model, Xo, k, tool_id, rival_id)[0]

    return dict(
        s=s, s_base=s_base,
        grad_norm=np.linalg.norm(g, axis=1),
        grad_x_in=(g * X).sum(1),
        ig=ig, occ=occ,
    )


# ====================================================================
# Helpers
# ====================================================================

def encode_request(token_to_id, text):
    words = text.lower().split()
    unknown = [w for w in words if w not in token_to_id]
    if unknown:
        raise KeyError(f"unknown words: {unknown}")
    ids = [token_to_id["<BOS>"]] + [token_to_id[w] for w in words] + [token_to_id[CALL]]
    return ids, ["<BOS>"] + words + [CALL]


def tool_distribution(model, ids, token_to_id):
    logits, _, _ = model.forward(ids)
    tids = [token_to_id[t] for t in TOOL_NAMES]
    z = logits[-1, tids]
    p = np.exp(z - z.max())
    return p / p.sum()


def bar(v, scale, width=18):
    n = int(round(abs(v) / scale * width)) if scale > 0 else 0
    return ("+" if v >= 0 else "-") * min(n, width)


# ====================================================================
# One request -> table
# ====================================================================

def explain(model, token_to_id, text, score="logp", vs=None, steps=64):
    ids, toks = encode_request(token_to_id, text)
    p = tool_distribution(model, ids, token_to_id)
    order = np.argsort(-p)
    tool = TOOL_NAMES[order[0]]
    tool_id = token_to_id[tool]

    rival_id = None
    if vs is not None or score == "margin":
        rival = vs if vs is not None else TOOL_NAMES[order[1]]
        rival_id = token_to_id[rival]

    r = attribute(model, ids, tool_id, rival_id, steps)

    print(f"\nrequest  : {text}")
    print("selected : " + "  ".join(f"{TOOL_NAMES[j]}={p[j]:.3f}" for j in order[:3]))
    if rival_id is None:
        print(f"score    : s = log P({tool} | request) = {r['s']:.4f}")
    else:
        print(f"score    : s = z[{tool}] - z[{rival}] = {r['s']:.4f}")

    scale = max(np.abs(r["ig"]).max(), 1e-12)
    print(f"\n  {'token':<14}{'|grad|':>9}{'grad*in':>10}{'IG':>10}{'occlude':>10}   IG bar")
    for i, t in enumerate(toks):
        fixed = i == 0 or i == len(toks) - 1
        if fixed:
            print(f"  {t:<14}{r['grad_norm'][i]:>9.4f}{'(fixed)':>10}{'':>10}{'':>10}")
            continue
        print(f"  {t:<14}{r['grad_norm'][i]:>9.4f}{r['grad_x_in'][i]:>10.4f}"
              f"{r['ig'][i]:>10.4f}{r['occ'][i]:>10.4f}   {bar(r['ig'][i], scale)}")

    total = r["ig"].sum()
    gap = r["s"] - r["s_base"]
    print(f"\n  completeness: sum IG = {total:.4f}   s(x) - s(baseline) = {gap:.4f}"
          f"   (diff {abs(total - gap):.1e}; shrink with --steps)")
    if rival_id is None and r["s"] > -1e-3:
        print("  note: P is saturated (~1) -> raw gradients ~0, grad*in/|grad| look empty;"
              " IG still works (it integrates the path). Try --score margin.")


# ====================================================================
# Test-set summary: share of |IG| on prefix / verb / object words
# ====================================================================

def split_roles(request, tool):
    words = request.split()
    prefix = ""
    for pf in sorted((p for p in PREFIXES if p), key=len, reverse=True):
        if request.startswith(pf + " "):
            prefix = pf
            break
    n_pre = len(prefix.split()) if prefix else 0
    rest = " ".join(words[n_pre:])
    verb = next(v for v in sorted(TOOLS[tool][0], key=len, reverse=True)
                if rest.startswith(v + " "))
    n_verb = len(verb.split())
    return ["prefix"] * n_pre + ["verb"] * n_verb + ["object"] * (len(words) - n_pre - n_verb)


def summary(model, token_to_id, score="margin", steps=32):
    _, test = make_split()
    share = {"prefix": [], "verb": [], "object": []}
    top_word = {}
    for req, tool in test:
        ids, toks = encode_request(token_to_id, req)
        p = tool_distribution(model, ids, token_to_id)
        order = np.argsort(-p)
        rival = token_to_id[TOOL_NAMES[order[1]]] if score == "margin" else None
        r = attribute(model, ids, token_to_id[tool], rival, steps)
        ig = np.abs(r["ig"][1:-1])
        roles = split_roles(req, tool)
        tot = ig.sum() + 1e-12
        for role in share:
            share[role].append(sum(v for v, ro in zip(ig, roles) if ro == role) / tot)
        w = toks[1:-1][int(np.argmax(ig))]
        top_word.setdefault(tool, {}).setdefault(w, 0)
        top_word[tool][w] += 1

    print(f"\nheld-out test set ({len(test)} requests), score={score}, |IG| share per role:")
    for role, v in share.items():
        print(f"  {role:<7} {np.mean(v):.3f}  {'#' * int(np.mean(v) * 40)}")
    print("\nmost-attributed word per tool (count over test requests):")
    for tool in TOOL_NAMES:
        items = sorted(top_word.get(tool, {}).items(), key=lambda x: -x[1])
        print(f"  {tool:<26} " + ", ".join(f"{w}({c})" for w, c in items[:4]))


# ====================================================================
# Finite-difference check of dS/dX_token
# ====================================================================

def check(model, token_to_id, text="could you open the config file", eps=1e-5):
    ids, _ = encode_request(token_to_id, text)
    k = len(ids) - 1
    X = model.p["E"][ids].copy()
    tid = token_to_id["filesystem.read_file"]
    rid = token_to_id["web.fetch"]
    rnd = np.random.default_rng(0)
    for name, rival in (("logp", None), ("margin", rid)):
        _, g = score_and_grad(model, X, k, tid, rival)
        num, ana = [], []
        for _ in range(20):
            i, j = rnd.integers(1, k), rnd.integers(X.shape[1])
            Xp, Xm = X.copy(), X.copy()
            Xp[i, j] += eps
            Xm[i, j] -= eps
            num.append((score_and_grad(model, Xp, k, tid, rival)[0]
                        - score_and_grad(model, Xm, k, tid, rival)[0]) / (2 * eps))
            ana.append(g[i, j])
        num, ana = np.array(num), np.array(ana)
        rel = np.linalg.norm(num - ana) / (np.linalg.norm(num) + np.linalg.norm(ana) + 1e-12)
        print(f"  dS/dX_token [{name:<6}] rel err {rel:.2e}  (|g| ~ {np.abs(ana).mean():.1e})")


def main():
    ap = argparse.ArgumentParser(description="Gradient attribution for tool selection")
    ap.add_argument("--text")
    ap.add_argument("--score", choices=["logp", "margin"], default="margin")
    ap.add_argument("--vs", choices=TOOL_NAMES, help="rival tool for the margin score")
    ap.add_argument("--steps", type=int, default=64, help="IG integration steps")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        sys.exit("tool_model.npz not found. Run: python train_tool_gpt.py")
    model, token_to_id, _, _, _ = load_model(MODEL_PATH)

    if args.check:
        check(model, token_to_id)
    elif args.summary:
        summary(model, token_to_id, steps=min(args.steps, 32))
    elif args.text:
        explain(model, token_to_id, args.text, args.score, args.vs, args.steps)
    else:
        while True:
            try:
                text = input("\n> ").replace("﻿", "").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if text in ("", "q", "quit", "exit"):
                break
            try:
                explain(model, token_to_id, text, args.score, args.vs, args.steps)
            except KeyError as e:
                print(f"  {e}")


if __name__ == "__main__":
    main()
