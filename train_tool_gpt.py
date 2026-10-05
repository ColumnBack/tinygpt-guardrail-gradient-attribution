"""
train_tool_gpt.py
=================

Train TinyGPT (tinygpt.py) as an MCP tool selector.
TinyGPT(tinygpt.py)를 MCP 도구 선택기로 학습한다.

    <BOS> could you open the config file <CALL> filesystem.read_file <EOS>

Loss / 손실
-----------
Ordinary LM training puts cross-entropy on EVERY next token. Here we only
care about "which tool after <CALL>", so the loss is MASKED to the tool
token (and the <EOS> after it):

    L = - sum_i w_i log P[i, y_i],   w_i = M_i / sum M,
    M_i = 1  if position i predicts the tool token or <EOS>, else 0

    G_logits = w[:,None] * (P - Y)        (same as loss_and_backward,
                                           but per-row weights instead of 1/T)

도구 토큰 위치에만 loss를 거는 masked cross-entropy.

Output: tool_model.npz

Usage:
    python train_tool_gpt.py              # train (resumes if saved)
    python train_tool_gpt.py --retrain
    python train_tool_gpt.py --check      # gradcheck masked loss only
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

from tinygpt import TinyGPT, Adam, build_dataset, save_state, load_state
from tool_data import CALL, TOOL_NAMES, make_split, to_sequence

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL_PATH = HERE / "tool_model.npz"


# ====================================================================
# Masked loss + full backward
#   (mirrors TinyGPT.loss_and_backward; only the 1/T becomes w_i)
# ====================================================================

def masked_loss_and_backward(model, input_ids, target_ids, mask):

    input_ids = np.asarray(input_ids, dtype=np.int64)
    target_ids = np.asarray(target_ids, dtype=np.int64)
    T = len(input_ids)

    w = mask / mask.sum()

    logits, X_final, caches = model.forward(input_ids)
    probs = model.softmax_rows(logits)

    loss = -np.sum(w * np.log(probs[np.arange(T), target_ids] + 1e-12))

    # G_logits = w (.) (P - Y)
    Glogits = probs.copy()
    Glogits[np.arange(T), target_ids] -= 1.0
    Glogits *= w[:, None]

    # tied LM head: logits = X E^T  -> two routes into E
    GX = Glogits @ model.p["E"]
    GE_from_lm = Glogits.T @ X_final

    grads = {}
    for n in reversed(range(model.N)):
        GX, block_grads = model.block_backward(GX, caches[n], n)
        grads.update(block_grads)

    GE = np.zeros_like(model.p["E"])
    np.add.at(GE, input_ids, GX)
    grads["E"] = GE + GE_from_lm

    GP = np.zeros_like(model.p["P"])
    GP[:T] = GX
    grads["P"] = GP

    return loss, grads


def tool_mask(ids, call_id):
    """Positions (in input_ids = ids[:-1]) that predict the tool and <EOS>."""
    k = ids.index(call_id)            # input position of <CALL>
    mask = np.zeros(len(ids) - 1)
    mask[k] = 1.0                     # <CALL> -> tool
    mask[k + 1] = 1.0                 # tool   -> <EOS>
    return mask


# ====================================================================
# Evaluation: argmax over the TOOL tokens only at the <CALL> position
# ====================================================================

def tool_probs(model, token_to_id, request):
    """P(tool | request), renormalised over the 10 tool tokens."""
    words = request.split()
    unknown = [w for w in words if w not in token_to_id]
    if unknown:
        raise KeyError(f"unknown words: {unknown}")
    ids = [token_to_id["<BOS>"]] + [token_to_id[w] for w in words] + [token_to_id[CALL]]
    logits, _, _ = model.forward(ids)
    tool_ids = [token_to_id[t] for t in TOOL_NAMES]
    z = logits[-1, tool_ids]
    p = np.exp(z - z.max())
    return p / p.sum()


def accuracy(model, token_to_id, pairs):
    hit = 0
    wrong = []
    for req, tool in pairs:
        pred = TOOL_NAMES[int(np.argmax(tool_probs(model, token_to_id, req)))]
        if pred == tool:
            hit += 1
        else:
            wrong.append((req, tool, pred))
    return hit / len(pairs), wrong


# ====================================================================
# Gradcheck of the masked loss (finite differences)
# ====================================================================

def gradcheck(model, ids, call_id, eps=1e-5, n=8, seed=0):
    rnd = np.random.default_rng(seed)
    x, y, m = ids[:-1], ids[1:], tool_mask(ids, call_id)
    _, ana = masked_loss_and_backward(model, x, y, m)
    worst = 0.0
    for name in ["E", "P", "WQ_0_0", "WO_1", "W1_0", "gamma2_1"]:
        if name not in model.p:
            continue
        flat = model.p[name].reshape(-1)
        # sample where the analytic gradient is non-negligible
        a_flat = ana[name].reshape(-1)
        cand = np.argsort(-np.abs(a_flat))[: max(4 * n, n)]
        idx = rnd.choice(cand, size=min(n, len(cand)), replace=False)
        num = np.zeros(len(idx))
        for j, i in enumerate(idx):
            old = flat[i]
            flat[i] = old + eps
            lp, _ = masked_loss_and_backward(model, x, y, m)
            flat[i] = old - eps
            lm, _ = masked_loss_and_backward(model, x, y, m)
            flat[i] = old
            num[j] = (lp - lm) / (2 * eps)
        a = a_flat[idx]
        # tensor-wise relative error (elementwise blows up on ~1e-7 entries)
        rel = np.linalg.norm(num - a) / (np.linalg.norm(num) + np.linalg.norm(a) + 1e-12)
        worst = max(worst, rel)
        print(f"  {name:<10} rel err {rel:.2e}")
    return worst


# ====================================================================
# Main
# ====================================================================

def main():
    ap = argparse.ArgumentParser(description="Train TinyGPT as an MCP tool selector")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--check", action="store_true", help="gradcheck the masked loss and exit")
    args = ap.parse_args()

    train_pairs, test_pairs = make_split()

    # vocab comes from train only; every test word also appears in train
    sentences = [to_sequence(r, t) for r, t in train_pairs]
    token_to_id, id_to_token, data = build_dataset(sentences)
    call_id = token_to_id[CALL]
    max_context = max(len(s) - 1 for s in data)

    model = TinyGPT(vocab_size=len(token_to_id), max_context=max_context,
                    d_model=64, n_heads=4, d_ff=128, n_layers=2, seed=42)
    n_params = sum(v.size for v in model.p.values())
    print(f"train {len(train_pairs)} | test {len(test_pairs)} | tools {len(TOOL_NAMES)} "
          f"| vocab {len(token_to_id)} | context {max_context} | params {n_params}")

    if args.check:
        print("gradcheck (masked loss):")
        w = gradcheck(model, data[0], call_id)
        print(f"  => worst relative error {w:.2e}  ({'OK' if w < 1e-5 else 'CHECK'})")
        return

    opt = Adam(model.p, lr=args.lr)
    done = 0
    if not args.retrain:
        loaded = load_state(MODEL_PATH, model, opt, sentences)
        if loaded is not None:
            done = loaded
            print(f"resumed {MODEL_PATH.name} at epoch {done}")

    acc0, _ = accuracy(model, token_to_id, test_pairs)
    print(f"before training: test acc {acc0:.3f}  (chance {1/len(TOOL_NAMES):.3f})")

    rng = np.random.default_rng(0)
    for epoch in range(done + 1, args.epochs + 1):
        total = 0.0
        for i in rng.permutation(len(data)):
            ids = data[i]
            loss, grads = masked_loss_and_backward(model, ids[:-1], ids[1:],
                                                   tool_mask(ids, call_id))
            opt.step(model.p, grads, clip_norm=1.0)
            total += loss
        save_state(MODEL_PATH, model, opt, epoch, sentences)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            tr_acc, _ = accuracy(model, token_to_id, train_pairs)
            te_acc, _ = accuracy(model, token_to_id, test_pairs)
            print(f"epoch {epoch:3d} | loss {total/len(data):.4f} "
                  f"| train acc {tr_acc:.3f} | test acc {te_acc:.3f}")

    te_acc, wrong = accuracy(model, token_to_id, test_pairs)
    print(f"\nfinal held-out test accuracy: {te_acc:.3f} ({len(test_pairs)-len(wrong)}/{len(test_pairs)})")
    for req, tool, pred in wrong:
        print(f"  MISS  {req:<48} gold={tool:<26} pred={pred}")
    print(f"\nsaved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
