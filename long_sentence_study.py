"""
long_sentence_study.py
======================

Debugging a long-input failure with gradient attribution.
긴 입력에서 단어 하나를 바꿨더니 틀린다 -> 원인을 gradient attribution으로 찾는다.

Data (made to look like real request logs) / 데이터 (실제 로그처럼):

    <context> so <request>          "my week is very busy so could you read the log file"
    <request> because <context>     "could you read the log file because the app crashed after the update"

  - 30 context clauses shared by ALL tools. A clause co-occurs with its
    topical tool ~70% of the time, otherwise with a random tool (a weak,
    natural correlation, like real logs).
    상황 설명 30개를 모든 도구가 공유. 주제가 맞는 도구와 ~70%만 같이 나온다.
  - Clauses contain other tools' words ("meeting", "link", "file", "weather",
    "numbers" ...), as real context does.
  - The request sits at the start OR the end, so its position varies.

Debugging workflow / 디버깅 절차:
  1. accuracy on held-out long requests
  2. single-word edit audit: in every correctly classified test request,
     replace each NOUN of the context by every other context noun -> which edits
     flip the tool?  (the request itself is untouched, so the gold tool
     is still the same)
  3. attribution on a flipped example:  S = Z[<call>, pred]
     (the logit of the wrong tool it picked) -> which tokens drive it
  4. verify the hypothesis: zero out the suspected token -> back to gold?
  5. root cause in the DATA: which tools did that word co-occur with in train
  6. --augment: add copies of each training request with a random context
     (counterfactual augmentation), retrain, re-run the audit

Usage:
    python long_sentence_study.py --retrain           # baseline (~1 min)
    python long_sentence_study.py --augment --retrain # after the fix
    python long_sentence_study.py --text "the team wants to talk so please read the log file"
"""

import argparse
import itertools
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from tinygpt import TinyGPT, Adam, build_dataset, save_state, load_state
from tool_data import CALL, PREFIXES, TOOLS, TOOL_NAMES, request_text, to_sequence
from train_tool_gpt import masked_loss_and_backward, tool_mask, accuracy
from tool_attribution import encode_request, explain, score_and_grad

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
MODEL_PATHS = {False: HERE / "long_model.npz", True: HERE / "long_model_aug.npz"}

# clause -> topical tool
CONTEXTS = {
    "the app crashed after the update": "filesystem.read_file",
    "i need to check a setting": "filesystem.read_file",
    "the server shows an error": "filesystem.read_file",
    "i finished the draft for the meeting": "filesystem.write_file",
    "the notes from the call are ready": "filesystem.write_file",
    "i wrote a summary of the report": "filesystem.write_file",
    "i cannot find the file my friend sent": "filesystem.list_directory",
    "the project folder got messy": "filesystem.list_directory",
    "i downloaded a lot of pictures": "filesystem.list_directory",
    "i am planning a trip to tokyo": "web.search",
    "my laptop is getting slow": "web.search",
    "i am learning about databases": "web.search",
    "my boss sent me a link": "web.fetch",
    "there is an article about the weather": "web.fetch",
    "i saw a page about git": "web.fetch",
    "i am going outside later": "weather.get_forecast",
    "we have a picnic this weekend": "weather.get_forecast",
    "i am driving to busan tomorrow": "weather.get_forecast",
    "my week is very busy": "calendar.create_event",
    "the team wants to talk": "calendar.create_event",
    "i keep forgetting the meeting": "calendar.create_event",
    "my boss asked for an update": "email.send",
    "the client is waiting for news": "email.send",
    "alice needs the numbers today": "email.send",
    "the manager wants the numbers": "db.query",
    "sales looked low last month": "db.query",
    "we need a count of new users": "db.query",
    "i fixed the bug in the file": "git.commit",
    "the feature is done": "git.commit",
    "the code finally works": "git.commit",
}
ALL_CTX = list(CONTEXTS)
CTX_BY_TOOL = {t: [c for c, ct in CONTEXTS.items() if ct == t] for t in TOOL_NAMES}
# nouns of the context clauses: an edit swaps a noun for another noun, so the
# edited sentence still reads naturally ("... the draft for the meeting"
# -> "... the draft for the trip")
NOUNS = sorted({"app", "update", "setting", "server", "error", "draft", "meeting",
                "notes", "call", "summary", "report", "file", "friend", "folder",
                "pictures", "trip", "tokyo", "laptop", "databases", "boss", "link",
                "article", "weather", "page", "git", "picnic", "weekend", "busan",
                "week", "team", "client", "news", "numbers", "manager", "sales",
                "month", "users", "bug", "feature", "code"})


def compose(ctx, req, ctx_first):
    """Return (text, ctx_start, ctx_len) with ctx_start as a WORD index."""
    n = len(ctx.split())
    if ctx_first:
        return f"{ctx} so {req}", 0, n
    return f"{req} because {ctx}", len(req.split()) + 1, n


def sample_ctx(tool, rnd, p_topical=0.7):
    return rnd.choice(CTX_BY_TOOL[tool]) if rnd.random() < p_topical else rnd.choice(ALL_CTX)


def make_long_split(augment=False, seed=0, test_frac=0.2):
    """Items are dicts: text, tool, start, n (context span in words)."""
    rnd = random.Random(seed)
    train, test = [], []
    for tool, (verbs, objs) in TOOLS.items():
        combos = list(itertools.product(verbs, objs))
        rnd.shuffle(combos)
        n_test = int(len(combos) * test_frac)
        held, kept = combos[:n_test], combos[n_test:]
        seen_v, seen_o = {v for v, _ in kept}, {o for _, o in kept}
        moved = [c for c in held if c[0] not in seen_v or c[1] not in seen_o]
        held = [c for c in held if c not in moved]
        kept += moved

        for v, o in kept:
            for _ in range(2):
                req = request_text(rnd.choice(PREFIXES), v, o)
                text, s, n = compose(sample_ctx(tool, rnd), req, rnd.random() < 0.5)
                train.append(dict(text=text, tool=tool, start=s, n=n))
                if augment:   # same request, random context
                    text, s, n = compose(rnd.choice(ALL_CTX), req, rnd.random() < 0.5)
                    train.append(dict(text=text, tool=tool, start=s, n=n))
        for v, o in held:
            req = request_text(rnd.choice(PREFIXES), v, o)
            text, s, n = compose(sample_ctx(tool, rnd), req, rnd.random() < 0.5)
            test.append(dict(text=text, tool=tool, start=s, n=n))
    rnd.shuffle(train)
    return train, test


# ====================================================================
# Train / load
# ====================================================================

def get_model(train, epochs, retrain, path):
    sentences = [to_sequence(x["text"], x["tool"]) for x in train]
    token_to_id, _, data = build_dataset(sentences)
    call_id = token_to_id[CALL]
    model = TinyGPT(vocab_size=len(token_to_id),
                    max_context=max(len(d) - 1 for d in data),
                    d_model=64, n_heads=4, d_ff=128, n_layers=2, seed=42)
    opt = Adam(model.p, lr=2e-3)
    done = 0 if retrain else (load_state(path, model, opt, sentences) or 0)
    if done:
        print(f"loaded {path.name} ({done} epochs)")
    rng = np.random.default_rng(0)
    for epoch in range(done + 1, epochs + 1):
        total = 0.0
        for i in rng.permutation(len(data)):
            ids = data[i]
            loss, grads = masked_loss_and_backward(model, ids[:-1], ids[1:],
                                                   tool_mask(ids, call_id))
            opt.step(model.p, grads, clip_norm=1.0)
            total += loss
        save_state(path, model, opt, epoch, sentences)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"epoch {epoch:3d} | loss {total/len(data):.4f}")
    return model, token_to_id


def predict(model, token_to_id, ids):
    logits, _, _ = model.forward(ids)
    tids = [token_to_id[t] for t in TOOL_NAMES]
    return TOOL_NAMES[int(np.argmax(logits[-1, tids]))]


# ====================================================================
# 2. single-word edit audit
# ====================================================================

def audit(model, token_to_id, test):
    """Return list of flips: (item, word_pos, old, new, pred)."""
    flips, n_checked, n_fragile = [], 0, 0
    max_len = model.max_context
    for x in test:
        ids, _ = encode_request(token_to_id, x["text"])
        if predict(model, token_to_id, ids) != x["tool"]:
            continue
        n_checked += 1
        words = x["text"].split()
        fragile = False
        for pos in range(x["start"], x["start"] + x["n"]):
            if words[pos] not in NOUNS:
                continue
            for new in NOUNS:
                if new == words[pos] or new not in token_to_id:
                    continue
                cand = list(ids)
                cand[pos + 1] = token_to_id[new]         # +1 for <BOS>
                if len(cand) > max_len:
                    continue
                pred = predict(model, token_to_id, cand)
                if pred != x["tool"]:
                    flips.append((x, pos, words[pos], new, pred))
                    fragile = True
        n_fragile += fragile
    return flips, n_checked, n_fragile


def edited_text(x, pos, new):
    w = x["text"].split()
    w[pos] = new
    return " ".join(w)


# ====================================================================
# 4. verify: zero out one token -> back to gold?
# ====================================================================

def occlude_predict(model, token_to_id, text, pos):
    ids, _ = encode_request(token_to_id, text)
    X = model.p["E"][ids].copy()
    X[pos + 1] = 0.0
    logits, _, _ = model.forward_embedded(X)
    tids = [token_to_id[t] for t in TOOL_NAMES]
    return TOOL_NAMES[int(np.argmax(logits[-1, tids]))]


# ====================================================================
# Report
# ====================================================================

def main():
    ap = argparse.ArgumentParser(description="Long-request edit debugging")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--augment", action="store_true",
                    help="counterfactual augmentation (random-context copies)")
    ap.add_argument("--text")
    args = ap.parse_args()

    train, test = make_long_split(args.augment)
    path = MODEL_PATHS[args.augment]
    print(f"model: {path.name} | train {len(train)} | test {len(test)}"
          f"{' | augmented' if args.augment else ''}")
    model, token_to_id = get_model(train, args.epochs, args.retrain, path)

    if args.text:
        explain(model, token_to_id, args.text.lower(), steps=256)
        return

    # 1 ---------------------------------------------------------------
    acc, _ = accuracy(model, token_to_id, [(x["text"], x["tool"]) for x in test])
    print("\n" + "=" * 76)
    print(f" 1. held-out long requests: accuracy {acc:.3f}")

    # 2 ---------------------------------------------------------------
    flips, n_checked, n_fragile = audit(model, token_to_id, test)
    print(f" 2. one-noun edits in the CONTEXT only (request untouched):")
    print(f"    {n_fragile}/{n_checked} correct requests can be flipped by ONE context noun"
          f"  ({len(flips)} flipping edits)")
    print("=" * 76)
    if not flips:
        return

    by_word = Counter((new, pred) for _, _, _, new, pred in flips)
    print("\n    most frequent flipping words -> tool they push to:")
    for (w, t), c in by_word.most_common(8):
        print(f"      {w:<12} -> {t:<26} {c:>4} edits")

    # 3 ---------------------------------------------------------------
    word, wrong_tool = by_word.most_common(1)[0][0]
    x, pos, old, new, pred = next(f for f in flips if f[3] == word and f[4] == wrong_tool)
    after = edited_text(x, pos, new)
    print("\n" + "-" * 76)
    print(f" 3. attribution on one flip  ('{old}' -> '{new}')")
    print("-" * 76)
    print(" before:")
    explain(model, token_to_id, x["text"], steps=256)
    print("\n after  (S = Z[<call>, wrong tool]: which tokens drive the wrong choice):")
    explain(model, token_to_id, after, steps=256)

    # 4 ---------------------------------------------------------------
    back = occlude_predict(model, token_to_id, after, pos)
    print("\n" + "-" * 76)
    print(f" 4. verify: zero out '{new}' in the edited request -> {back}"
          f"  ({'back to gold: hypothesis confirmed' if back == x['tool'] else 'NOT back to gold'})")

    # 5 ---------------------------------------------------------------
    co = Counter(t["tool"] for t in train if new in t["text"].split())
    print(f" 5. root cause in the training data: '{new}' appeared with")
    for t, c in co.most_common():
        print(f"      {t:<26} {c:>4}  {'#' * c}")
    print("    (gold tool for the edited request is "
          f"{x['tool']}; the word was learned as evidence for {pred})")


if __name__ == "__main__":
    main()
