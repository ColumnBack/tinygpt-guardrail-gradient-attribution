"""
select_tool.py
==============

Type a request, see which MCP tool the retrained TinyGPT selects.
요청을 입력하면 재학습된 TinyGPT가 고른 MCP 도구와 확률을 보여준다.

    python select_tool.py
    python select_tool.py --text "could you open the config file"
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

from tinygpt import load_model
from tool_data import TOOL_NAMES
from train_tool_gpt import MODEL_PATH, tool_probs

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def show(model, token_to_id, text, k=3):
    try:
        p = tool_probs(model, token_to_id, text)
    except KeyError as e:
        print(f"  {e}  (vocabulary is the training corpus only)")
        return
    order = np.argsort(-p)
    print(f"  request : {text}")
    print(f"  CALL    : {TOOL_NAMES[order[0]]}")
    for j in order[:k]:
        bar = "█" * int(round(p[j] * 30))
        print(f"    {TOOL_NAMES[j]:<26} {p[j]:.4f} {bar}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="one request (otherwise interactive)")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        sys.exit("tool_model.npz not found. Run: python train_tool_gpt.py")
    model, token_to_id, _, _, epoch = load_model(MODEL_PATH)
    print(f"model: {MODEL_PATH.name} | {epoch} epochs | tools: {', '.join(TOOL_NAMES)}")

    if args.text:
        show(model, token_to_id, args.text.lower(), args.top)
        return
    while True:
        try:
            text = input("\n> ").replace("﻿", "").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if text in ("", "q", "quit", "exit"):
            break
        show(model, token_to_id, text, args.top)


if __name__ == "__main__":
    main()
