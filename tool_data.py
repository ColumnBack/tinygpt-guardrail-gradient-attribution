"""
tool_data.py
============

MCP-style tool-selection corpus for TinyGPT.
TinyGPT가 MCP 도구를 고르도록 학습시키는 코퍼스.

Sequence format / 시퀀스 포맷 (word-level, whitespace split):

    <BOS> please read the config file <CALL> filesystem.read_file <EOS>
    `---------- user request -------'  ^      `-- tool token --'

The model reads the request, and at the <CALL> position its next-token
distribution over the tool tokens IS the tool selection.
모델은 요청을 읽고, <CALL> 위치의 다음 토큰 분포가 곧 "도구 선택"이다.

Requests are generated from templates (prefix + verb + object). Some
(verb, object) combinations are held out as a TEST set, so test requests
use only known words but in combinations the model never saw.
템플릿(prefix + 동사 + 목적어)으로 생성하고, 일부 조합은 test로 떼어 둔다.
→ test 문장은 아는 단어로만 되어 있지만 처음 보는 조합이다.
"""

import itertools
import random

CALL = "<call>"   # lowercase: tinygpt.build_dataset lowercases every token

# ----------------------------------------------------------------------
# tool -> (verbs, objects)
#   Verbs/objects deliberately overlap across tools ("show", "open",
#   "file", "save", "get" ...), so the model must use the COMBINATION.
#   That is exactly what attribution should reveal later.
#   동사/목적어를 일부러 도구끼리 겹치게 했다. 모델은 "조합"을 봐야 하고,
#   attribution이 바로 그걸 드러내야 한다.
# ----------------------------------------------------------------------
TOOLS = {
    "filesystem.read_file": (
        ["read", "open", "show", "display", "print"],
        ["the config file", "notes.txt", "the readme file",
         "the log file", "main.py", "the contents of notes.txt"],
    ),
    "filesystem.write_file": (
        ["write", "save", "overwrite", "store", "put"],
        ["this text to notes.txt", "the draft to a file",
         "the report to disk", "the output into main.py",
         "these lines in the config file", "my notes to a new file"],
    ),
    "filesystem.list_directory": (
        ["list", "show", "browse", "enumerate", "display"],
        ["the folder", "the directory", "files in the downloads folder",
         "the contents of the src directory", "what is in the folder",
         "everything in the project directory"],
    ),
    "web.search": (
        ["search", "google", "look up", "find", "research"],
        ["news about ai", "the best laptop this year",
         "python tutorials online", "reviews of the new phone",
         "cheap flights to tokyo", "recipes for dinner"],
    ),
    "web.fetch": (
        ["fetch", "download", "get", "open", "load"],
        ["the url", "this webpage", "the page at the link",
         "the website html", "the article from the link",
         "the content of this url"],
    ),
    "weather.get_forecast": (
        ["check", "get", "show", "tell me", "look up"],
        ["the weather in seoul", "the forecast for tomorrow",
         "if it will rain today", "the temperature in busan",
         "the weather this weekend", "the forecast in tokyo"],
    ),
    "calendar.create_event": (
        ["schedule", "book", "add", "set up", "create"],
        ["a meeting tomorrow", "an event on friday",
         "a call with the team", "an appointment at noon",
         "a reminder for monday", "a meeting with my boss"],
    ),
    "email.send": (
        ["send", "email", "mail", "write", "forward"],
        ["a message to the team", "an email to my boss",
         "the report to alice", "a note to the client",
         "a thank you email to bob", "the invoice to the customer"],
    ),
    "db.query": (
        ["query", "select", "count", "look up", "get"],
        ["the users table", "rows from the orders table",
         "the database for sales", "records in the customers table",
         "the total sales from the database", "all orders in the table"],
    ),
    "git.commit": (
        ["commit", "save", "push", "check in", "record"],
        ["my changes", "the code changes", "the staged files",
         "this fix to the repo", "the new feature to git",
         "my work to the branch"],
    ),
}

PREFIXES = ["", "please", "can you", "could you", "i want to", "help me"]

TOOL_NAMES = list(TOOLS)


def request_text(prefix, verb, obj):
    return " ".join(w for w in (prefix, verb, obj) if w)


def to_sequence(request, tool):
    """The training string (BOS/EOS are added by build_dataset)."""
    return f"{request} {CALL} {tool}"


def make_split(seed=0, test_frac=0.2):
    """Return (train_pairs, test_pairs), each a list of (request, tool).

    Held out by (verb, object) COMBINATION, so every test word still
    appears somewhere in train (no <UNK>).
    (동사, 목적어) 조합 단위로 test를 떼므로 test의 모든 단어는 train에 있다.
    """
    rnd = random.Random(seed)
    train, test = [], []

    for tool, (verbs, objs) in TOOLS.items():
        combos = list(itertools.product(verbs, objs))
        rnd.shuffle(combos)
        n_test = int(len(combos) * test_frac)
        held, kept = combos[:n_test], combos[n_test:]

        # make sure every verb and object of this tool survives in train
        seen_v = {v for v, _ in kept}
        seen_o = {o for _, o in kept}
        moved = [c for c in held if c[0] not in seen_v or c[1] not in seen_o]
        held = [c for c in held if c not in moved]
        kept += moved

        for v, o in kept:
            # each train combo appears with 2 random prefixes
            for prefix in rnd.sample(PREFIXES, 2):
                train.append((request_text(prefix, v, o), tool))
        for v, o in held:
            test.append((request_text(rnd.choice(PREFIXES), v, o), tool))

    rnd.shuffle(train)
    return train, test


if __name__ == "__main__":
    tr, te = make_split()
    print(f"train {len(tr)}  test {len(te)}  tools {len(TOOLS)}")
    for r, t in tr[:8]:
        print(f"  {r:<50} -> {t}")
