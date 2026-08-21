#!/usr/bin/env python3
"""Generate the synthetic datasets for the workshop demos.

Everything here is invented. There is no real customer data, no real names and no
real ticket text — the point of the demos is the technique, not the content.

    python3 make_data.py --out ../data
"""
import argparse, json, random, pathlib

random.seed(7)

SYMPTOMS = [
    ("printer in bay {n} is making a grinding noise again", "facilities", "medium", True),
    ("laptop won't connect to the guest wifi", "it-support", "low", False),
    ("badge reader on the {side} door is dead", "facilities", "high", False),
    ("payroll export failed overnight, third time this week", "applications", "high", True),
    ("monitor flickers when I plug in the dock", "it-support", "low", False),
    ("conveyor {n} stopped mid-shift, no error on the panel", "engineering", "high", False),
    ("shared drive is read-only for the whole team", "it-support", "high", False),
    ("air conditioning in room {n} has been off since monday", "facilities", "medium", True),
    ("vpn drops every twenty minutes or so", "it-support", "medium", True),
    ("cannot book the meeting room from the calendar plugin", "applications", "low", False),
    ("coolant leak near press {n}", "engineering", "high", False),
    ("new starter has no account yet, starts tomorrow", "it-support", "medium", False),
]
PREFIX = ["", "hi — ", "morning, ", "quick one: ", "sorry to bother you but ", "again — "]
SUFFIX = ["", " thanks", " please advise", " this is urgent", " no rush", " second time reporting this"]

SYSTEM = ("You triage support tickets. Reply with JSON only, using exactly the keys "
          "severity, team and repeat. severity is one of low, medium, high.")

def ticket():
    text, team, sev, rep = random.choice(SYMPTOMS)
    text = text.format(n=random.randint(1, 9), side=random.choice(["north", "south", "east"]))
    # a repeat report is only a repeat if the wording says so
    repeat = rep or "again" in text or "third time" in text or "second time" in text
    body = random.choice(PREFIX) + text + random.choice(SUFFIX)
    if "second time" in body:
        repeat = True
    return body, {"severity": sev, "team": team, "repeat": repeat}

def row(body, label):
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": body},
        {"role": "assistant", "content": json.dumps(label, separators=(",", ":"))},
    ]}

TOOLS = """You can call exactly these tools:
  lookup_ticket_history(ticket_id: str)
  check_parts_stock(part: str)
  escalate(team: str, reason: str)
Call one tool at a time as TOOL: name(args). Wait for RESULT before continuing."""

PARTS = ["belt-40mm", "roller-A2", "fuse-15a", "seal-kit-9", "bearing-77"]

def trace():
    """One agent trace. Roughly a third contain a failure and a recovery, on purpose —
    a model that has only seen clean traces never learns to handle a bad result."""
    tid = "T-%04d" % random.randint(1000, 9999)
    part = random.choice(PARTS)
    msgs = [{"role": "system", "content": TOOLS},
            {"role": "user", "content": f"Ticket {tid} is back again. Work out if it is a repeat and "
                                        f"whether we have {part} in stock, then escalate if needed."}]
    if random.random() < 0.35:                      # the failure-and-recovery path
        msgs += [
            {"role": "assistant", "content": f"TOOL: lookup_ticket_history({tid.lower()})"},
            {"role": "user", "content": "RESULT: error — ticket id must be upper case"},
            {"role": "assistant", "content": f"TOOL: lookup_ticket_history({tid})"},
        ]
    else:
        msgs += [{"role": "assistant", "content": f"TOOL: lookup_ticket_history({tid})"}]
    msgs += [
        {"role": "user", "content": "RESULT: 3 prior reports in 30 days"},
        {"role": "assistant", "content": f"TOOL: check_parts_stock({part})"},
        {"role": "user", "content": "RESULT: 0 in stock, 5 day lead time"},
        {"role": "assistant", "content": "TOOL: escalate(engineering, repeat fault with no parts on hand)"},
        {"role": "user", "content": "RESULT: escalated, ref E-8811"},
        {"role": "assistant", "content": f"{tid} is a repeat fault (3 reports in 30 days) and {part} is "
                                         f"out of stock with a 5 day lead time. Escalated to engineering "
                                         f"as E-8811."},
    ]
    return {"messages": msgs}

def write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"  {path.name:24s} {len(rows):>5} rows")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    seen, rows = set(), []
    while len(rows) < 700:
        b, l = ticket()
        if b in seen:
            continue
        seen.add(b); rows.append(row(b, l))
    random.shuffle(rows)
    print("writing datasets:")
    write(out / "triage_train.jsonl", rows[:600])
    write(out / "triage_test.jsonl",  rows[600:700])
    write(out / "agent_traces.jsonl", [trace() for _ in range(400)])
    write(out / "agent_eval.jsonl",   [trace() for _ in range(30)])
