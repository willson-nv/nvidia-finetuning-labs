#!/usr/bin/env python3
"""The Demo C grader. No human labels it, no reward model judges it — the rules do.

This is deliberately short enough to read aloud from the screen.
"""
import json, re

TEAMS = {"facilities", "it-support", "applications", "engineering"}
SEVERITIES = {"low", "medium", "high"}

def reward(output: str, expected: dict | None = None) -> float:
    score = 0.0
    m = re.search(r"\{.*\}", output, re.S)
    if not m:
        return -0.5                                   # no JSON at all
    try:
        got = json.loads(m.group(0))
    except json.JSONDecodeError:
        return -0.5                                   # looked like JSON, wasn't
    score += 1.0                                      # it parsed
    if got.get("severity") in SEVERITIES:
        score += 1.0
    if got.get("team") in TEAMS:
        score += 1.0
    if output.strip() != m.group(0).strip():
        score -= 0.5                                  # prose outside the JSON
    if expected and got.get("team") == expected.get("team"):
        score += 1.0
    return score

if __name__ == "__main__":
    for probe in ['{"severity":"high","team":"engineering","repeat":true}',
                  'Sure! Here is the JSON: {"severity":"high","team":"engineering","repeat":true}',
                  'I think this is a facilities issue.']:
        print(f"{reward(probe):>5.1f}   {probe[:64]}")
