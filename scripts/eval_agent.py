#!/usr/bin/env python3
"""Demo D scoreboard — run the agent for a whole episode and score what it did.

    python3 eval_agent.py --model <HF_ID>                                    # before
    python3 eval_agent.py --model <HF_ID> --adapter ../checkpoints/demo-d    # after
    python3 eval_agent.py --model <HF_ID> --show 1                           # + a transcript

Why an environment loop: a trace in this dataset is 9-11 messages, and the
escalate() call lands on the third or fourth *assistant* turn. Generating a
single completion from [system, user] can therefore only ever produce the first
tool call — escalation is unreachable by construction, and "called a tool at
all" is satisfied trivially by any model that can read the system prompt. The
first version of this script did exactly that and scored base and tuned models
identically, which said nothing about either.

So: simulate the tool results, let the model take turns until it stops or hits
the cap, and score the episode. The environment rules below are copied from the
traces in make_data.py so that what the model was trained against and what it is
tested against agree.
"""
import argparse, json, re, pathlib

TOOLS = {"lookup_ticket_history", "check_parts_stock", "escalate"}
CALL = re.compile(r"TOOL:\s*([a-z_]+)\((.*?)\)")
KWARG = re.compile(r"^\s*[a-z_]+\s*=\s*(.*)$")
UPPER_ERR = "RESULT: error — ticket id must be upper case"


def parse_arg(raw: str):
    """Pull the value out of one tool argument. Returns (value, used_kwargs).

    The traces call tools positionally — `lookup_ticket_history(T-9454)` — but a
    base instruct model reaches for Python-style keywords: `ticket_id="T-9454"`.
    That difference has to be handled explicitly. Matching on the raw argument
    string means `ticket_id="T-6478"` fails an isupper() test because of the
    *keyword*, and the environment then reports a casing error for a correctly
    cased ID — a misdiagnosis the model cannot possibly act on.
    """
    used_kwargs = False
    m = KWARG.match(raw)
    if m:
        raw, used_kwargs = m.group(1), True
    return raw.strip().strip('"').strip("'").strip(), used_kwargs


def env_reply(tool: str, args: str) -> str:
    """The deterministic environment the traces were generated against."""
    if tool not in TOOLS:
        return "RESULT: error — no such tool"
    value, used_kwargs = parse_arg(args.split(",")[0])
    if used_kwargs:
        # A real failure — the harness parses positional calls — but say so
        # accurately. An error the model can act on is the whole point.
        return (f"RESULT: error — pass arguments positionally, as "
                f"{tool}({value or 'T-1234'})")
    if tool == "lookup_ticket_history":
        # the deliberate trap: 155 of the 400 training traces call this with a
        # lowercase id, get told off, and retry. Recovering is the behaviour
        # Demo D is actually about.
        return "RESULT: 3 prior reports in 30 days" if value.isupper() else UPPER_ERR
    if tool == "check_parts_stock":
        return "RESULT: 0 in stock, 5 day lead time"
    return "RESULT: escalated, ref E-8811"


def render(tok, msgs):
    """Generation prompt as a dict, with reasoning off.

    Two version traps. Thinking: Qwen3-class models default to thinking ON, so
    the model burns the token budget in a <think> block and never writes
    `TOOL: ...`. Return type: transformers v5 hands back a BatchEncoding where
    v4 returned a bare tensor, and passing that positionally into generate()
    dies inside the library on `inputs_tensor.shape[0]`.
    """
    kw = dict(add_generation_prompt=True, tokenize=True, return_tensors="pt")
    for attempt in (dict(return_dict=True, enable_thinking=False),
                    dict(return_dict=True),
                    dict(enable_thinking=False),
                    dict()):
        try:
            enc = tok.apply_chat_template(msgs, **kw, **attempt)
            break
        except TypeError:
            continue
    else:
        raise RuntimeError("apply_chat_template rejected every argument combination")
    if hasattr(enc, "keys"):
        return {k: enc[k] for k in ("input_ids", "attention_mask") if k in enc}
    return {"input_ids": enc}


def trim(text: str):
    """Cut the turn at the point the model starts writing the environment.

    A model that has not learned to wait will happily invent `RESULT: ...` and
    play both sides of the conversation. That is a real failure mode and worth
    counting, but it must not be allowed to contaminate the transcript.
    """
    m = re.search(r"\n?\s*RESULT:", text)
    if m:
        return text[:m.start()].strip(), True
    return text.strip(), False


def run_episode(model, tok, msgs, max_new, max_turns):
    msgs = list(msgs)
    calls, transcript, hallucinated = [], [], False
    saw_err = made_lowercase = used_kwargs = recovered = looped = False

    for _ in range(max_turns):
        enc = {k: v.to(model.device) for k, v in render(tok, msgs).items()}
        n_in = enc["input_ids"].shape[-1]
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        text = tok.decode(out[0][n_in:], skip_special_tokens=True)
        text, faked = trim(text)
        hallucinated = hallucinated or faked
        if not text:
            break
        transcript.append(("assistant", text))
        msgs.append({"role": "assistant", "content": text})

        found = CALL.findall(text)
        if not found:
            break                       # a prose answer ends the episode
        tool, args = found[0]
        calls.append((tool, args))
        value, kw = parse_arg(args.split(",")[0])
        used_kwargs = used_kwargs or kw
        if tool == "lookup_ticket_history" and not kw and not value.isupper():
            made_lowercase = True

        reply = env_reply(tool, args)
        is_err = reply.startswith("RESULT: error")
        if saw_err and not is_err:
            recovered = True            # took the correction and moved on
        saw_err = saw_err or is_err
        transcript.append(("env", reply))
        msgs.append({"role": "user", "content": reply})

        # Circuit breaker. Repeating the identical call is a diagnosable failure
        # in its own right, and letting it run to the turn cap on all 30 tasks
        # wastes minutes and makes "avg tool calls" meaningless.
        if len(calls) >= 3 and len(set(calls[-3:])) == 1:
            looped = True
            break

    names = [c[0] for c in calls]
    return {
        "made_a_call":    bool(calls),
        "all_valid":      bool(calls) and all(n in TOOLS for n in names),
        "invented_tool":  any(n not in TOOLS for n in names),
        "hallucinated":   hallucinated,
        "checked_stock":  "check_parts_stock" in names,
        "escalated":      "escalate" in names,
        # a closing summary only counts if the agent actually did the work first —
        # a model that writes prose and never touches a tool has not "finished"
        "finished":       bool(calls) and transcript[-1][0] == "assistant"
                          and not CALL.search(transcript[-1][1]),
        "lowercase":      made_lowercase,
        "used_kwargs":    used_kwargs,
        "recovered":      recovered,
        "looped":         looped,
        "turns":          len(calls),
    }, transcript


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="../data/agent_eval.jsonl")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--show", type=int, default=0, help="print this many transcripts")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map="auto")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    tasks = [json.loads(l) for l in
             pathlib.Path(a.data).read_text().splitlines() if l.strip()]
    keys = ["made_a_call", "all_valid", "invented_tool", "hallucinated",
            "checked_stock", "escalated", "finished", "lowercase",
            "used_kwargs", "recovered", "looped"]
    agg = {k: 0 for k in keys}
    turns = []

    for i, t in enumerate(tasks):
        prompt = [m for m in t["messages"] if m["role"] in ("system", "user")][:2]
        r, transcript = run_episode(model, tok, prompt, a.max_new, a.max_turns)
        for k in keys:
            agg[k] += bool(r[k])
        turns.append(r["turns"])
        if i < a.show:
            print(f"\n----- transcript {i + 1} " + "-" * 40)
            for who, line in transcript:
                print(f"  {'model' if who == 'assistant' else '  env':>6}: {line}")

    n = len(tasks)
    label = "AFTER  (tuned)" if a.adapter else "BEFORE (base)"
    print("\n" + "=" * 56)
    print(f"  {label}   {n} held-out episodes")
    print(f"  called a tool at all      {agg['made_a_call']:>3}/{n}")
    print(f"  every call was valid      {agg['all_valid']:>3}/{n}")
    print(f"  invented a tool name      {agg['invented_tool']:>3}/{n}   <-- lower is better")
    print(f"  faked the RESULT itself   {agg['hallucinated']:>3}/{n}   <-- lower is better")
    print(f"  checked parts stock       {agg['checked_stock']:>3}/{n}")
    print(f"  reached the escalation    {agg['escalated']:>3}/{n}   <-- the task")
    print(f"  wrote a closing summary   {agg['finished']:>3}/{n}")
    print(f"  avg tool calls per task   {sum(turns) / max(1, n):>6.1f}")
    print("  " + "-" * 52)
    print(f"  called with key=value args  {agg['used_kwargs']:>3}/{n}   <-- harness mismatch")
    print(f"  tripped the upper-case rule {agg['lowercase']:>3}/{n}")
    print(f"  got an error and moved on   {agg['recovered']:>3}/{n}")
    print(f"  looped on the same call     {agg['looped']:>3}/{n}   <-- gave up")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()
