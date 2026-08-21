#!/usr/bin/env python3
"""Demo D scoreboard — did the agent finish the task, and were its tool calls valid?

    python3 eval_agent.py --model <HF_ID>                      # before
    python3 eval_agent.py --model <HF_ID> --adapter ../checkpoints/demo-d   # after
"""
import argparse, json, re, pathlib

TOOLS = {"lookup_ticket_history", "check_parts_stock", "escalate"}
CALL = re.compile(r"TOOL:\s*([a-z_]+)\((.*?)\)")

def score(transcript: str) -> dict:
    calls = CALL.findall(transcript)
    valid = [c for c in calls if c[0] in TOOLS]
    return {
        "made_a_call":   bool(calls),
        "all_valid":     bool(calls) and len(valid) == len(calls),
        "invented_tool": any(c[0] not in TOOLS for c in calls),
        "escalated":     any(c[0] == "escalate" for c in valid),
        "n_calls":       len(calls),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="../data/agent_eval.jsonl")
    ap.add_argument("--max-new", type=int, default=256)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    tasks = [json.loads(l) for l in pathlib.Path(a.data).read_text().splitlines() if l.strip()]
    agg = {"made_a_call": 0, "all_valid": 0, "invented_tool": 0, "escalated": 0}
    for t in tasks:
        prompt = [m for m in t["messages"] if m["role"] in ("system", "user")][:2]
        ids = tok.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to(model.device)
        out = model.generate(ids, max_new_tokens=a.max_new, do_sample=False)
        text = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
        for k, v in score(text).items():
            if k in agg and v:
                agg[k] += 1

    n = len(tasks)
    label = "AFTER  (tuned)" if a.adapter else "BEFORE (base)"
    print("\n" + "=" * 52)
    print(f"  {label}   {n} held-out tasks")
    print(f"  called a tool at all      {agg['made_a_call']:>3}/{n}")
    print(f"  every call was valid      {agg['all_valid']:>3}/{n}")
    print(f"  invented a tool name      {agg['invented_tool']:>3}/{n}   <-- lower is better")
    print(f"  reached the escalation    {agg['escalated']:>3}/{n}   <-- the task")
    print("=" * 52 + "\n")

if __name__ == "__main__":
    main()
