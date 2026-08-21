# RUN.md — every command, and what you should see

Command-by-command for the four demos. Each step shows the command, the output to expect,
and what it means if you get something else.

**Verified vs expected:** steps marked ✅ were actually executed and the output below is
real. Steps marked ⚠️ are written against current library APIs but have **not** been run
on a GPU — treat those outputs as the expected shape, and run them yourself before the day.

---

## Part 0 — Before the workshop

### 0.1 Check the GPU ⚠️

```bash
nvidia-smi
```

**Expect:** one row per GPU with total memory. Note the number — you will quote it in Demo B.

**If it fails:** nothing else in this file will work. Fix drivers first.

---

### 0.2 Install ⚠️

```bash
pip install torch transformers peft trl datasets accelerate bitsandbytes
```

**Expect:** a long install, ending with `Successfully installed ...`.

**Then confirm the pieces are actually importable:**

```bash
python3 -c "import torch, transformers, peft, trl; \
print('torch', torch.__version__, '| cuda', torch.cuda.is_available()); \
print('transformers', transformers.__version__, '| peft', peft.__version__, '| trl', trl.__version__)"
```

**Expect:** `cuda True`. If it says `False`, the GPU scripts will fall back to CPU and
take hours instead of minutes.

---

### 0.3 Pick and pre-download the base model ⚠️

**On Brev, steps 0.1–0.5 are all done for you by `setup.sh` — see BREV.md. Skip to Part 1.**

Pinned model: **`Qwen/Qwen3-8B`** — Apache-2.0, ungated (no HF token to forget), standard
`q_proj/k_proj/v_proj/o_proj` attention so the LoRA target list in `train_lora.py` is correct.

```bash
export BASE=Qwen/Qwen3-8B
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os; m = os.environ['BASE']
AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m)
print('cached OK:', m)"
```

**Expect:** download progress bars, then `cached OK: ...`.

**Why bother:** so nothing downloads from the podium. A 16 GB pull on conference wifi is
how demos die.

**Needs `transformers >= 4.51`.** Below that the qwen3 architecture is unknown and you get
a bare `KeyError: 'qwen3'`.

> **⚠️ Qwen3 thinks by default.** It is a hybrid reasoning model, and
> `apply_chat_template` defaults to `enable_thinking=True` — so the model opens every reply
> with a `<think>` block. Reading its own chat template, the training render ends with
> `<|im_start|>assistant\n<think>\n\n</think>\n\n{JSON}` and the `enable_thinking=False`
> generation prompt ends the same way, so **pass `enable_thinking=False` at inference and
> training and generation line up exactly.** Every snippet below already does.
> `eval_agent.py` does it too — without it the model burns all 256 tokens reasoning, never
> writes `TOOL: ...`, and the scoreboard reads zero for a reason that has nothing to do with
> fine-tuning. `setup.sh` prints all three renderings so you can confirm this yourself.
>
> If you swap in a non-reasoning model, the flag is harmless — templates that do not use it
> ignore it.

---

### 0.4 Generate the data ✅

```bash
cd nvidia-finetuning-labs/scripts
python3 make_data.py --out ../data
```

**Real output:**

```
writing datasets:
  triage_train.jsonl         600 rows
  triage_test.jsonl          100 rows
  agent_traces.jsonl         400 rows
  agent_eval.jsonl            30 rows
```

**Already done** — the files are in `data/`. Re-run only if you want different data;
the seed is fixed so you will get the same rows back.

---

### 0.5 Sanity-check the grader ✅

```bash
python3 reward.py
```

**Real output:**

```
  3.0   {"severity":"high","team":"engineering","repeat":true}
  2.5   Sure! Here is the JSON: {"severity":"high","team":"engineering",
 -0.5   I think this is a facilities issue.
```

**What it means:** clean JSON scores 3.0. The same JSON with a chatty preamble loses half
a point. Prose with no JSON at all is punished. That is the entire reward design, and it is
worth showing on screen in Demo C.

---

## Part 1 — Demo A · make it obey

### 1.1 Show the failure first ⚠️

```bash
python3 -c "
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
m = os.environ['BASE']
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16, device_map='auto')
msgs = [{'role':'system','content':'You triage support tickets. Reply with JSON only, using exactly the keys severity, team and repeat. severity is one of low, medium, high.'},
        {'role':'user','content':'printer in bay 3 is making a grinding noise again'}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt', enable_thinking=False).to(model.device)
out = model.generate(ids, max_new_tokens=120, do_sample=False)
print(tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))"
```

**Expect:** a helpful paragraph, or JSON wrapped in explanation, or invented extra keys.
Occasionally a well-trained instruct model gets this right — if so, ask it for a harder
ticket, or point out that consistency across 600 tickets is the real problem.

**On screen:** this is the "before". Do not skip it — the whole demo is a comparison.

---

### 1.2 Show the data ⚠️

```bash
head -n 2 ../data/triage_train.jsonl | python3 -m json.tool --json-lines
```

**Expect:** two records, each with a system / user / assistant turn.

**Say:** the model only learns from the assistant turn — the rest is context.

---

### 1.3 Start the training run ⚠️

```bash
python3 train_lora.py --model $BASE --out ../checkpoints/demo-a 2>&1 | tee ../results/demo-a.log
```

**Expect immediately:**

```
=== LoRA (bf16 base) · rank 16 · 2.0 epochs ===
```

then a progress bar with a falling `loss`. Loss should drop noticeably in the first
~20 steps. **Leave it running and go back to the slides** — this is your window for the
LoRA and QLoRA slides.

**Expect at the end:**

```
====================================================
  LoRA (bf16 base)
  peak GPU memory       xx.x GB   <-- the Demo B number
  wall clock            xx.x min
  adapter written   ../checkpoints/demo-a
====================================================
```

**If loss is flat:** the chat template probably is not matching. Check `assistant_only_loss`
is supported by your TRL version — if not, remove it and re-run.

**If you hit OOM:** drop `--bs 2` or `--seq 512`.

---

### 1.4 The moment — adapter size ⚠️

```bash
du -sh ../checkpoints/demo-a
du -sh "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--*/snapshots/*/ | tail -1
```

**Expect:** roughly **100 MB** against roughly **16 GB**.

**Write on the slide:** adapter size, base model size, training time.

---

### 1.5 The after ⚠️

Re-run the command from 1.1 with the adapter loaded:

```bash
python3 -c "
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
m = os.environ['BASE']
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16, device_map='auto')
model = PeftModel.from_pretrained(model, '../checkpoints/demo-a')
msgs = [{'role':'system','content':'You triage support tickets. Reply with JSON only, using exactly the keys severity, team and repeat. severity is one of low, medium, high.'},
        {'role':'user','content':'printer in bay 3 is making a grinding noise again'}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt', enable_thinking=False).to(model.device)
out = model.generate(ids, max_new_tokens=120, do_sample=False)
print(tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))"
```

**Expect:** `{"severity":"medium","team":"facilities","repeat":true}` and nothing else.

**If it still waffles:** train a third epoch, or check the adapter path actually loaded —
PEFT fails quietly if the directory is wrong.

---

## Part 2 — Demo B · make it fit

### 2.1 Open a memory monitor in a second pane ⚠️

```bash
nvidia-smi dmon -s um
```

**Expect:** a row per second, `mem` column climbing during training. Leave this visible —
watching the number move is half the lesson.

---

### 2.2 Run the identical script with one flag ⚠️

```bash
python3 train_lora.py --model $BASE --out ../checkpoints/demo-b --qlora 2>&1 | tee ../results/demo-b.log
```

**Expect the header to change and nothing else about the command:**

```
=== QLoRA (4-bit base) · rank 16 · 2.0 epochs ===
```

**Expect at the end:** peak memory **substantially lower** than Demo A, and wall clock
**slightly higher**.

**Say the slower part out loud before anyone notices it.** That trade is the honest content
of this demo.

---

### 2.3 Print the comparison ⚠️

```bash
python3 compare.py
```

**Expect:**

```
                        peak GPU   minutes
LoRA (bf16 base)           xx.x GB      xx.x
QLoRA (4-bit base)         xx.x GB      xx.x

  QLoRA used xx% less memory, and took +x.x min longer.
```

**Fill the three boxes on the slide from this.**

---

### 2.4 Same question, both models ⚠️

Run 1.5 twice, pointing `--adapter` at `demo-a` then `demo-b`.

**Expect:** for a task this shaped, output that is hard to tell apart. That is the point —
you paid memory, not quality.

---

## Part 3 — Demo C · make it better *(run this the week before)*

### 3.1 Show the grader ✅

```bash
cat reward.py
```

**On screen:** twelve lines. Read them aloud. No human labelled anything, no reward model
was trained.

---

### 3.2 Run the RL job ⚠️ — **not live**

Use a small model (1–1.5B) so the loop is fast, starting from the Demo A behaviour.
Whichever library you use, save two things into `../results/`:

- `reward_curve.png` — reward against training step
- `samples_before_after.md` — five outputs from step 0 and five from the end

**Expect:** a curve that rises quickly then flattens. Flat-from-the-start means the reward
is not discriminating; wildly spiky means the learning rate is too high.

**In the room, show the saved artefacts only.** A live RL run can plateau for twenty
minutes and teach nothing.

---

## Part 4 — Demo D · make it act

### 4.1 Train the agent adapter ⚠️ *(before the workshop)*

```bash
python3 train_lora.py --model $BASE --data ../data/agent_traces.jsonl \
  --out ../checkpoints/demo-d --epochs 3
```

**Expect:** the same training output as Demo A. Three epochs because traces are longer and
sparser than the triage pairs.

---

### 4.2 Score the base model ⚠️

```bash
python3 eval_agent.py --model $BASE 2>&1 | tee ../results/agent-before.log
```

**Expect something like:**

```
====================================================
  BEFORE (base)   30 held-out tasks
  called a tool at all       xx/30
  every call was valid       xx/30
  invented a tool name       xx/30   <-- lower is better
  reached the escalation     xx/30   <-- the task
====================================================
```

**Expect the "invented a tool name" number to be non-trivial.** That is the failure the
demo is built around.

---

### 4.3 Score the tuned model ⚠️

```bash
python3 eval_agent.py --model $BASE --adapter ../checkpoints/demo-d 2>&1 | tee ../results/agent-after.log
```

**Expect:** `every call was valid` and `reached the escalation` clearly higher, `invented a
tool name` near zero.

**Show both logs side by side.** That is the scoreboard, and the close of the workshop.

---

### 4.4 Show one training trace ⚠️

```bash
python3 -c "
import json
r = json.loads(open('../data/agent_traces.jsonl').read().splitlines()[1])
for m in r['messages'][1:6]: print(f\"{m['role']:>9}: {m['content'][:78]}\")"
```

**Real output from this dataset:**

```
     user: Ticket T-9454 is back again. Work out if it is a repeat and whether we have be
assistant: TOOL: lookup_ticket_history(t-9454)
     user: RESULT: error — ticket id must be upper case
assistant: TOOL: lookup_ticket_history(T-9454)
     user: RESULT: 3 prior reports in 30 days
```

**Point at the lowercase call and the retry.** The errors are in the training data on
purpose — a model that has only seen clean traces has never learned to recover.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` | batch or sequence too large | `--bs 2` or `--seq 512`; add `--qlora` |
| Loss flat from step 1 | chat template mismatch, or loss on the wrong turns | check `assistant_only_loss` support in your TRL version |
| Output still waffles after training | adapter not loaded | check the path — PEFT fails quietly on a wrong directory |
| `bitsandbytes` import error with `--qlora` | wheel does not match CUDA | reinstall bitsandbytes against your CUDA version |
| Tokenizer complains about a pad token | model has none | already handled — the script sets pad to eos |
| `nvidia-smi dmon` not found | older driver package | use `watch -n1 nvidia-smi` instead |

---

## The rehearsal rule

Run Part 1 and Part 2 **end to end, twice**, on the actual box. Write the real numbers into
`DEMO-PLAN.md` and build your talk track around those, not around the placeholders here.

Capture a fallback for everything — finished checkpoints stay in `checkpoints/`, logs and
curves in `results/`, plus a screen recording of each run. A demo that fails live with
nothing to fall back on costs more credibility than skipping it.
