#!/usr/bin/env python3
"""Demos A and B — the SAME script. Demo B just adds --qlora.

That is the whole point: if the two runs used different code the memory comparison
would prove nothing.

    # Demo A
    python3 train_lora.py --model <HF_ID> --out ../checkpoints/demo-a
    # Demo B — one flag different
    python3 train_lora.py --model <HF_ID> --out ../checkpoints/demo-b --qlora

Pin --model the week of the workshop; this ecosystem moves fast.
Requires: transformers, peft, trl, datasets, accelerate (+ bitsandbytes for --qlora).
"""
import argparse, json, time, pathlib


def _accepted(cls):
    import dataclasses, inspect
    if dataclasses.is_dataclass(cls):
        return {f.name for f in dataclasses.fields(cls)}
    return set(inspect.signature(cls.__init__).parameters) - {"self"}


# Argument names drift between transformers/TRL majors — transformers v5, for
# instance, deprecated warmup_ratio in favour of warmup_steps, which now also
# takes a float < 1 and means the same thing. Pinning versions for a workshop
# that runs on whatever pip resolved this morning is a losing game, so instead
# translate what we know and refuse to run quietly on what we do not.
RENAMES = {
    "warmup_ratio":        "warmup_steps",
    "evaluation_strategy": "eval_strategy",
    "max_seq_length":      "max_length",
    "max_length":          "max_seq_length",
}

# Dropping these silently would change what the demo actually demonstrates,
# so they are fatal rather than a warning.
LOAD_BEARING = {"assistant_only_loss", "gradient_accumulation_steps", "bf16"}


def build_sft_config(SFTConfig, **kw):
    valid = _accepted(SFTConfig)
    out, dropped = {}, []
    for k, v in kw.items():
        if k in valid:
            out[k] = v
        elif k in RENAMES and RENAMES[k] in valid:
            out[RENAMES[k]] = v
            print(f"  note: SFTConfig here wants '{RENAMES[k]}', not '{k}' — translated")
        else:
            dropped.append(k)
    fatal = [k for k in dropped if k in LOAD_BEARING]
    if fatal:
        raise SystemExit(
            f"\n  SFTConfig in this TRL build does not accept {fatal}.\n"
            "  These change what is trained, so continuing would quietly demo\n"
            "  something other than what the slides claim. Check the TRL release\n"
            "  notes for the new spelling before the workshop.\n"
            f"  TRL accepted args: {sorted(valid)}\n")
    if dropped:
        print(f"  warning: SFTConfig does not accept {dropped} — dropped (cosmetic only)")
    return SFTConfig(**out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HuggingFace id of an instruct model")
    ap.add_argument("--data", default="../data/triage_train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--qlora", action="store_true", help="load the base model in 4-bit")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    mode = "QLoRA (4-bit base)" if a.qlora else "LoRA (bf16 base)"
    print(f"\n=== {mode} · rank {a.rank} · {a.epochs} epochs ===\n")

    quant = None
    if a.qlora:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,          # "compress the compression"
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, quantization_config=quant, dtype=torch.bfloat16, device_map="auto")

    peft_cfg = LoraConfig(
        r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    ds = load_dataset("json", data_files=a.data, split="train")

    cfg = build_sft_config(
        SFTConfig,
        output_dir=a.out, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bs, gradient_accumulation_steps=4,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=5, save_strategy="epoch", bf16=True,
        max_length=a.seq, gradient_checkpointing=True, report_to=[],
        # loss only on the answer, not on the question — see the data slide
        assistant_only_loss=True,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=peft_cfg)

    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    trainer.save_model(a.out)

    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    stats = {"mode": mode, "rank": a.rank, "minutes": round(secs / 60, 1),
             "peak_gpu_gb": round(peak, 1)}
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "run_stats.json").write_text(json.dumps(stats, indent=2))

    print("\n" + "=" * 52)
    print(f"  {mode}")
    print(f"  peak GPU memory   {stats['peak_gpu_gb']:>8} GB   <-- the Demo B number")
    print(f"  wall clock        {stats['minutes']:>8} min")
    print(f"  adapter written   {a.out}")
    print("=" * 52 + "\n")

if __name__ == "__main__":
    main()
