#!/usr/bin/env bash
# Brev setup for the fine-tuning workshop labs.
#
#   brev start --name ft-labs --gpu <A100-80GB-instance-type> \
#     --setup-script https://raw.githubusercontent.com/willson-nv/nvidia-finetuning-labs/main/setup.sh
#
# or, once you are already on the box:
#
#   curl -fsSL https://raw.githubusercontent.com/willson-nv/nvidia-finetuning-labs/main/setup.sh | bash
#
# Safe to re-run. Brev runs setup scripts on every instance start, and a stopped
# instance loses everything outside /home/ubuntu/workspace, so this script is
# written to be idempotent and to keep every expensive artefact on the persistent
# disk: the repo, the virtualenv and the ~16 GB of model weights.
set -euo pipefail

REPO_URL="https://github.com/willson-nv/nvidia-finetuning-labs.git"
WORK="${WORK:-/home/ubuntu/workspace}"
REPO="$WORK/nvidia-finetuning-labs"
VENV="$WORK/venv"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"

say() { printf '\n\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m!!\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------- 0. the GPU
say "GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "nvidia-smi not found. Nothing below will work on CPU in any useful time."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ------------------------------------------------- 1. persistent cache layout
# Only /home/ubuntu/workspace survives a stop/start. The default HF cache lives
# in ~/.cache, which does not — so without this you re-download 16 GB every time
# you resume the instance, which is exactly the wrong thing to discover on the
# morning of a workshop.
say "Persistent paths under $WORK"
mkdir -p "$WORK" "$WORK/hf"
export HF_HOME="$WORK/hf"
if ! grep -q 'HF_HOME' ~/.bashrc 2>/dev/null; then
  {
    echo ''
    echo '# fine-tuning workshop — keep the model cache on the persistent disk'
    echo "export HF_HOME=$WORK/hf"
    echo "export BASE=$BASE_MODEL"
    # guarded, so a missing or half-built venv does not throw an error into
    # every new login shell
    echo "[ -f $VENV/bin/activate ] && source $VENV/bin/activate"
  } >> ~/.bashrc
fi
echo "HF_HOME=$HF_HOME"

# --------------------------------------------------------------- 2. the repo
say "Repo"
# Never let git block on a credential prompt. Under Brev's systemd lifecycle
# there is no tty, so an auth prompt does not appear — it just dies with
# "could not read Username for 'https://github.com'", which reads like a network
# fault and is actually "this repo is private".
export GIT_TERMINAL_PROMPT=0

clone_url() {
  if [ -n "${GIT_TOKEN:-}" ]; then
    printf '%s' "$REPO_URL" | sed "s#https://#https://x-access-token:${GIT_TOKEN}@#"
  else
    printf '%s' "$REPO_URL"
  fi
}

if [ -d "$REPO/.git" ]; then
  git -C "$REPO" pull --ff-only || warn "pull failed — carrying on with the checkout already on disk"
elif [ -f "$REPO/scripts/make_data.py" ]; then
  # Copied up by hand (scp/rsync) rather than cloned. Perfectly fine.
  say "Using the existing checkout at $REPO (no git metadata)"
elif git clone "$(clone_url)" "$REPO" 2>/tmp/clone.err; then
  # Make sure a token never persists in .git/config.
  git -C "$REPO" remote set-url origin "$REPO_URL"
else
  sed 's/^/    /' /tmp/clone.err >&2
  cat >&2 <<EOF

  Could not clone $REPO_URL

  Almost always one of two things, and the git message does not distinguish them:

    1. The repo is PRIVATE. Anonymous clone fails. Either make it public, or
       re-run with a token:

         GIT_TOKEN=<personal, read-only, throwaway PAT> bash setup.sh

       Scope it to this one repo, and revoke it when the workshop is over.

    2. The branch has no setup.sh yet because it was never pushed. Check from
       your laptop:

         git log --oneline origin/main..HEAD

       Anything listed is not on GitHub, so neither is this script's repo copy.

  Or skip git entirely and copy the folder up from your laptop:

     scp -r nvidia-finetuning-labs ft-labs:$WORK/
     brev shell ft-labs -c "bash $REPO/setup.sh"

EOF
  exit 1
fi
git -C "$REPO" log --oneline -1 2>/dev/null || true

# ------------------------------------------------------- 3. python + packages
say "Virtualenv"
# Test for a working interpreter, not just the directory. A half-built venv —
# which is what you get when ensurepip is missing, because the tree is created
# before the failure — would otherwise be silently accepted here and only show
# up as "No such file or directory" on the activate below.
if [ ! -x "$VENV/bin/python" ]; then
  rm -rf "$VENV"
  if ! python3 -m venv "$VENV" 2>/tmp/venv.err; then
    sed 's/^/    /' /tmp/venv.err >&2
    warn "venv creation failed — installing the python venv package and retrying"
    PYV="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    sudo apt-get update -qq
    sudo apt-get install -y -qq "python${PYV}-venv" \
      || sudo apt-get install -y -qq python3-venv
    rm -rf "$VENV"
    python3 -m venv "$VENV"
  fi
fi
[ -f "$VENV/bin/activate" ] || { echo "venv still incomplete at $VENV" >&2; exit 1; }

# activate references PS1, which is unset in a non-interactive shell. Under
# `set -u` that aborts the script, so drop the check just for this line.
set +u
# shellcheck disable=SC1091
source "$VENV/bin/activate"
set -u
echo "  $(python -V) at $(command -v python)"
python -m pip install -q --upgrade pip wheel

# --- torch, matched to the driver ------------------------------------------
# CUDA 12 -> 13 was a MAJOR bump, so minor-version compatibility does not cover
# it: a cu130 wheel needs a 580+ driver and simply reports cuda=False on
# anything older. Brev's A100 image ships driver 565 (CUDA 12.7), while plain
# `pip install torch` now resolves to the cu130 build off PyPI — which is why
# you get a perfectly healthy nvidia-smi next to torch.cuda.is_available()
# False. Note cu128 is NOT an option here: it was dropped from the 2.13 build
# matrix, leaving cu126 as the CUDA 12 line.
say "PyTorch, matched to the driver"
CUDA_HDR="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -1)"
CUDA_MAJ="${CUDA_HDR%%.*}"
echo "  driver advertises CUDA ${CUDA_HDR:-unknown}"
case "${CUDA_MAJ:-}" in
  12)    TORCH_IDX="https://download.pytorch.org/whl/cu126" ;;
  13|14) TORCH_IDX="https://download.pytorch.org/whl/cu130" ;;
  *)     warn "could not read a CUDA version from nvidia-smi — using the PyPI default"
         TORCH_IDX="" ;;
esac
[ -n "$TORCH_IDX" ] && echo "  using $TORCH_IDX"

need_torch=1
if python -c "import torch" 2>/dev/null; then
  HAVE="$(python -c 'import torch; print(torch.version.cuda or "cpu")')"
  if [ "${HAVE%%.*}" = "${CUDA_MAJ:-}" ]; then
    need_torch=0
    echo "  torch already built for CUDA $HAVE — keeping it"
  else
    warn "installed torch is built for CUDA $HAVE but the driver is CUDA $CUDA_HDR — reinstalling"
  fi
fi
if [ "$need_torch" = 1 ]; then
  # shellcheck disable=SC2086
  pip install -q --force-reinstall ${TORCH_IDX:+--index-url "$TORCH_IDX"} torch
fi

say "Everything else (this is the slow step, ~3-5 min on a cold box)"
# transformers >= 4.51 is the floor for the qwen3 architecture; below it you get
# a bare `KeyError: 'qwen3'`, which is not an obvious error message at 9am.
# Installed after torch and from PyPI, so the pinned CUDA build is left alone.
pip install -q \
  "transformers>=4.51" \
  "peft" \
  "trl" \
  "datasets" \
  "accelerate" \
  "bitsandbytes"

# Record what actually got installed. Pin from this file, not from guesses.
pip freeze > "$REPO/env.lock"
echo "wrote $REPO/env.lock"

# ------------------------------------------------------------- 4. preflight
say "Preflight"
python - <<'PY'
import sys
import torch, transformers, peft, trl

print(f"  python        {sys.version.split()[0]}")
print(f"  torch         {torch.__version__}   cuda={torch.cuda.is_available()}")
print(f"  transformers  {transformers.__version__}")
print(f"  peft          {peft.__version__}")
print(f"  trl           {trl.__version__}")

if not torch.cuda.is_available():
    built = torch.version.cuda or "cpu-only"
    print(f"\n  FAIL: torch cannot see the GPU. This build targets CUDA {built}.")
    print("  If nvidia-smi works, the wheel does not match the driver. Reinstall with")
    print("  the index for the driver's CUDA major version:")
    print("      pip install --force-reinstall --index-url \\")
    print("        https://download.pytorch.org/whl/cu126 torch      # driver CUDA 12.x")
    print("      pip install --force-reinstall --index-url \\")
    print("        https://download.pytorch.org/whl/cu130 torch      # driver CUDA 13.x")
    sys.exit(1)

cap = torch.cuda.get_device_capability()
name = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  gpu           {name}  sm_{cap[0]}{cap[1]}  {total:.0f} GB")

# The QLoRA demo dies here or nowhere. bitsandbytes only compiles sm100/sm120
# into its CUDA 12.8+ and 13.x wheels; the 11.8-12.6 wheels stop at sm90. On an
# A100 (sm80) every wheel works, but check anyway so a later GPU switch surfaces
# now rather than mid-demo.
import bitsandbytes, torch as t
print(f"  bitsandbytes  {bitsandbytes.__version__}")
try:
    x = t.nn.Linear(64, 64).cuda()
    from bitsandbytes.nn import Linear4bit
    q = Linear4bit(64, 64, compute_dtype=t.bfloat16).cuda()
    _ = q(t.randn(2, 64, device="cuda", dtype=t.float16))
    print("  4-bit kernel  OK")
except Exception as e:
    print(f"  4-bit kernel  FAIL: {e}")
    print("  --> --qlora (Demo B) will not run on this GPU with this wheel.")
PY

# --------------------------------------------------- 5. datasets + the model
say "Datasets"
cd "$REPO/scripts"
python make_data.py --out ../data

say "Base model: $BASE_MODEL (~16 GB, cached to $HF_HOME)"
python - <<PY
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
m = "$BASE_MODEL"
AutoTokenizer.from_pretrained(m)
AutoModelForCausalLM.from_pretrained(m)
print("  cached OK:", m)
PY

# ------------------------------------------------ 6. the chat-template check
# Qwen3 is a hybrid thinking model and its template defaults to thinking ON.
# Left alone, every generation opens with a <think> block, the 256-token budget
# in eval_agent.py is spent reasoning, and the triage demo never emits clean
# JSON. Print the rendered prompts so the behaviour is visible before the day
# rather than inferred from a slide.
say "Chat template — confirm the <think> handling with your own eyes"
python - <<PY
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("$BASE_MODEL")
row = json.loads(open("../data/triage_train.jsonl").readline())
msgs = row["messages"]

print("\n--- TRAINING example, as TRL will render it -------------------")
print(repr(tok.apply_chat_template(msgs, tokenize=False)))

for flag in (True, False):
    print(f"\n--- GENERATION prompt, enable_thinking={flag} ------------------")
    print(repr(tok.apply_chat_template(msgs[:-1], add_generation_prompt=True,
                                       tokenize=False, enable_thinking=flag)))
print("""
Read those three. The training render and the enable_thinking=False prompt
should end the same way. If they do, always pass enable_thinking=False at
inference and the demo is consistent. If they differ, fix that before the day.
""")
PY

# ---------------------------------------------------------------- 7. summary
say "Ready"
cat <<EOF

  repo     $REPO
  venv     $VENV          (auto-activated on your next login)
  cache    $HF_HOME
  model    \$BASE = $BASE_MODEL

  Open a fresh shell so ~/.bashrc takes effect, then:

    cd $REPO/scripts
    python reward.py                       # 12-line grader, no GPU needed
    python train_lora.py --model \$BASE --out ../checkpoints/demo-a

  Full command-by-command walkthrough: $REPO/RUN.md

  Before you stop this instance, push anything you care about. A stopped Brev
  instance can fail to restart if the provider is out of A100 capacity, and the
  data is unreachable until it is not.

EOF
