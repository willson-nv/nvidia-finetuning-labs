# Running these labs on NVIDIA Brev

Target box: **one A100 80GB**. Everything here assumes you are driving from your
own laptop and the audience only sees your screen.

Do this **the week before**, not the morning of. The one step that can genuinely
fail on the day is capacity — A100s are popular and Brev pulls from whatever
provider has stock.

---

## 0. Publish the repo first — everything else depends on it

Both the `--setup-script` URL and the clone inside `setup.sh` fetch from GitHub
anonymously. The Brev lifecycle script runs under systemd with no terminal, so a
private repo cannot prompt for credentials — it dies with
`could not read Username for 'https://github.com'`, which looks like a network
fault and is not one.

**From your laptop:**

```bash
cd nvidia-finetuning-labs
git log --oneline origin/main..HEAD     # anything listed is NOT on GitHub yet
git push origin main
```

Then make the repo public: **GitHub → the repo → Settings → General → Danger
Zone → Change repository visibility → Public.**

**Verify both URLs before you spend money on a GPU:**

```bash
git ls-remote https://github.com/willson-nv/nvidia-finetuning-labs.git | head -1
curl -fsSI https://raw.githubusercontent.com/willson-nv/nvidia-finetuning-labs/main/setup.sh | head -1
```

You want a commit hash from the first and `HTTP/2 200` from the second. A 404 on
the second means the push has not landed on `main`.

Staying private instead? Everything below still works, but pass a token:

```bash
GIT_TOKEN=<personal, read-only, throwaway PAT> bash setup.sh
```

Scope it to this one repo and revoke it when the workshop is over. `setup.sh`
resets the remote afterwards so the token is not left behind in `.git/config`.

---

## 1. Install and log in (once, on your laptop)

```bash
brew install brevdev/homebrew-brev/brev
brev --version
brev login
```

`brev login` opens a browser. Credentials and SSH keys land in `~/.brev/`.

---

## 2. Find an A100 80GB you can actually get

```bash
brev search --gpu-name A100 --min-vram 80 --sort price --wide
```

You get a table of instance types across providers with `$/HR`, boot time, and a
`FEATURES` column. **Pick one with an `S`** — stoppable, so you can pause between
your rehearsal and the workshop instead of paying to idle or rebuilding.

```
TYPE            GPU    COUNT  VRAM   TOTAL   $/HR    BOOT  FEATURES
...             A100   1      80GB   80GB    $x.xx   Xm    S R P
```

Copy the `TYPE` string. It is provider-specific, so there is no single correct
value to hard-code here.

---

## 3. Launch it with the setup script attached

```bash
brev start --name ft-labs \
  --gpu "<TYPE from step 2>" \
  --setup-script https://raw.githubusercontent.com/willson-nv/nvidia-finetuning-labs/main/setup.sh
```

Brev runs `setup.sh` automatically once the instance is up. It clones the repo,
builds a virtualenv, installs the stack, generates the datasets, pulls the ~16 GB
base model, and runs a preflight that checks CUDA, the bitsandbytes 4-bit kernel,
and the chat template.

Budget **10–15 minutes** for the first run, nearly all of it the model download.

Prefer to watch it happen? Launch bare and run the script yourself:

```bash
brev start --name ft-labs --gpu "<TYPE>" --empty
brev shell ft-labs
curl -fsSL https://raw.githubusercontent.com/willson-nv/nvidia-finetuning-labs/main/setup.sh | bash
```

---

## 4. Connect

```bash
brev shell ft-labs        # or: ssh ft-labs
```

If SSH fails or you created the instance in the web console instead:

```bash
brev refresh              # re-syncs ~/.brev/ssh_config with current IPs
brev list
```

For a real editor, VS Code Remote-SSH connects to the host name `ft-labs` with no
extra configuration once `brev refresh` has run.

Your first login auto-activates the virtualenv and sets `$BASE` and `$HF_HOME`.
Check it landed:

```bash
echo $BASE          # Qwen/Qwen3-8B
nvidia-smi
cd /home/ubuntu/workspace/nvidia-finetuning-labs/scripts
python reward.py    # instant, no GPU — proves the environment is alive
```

Then follow **RUN.md** command by command.

---

## 5. The two things that will bite you

### Only `/home/ubuntu/workspace` survives a stop

| Location | Survives stop | Survives delete |
|---|---|---|
| `/home/ubuntu/workspace` | yes | no |
| `~/.cache`, `/tmp` | **no** | no |
| installed system packages | yes | no |

The default Hugging Face cache is `~/.cache/huggingface`, which is on the wrong
side of that line. `setup.sh` repoints `HF_HOME` to
`/home/ubuntu/workspace/hf` and writes it into `~/.bashrc` — so keep using the
login shell rather than exporting your own paths, or you will re-download 16 GB
after your first stop.

### Stopping is not free of risk

When you stop, Brev hands the GPU back to the provider. On restart it tries to
get the *same* GPU type in the *same* region. If that capacity is gone, the
restart fails and your data is unreachable until it returns.

So: **push to git before every stop**, and if you are pausing for more than a day
consider `brev delete` and a clean relaunch instead — `setup.sh` rebuilds the box
in about fifteen minutes, and you stop paying for storage.

```bash
git -C /home/ubuntu/workspace/nvidia-finetuning-labs status
brev stop ft-labs
```

Checkpoints and logs are gitignored on purpose — they are large and regenerable.
If you want to keep a good run, pull it down instead:

```bash
scp -r ft-labs:/home/ubuntu/workspace/nvidia-finetuning-labs/results ./results-from-brev
```

---

## 6. Workshop-day sequence

```bash
brev start ft-labs        # restart the stopped instance
brev refresh              # the IP will have changed
brev shell ft-labs
nvidia-smi                # confirm the GPU before you talk to anyone
cd /home/ubuntu/workspace/nvidia-finetuning-labs/scripts
python reward.py          # 5-second smoke test
```

Start it **before** the room fills. If capacity has vanished overnight you want
to find out with an hour in hand, not while sharing your screen.

---

## 7. Why A100 80GB and not something faster

The training runs here are tiny — about 80,000 tokens for Demos A and B, 230,000
for Demo D. An H100 would finish maybe ninety seconds sooner and the model
download would still dominate. What the A100 buys you is boring reliability:
`sm_80` is in every bitsandbytes wheel ever built, bf16 is native, and 80 GB
means nothing OOMs while you are talking.

The one thing the big card costs you is drama in Demo B: on 80 GB both the LoRA
and QLoRA runs fit with ~50 GB spare, so the memory saving is a number rather
than a constraint anyone feels. If you want that demo to land harder, quote the
peak-memory figures against a 24 GB workstation card rather than against the 80
you are running on.

---

## Reference

- [Brev quickstart](https://docs.nvidia.com/brev/getting-started/quickstart)
- [Instance management and setup scripts](https://docs.nvidia.com/brev/cli/instance-management)
- [GPU search and filtering](https://docs.nvidia.com/brev/cli/search-discovery)
- [Data persistence and lifecycle](https://docs.nvidia.com/brev/concepts/gpu-instances)
