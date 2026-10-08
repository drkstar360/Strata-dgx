#!/bin/sh
# Entry point for the Strata container. The engine is compiled during docker
# build and lives in the image, so the first start only downloads the model.
# The install config is kept on the /data volume so a recreated container skips
# the setup pass and goes straight to serving.
set -e
cd /opt/strata || exit 1

STRATA_DATA="${STRATA_DATA:-/data}"
FAMILY="${FAMILY:-qwen}"
MODEL="${MODEL:-IQ2_XS}"
CONTEXT="${CONTEXT:-32768}"
VISION="${VISION:-no}"          # no | yes | cpu (the image encoder on the CPU)
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-}"
KV="${KV:-}"                    # int8 | q4_0 | k8v4; empty: setup.py's own default (int8)
GPUS="${GPUS:-}"                # "0,2" or "all": one model across several cards (docs/MULTI_GPU.md)
GPU="${GPU:-}"                  # one card, numbered as nvidia-smi numbers them
LAYER_SPLIT="${LAYER_SPLIT:-}"  # with GPUS: where each later card's layers start (default: auto)
LOW_RAM="${LOW_RAM:-auto}"      # on: the experts come from the pack's experts.bin, not from RAM
GGUF_DIR="${GGUF_DIR:-}"        # a mounted folder with GGUF files you already have: no download
RESIDENT_BUDGET_GIB="${RESIDENT_BUDGET_GIB:-}"   # UD-Q4_K_XL: GiB of experts kept in RAM (default: setup's pick)
KV_STREAMING="${KV_STREAMING:-}" # auto | on | off; empty: setup.py's own default (auto)
STRATA_BIND="${STRATA_BIND:-127.0.0.1}"   # docker-compose.spark.yml: the host address the port is published on

# Never reachable beyond this machine without an API key (AGENTS.md): docker-compose.spark.yml publishes the port on
# STRATA_BIND, so any other address needs API_KEY. Checked first, before a setup pass downloads anything.
case "$STRATA_BIND" in
  127.0.0.1|localhost|::1) ;;
  *) if [ -z "$(printf '%s' "$API_KEY" | tr -d ' \t\r\n')" ]; then
       echo "STRATA_BIND=$STRATA_BIND publishes the server beyond this machine: set API_KEY too (clients send it as" >&2
       echo "Authorization: Bearer <key> or x-api-key), or keep STRATA_BIND=127.0.0.1." >&2
       exit 1
     fi ;;
esac
# The key the server checks on every start (it reads STRATA_API_KEY before the config's), so a new key needs no setup
# pass. Exported only when set: an empty STRATA_API_KEY stops the server (#213).
if [ -n "$API_KEY" ]; then export STRATA_API_KEY="$API_KEY"; fi

# setup.py starts the newest strata-*.json it finds, so link in exactly the one
# this family and model were set up with. The config is the recorded output of
# that setup (the pack, the profile, the quant, the KV decision), not settings
# the entry point could rebuild from env vars. qwen has an empty family tag.
case "$FAMILY" in qwen) prefix="" ;; *) prefix="${FAMILY}-" ;; esac
tag="${prefix}$(printf '%s' "$MODEL" | tr 'A-Z' 'a-z')"
cfg="$STRATA_DATA/config/strata-$tag.json"
mkdir -p "$STRATA_DATA/config"

# REINSTALL is only needed to change settings for a model that is already set up
# (context, vision, KV, host, api_key). Switching between models already on the
# volume needs no setup pass: their config is already there.
#
# KV / GPUS / GPU / LAYER_SPLIT are passed only when set, so an unset one keeps
# setup.py's own default. LOW_RAM is always passed: setup.py measures the PC's RAM
# from /proc/meminfo, which in a container is the host's total, not the container's
# limit, so a memory-capped container has to ask for the low-RAM mode itself.
if [ "${REINSTALL:-0}" = "1" ] || [ ! -f "$cfg" ]; then
  if [ -n "$GGUF_DIR" ]; then
    echo "Setting up $tag from the GGUF files in $GGUF_DIR (the engine is already in the image)."
  else
    echo "Setting up $tag: downloading the model (~70 GB; the engine is already in the image)."
  fi
  set -- --family "$FAMILY" --model "$MODEL" --context "$CONTEXT" --vision "$VISION" \
    --data-dir "$STRATA_DATA" --host "$HOST" --api-key "$API_KEY" \
    --port "$PORT" --no-start --low-ram "$LOW_RAM"
  if [ -n "$KV" ]; then set -- "$@" --kv "$KV"; fi
  if [ -n "$GPUS" ]; then set -- "$@" --gpus "$GPUS"; fi
  if [ -n "$GPU" ]; then set -- "$@" --gpu "$GPU"; fi
  if [ -n "$LAYER_SPLIT" ]; then set -- "$@" --layer-split "$LAYER_SPLIT"; fi
  if [ -n "$GGUF_DIR" ]; then set -- "$@" --gguf-dir "$GGUF_DIR"; fi
  if [ -n "$RESIDENT_BUDGET_GIB" ]; then set -- "$@" --resident-budget-gib "$RESIDENT_BUDGET_GIB"; fi
  if [ -n "$KV_STREAMING" ]; then set -- "$@" --kv-streaming "$KV_STREAMING"; fi
  .venv/bin/python setup.py --setup --yes "$@"
  [ -e "/opt/strata/strata-$tag.json" ] && { cmp -s "/opt/strata/strata-$tag.json" "$cfg" || cp -f "/opt/strata/strata-$tag.json" "$cfg"; }
else
  # #1244: the copy on the volume is the one that counts, so a regular file left in /opt/strata by an earlier setup
  # (or by an image built with one) must not stand in for it: edits to /data/config would be ignored
  ln -sfn "$cfg" "/opt/strata/strata-$tag.json"
fi

# Settings applied to the model's config on every start, after any setup pass (which rewrites the engine's args), in
# the config on the volume and in the one that starts (a copy after a setup pass, else the link to it). Unset, the
# config's own value stays (what the web page or an earlier start wrote).
#  MODEL_ALIASES ("qwen,local-model"): other names the model answers to and /v1/models lists (the config's "aliases",
#    docs/DETAILS.md "Model aliases").
#  EXPERT_CACHE (auto | N): the engine's --expert-cache, the GPU's copy of the most-used experts in slots of one expert
#    each. auto takes the GPU memory left after the model loads; on a GPU that shares the RAM (DGX Spark) that is
#    MemAvailable less STRATA_UMA_HEADROOM_GIB, a second copy of experts already in RAM (docs/DGX_SPARK.md).
#  PARALLEL (N): up to N requests decode together in the engine's batch slots, the config's "parallel"
#    (docs/BATCHING.md); 1 is one at a time, more wait their turn. The server caps it at 8 and says so.
if [ -n "${MODEL_ALIASES:-}" ] || [ -n "${EXPERT_CACHE:-}" ] || [ -n "${PARALLEL:-}" ]; then
  python3 - "${MODEL_ALIASES:-}" "${EXPERT_CACHE:-}" "${PARALLEL:-}" "$cfg" "/opt/strata/strata-$tag.json" <<'PYEOF'
import json, os, sys
aliases, cache, parallel = sys.argv[1], sys.argv[2].strip().lower(), sys.argv[3].strip()
if cache and cache != "auto" and not (cache.isdigit() and int(cache) > 0):
    sys.exit(f"EXPERT_CACHE={sys.argv[2]!r}: expected auto or a whole number of slots above 0 (the engine needs a cache)")
if parallel and not (parallel.isdigit() and int(parallel) > 0):
    sys.exit(f"PARALLEL={sys.argv[3]!r}: expected a whole number of requests at once, 1 or more (1: one at a time)")
names = [x.strip() for x in aliases.split(",") if x.strip()]
for path in dict.fromkeys(os.path.realpath(p) for p in sys.argv[4:] if os.path.isfile(p)):
    with open(path, encoding="utf-8-sig") as f:
        cfg = json.load(f)
    old = json.dumps(cfg)
    if aliases:
        cfg["aliases"] = names
    if cache:
        args = [str(x) for x in cfg.get("args") or []]
        if "--expert-cache" in args[:-1]:
            args[args.index("--expert-cache") + 1] = cache
        else:
            args += ["--expert-cache", cache]
        cfg["args"] = args
    if parallel:
        cfg["parallel"] = int(parallel)
    if json.dumps(cfg) != old:
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(cfg, indent=1))
if aliases:
    print("Model aliases: " + (", ".join(names) or "none"))
if cache:
    print("Expert cache: " + cache)
if parallel:
    print(f"Requests at once: {int(parallel)}")
PYEOF
fi

# Later starts skip straight here: setup.py finds the installed config and
# launches serve/server.py (OpenAI- and Anthropic-compatible API on :8080).
# GPUS / GPU / LAYER_SPLIT are repeated on purpose. Given at the start they pin the
# cards for this model, and setup.py saves them in its config; without them a config
# that names one card is offered once to a pair, on its own, when the host has two
# cards that can share the model (setup.py's offer_together, docs/MULTI_GPU.md).
set -- --port "$PORT"
if [ -n "$GPUS" ]; then set -- "$@" --gpus "$GPUS"; fi
if [ -n "$GPU" ]; then set -- "$@" --gpu "$GPU"; fi
if [ -n "$LAYER_SPLIT" ]; then set -- "$@" --layer-split "$LAYER_SPLIT"; fi
exec .venv/bin/python setup.py "$@"
