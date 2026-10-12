#!/bin/bash
# Slurm launcher for PPO training on the Teaching Labs cluster. From the project root:
#
#   mkdir -p runs
#   sbatch -p general --gres gpu -c 2 --time=180 -J pf-ppo \
#          -o runs/ppo-%j.out -e runs/ppo-%j.err scripts/train_ppo.sh --time-limit-min 170
#
# Arguments are passed on to python/train_ppo.py. The Weights & Biases key is
# read from ~/.config/wandb/api_key unless WANDB_API_KEY is already set.
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

make sim
if [ -z "$WANDB_API_KEY" ] && [ -f "$HOME/.config/wandb/api_key" ]; then
    export WANDB_API_KEY="$(cat "$HOME/.config/wandb/api_key")"
fi
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec python3 -u python/train_ppo.py "$@"
