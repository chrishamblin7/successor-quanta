#!/usr/bin/env bash
set -euo pipefail

# Startup script for 8xA100-40 cloud instance.
# Runs 8 successor-quanta experiments in parallel, one per GPU.

echo "[successor-quanta] Starting (user: $USER, home: $HOME)"

sudo touch /root/no_autodelete

# Prerequisites
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends make curl ca-certificates nvtop

# UV
echo "[UV] Installing uv..."
curl -fsSL https://astral.sh/uv/install.sh | bash
# shellcheck disable=SC1090
source "$HOME/.local/bin/env"
export PATH="$HOME/.local/bin:$PATH"

# Clone repo
PROJECTS_DIR="$HOME/projects"
mkdir -p "$PROJECTS_DIR"
cd "$PROJECTS_DIR"

echo "[successor-quanta] Cloning repo..."
git clone "https://github.com/chrishamblin7/successor-quanta.git" successor-quanta
cd successor-quanta

# Setup environment
echo "[successor-quanta] Setting up environment..."
uv venv -p "3.11" --managed-python --clear
source .venv/bin/activate
uv sync

###############################################################################
# Training jobs (8 experiments, one per GPU)
###############################################################################

CONFIGS_DIR="experiments/configs"
TRAIN_CMD="python experiments/train_successor.py"

run_experiment() {
  local gpu="$1"
  local config="$2"
  local name
  name="$(basename "$config" .yaml)"

  echo "[Train] GPU=${gpu} config=${config} name=${name}"
  CUDA_VISIBLE_DEVICES="$gpu" $TRAIN_CMD --config "$config" &
}

run_experiment 0 "$CONFIGS_DIR/L2_D128_base2_rope.yaml"
PID_0=$!
run_experiment 1 "$CONFIGS_DIR/L6_D128_base2_rope.yaml"
PID_1=$!
run_experiment 2 "$CONFIGS_DIR/L2_D512_base2_rope.yaml"
PID_2=$!
run_experiment 3 "$CONFIGS_DIR/L2_D128_base2_rope_wd.yaml"
PID_3=$!
run_experiment 4 "$CONFIGS_DIR/L2_D128_base3_rope.yaml"
PID_4=$!
run_experiment 5 "$CONFIGS_DIR/L2_D128_base2_rope_powerlaw.yaml"
PID_5=$!
run_experiment 6 "$CONFIGS_DIR/L2_D128_base2_learned.yaml"
PID_6=$!
run_experiment 7 "$CONFIGS_DIR/L2_D128_base2_sinusoidal.yaml"
PID_7=$!

echo "[Train] Launched 8 experiments: pids=${PID_0} ${PID_1} ${PID_2} ${PID_3} ${PID_4} ${PID_5} ${PID_6} ${PID_7}"

FAIL=0
for pid in $PID_0 $PID_1 $PID_2 $PID_3 $PID_4 $PID_5 $PID_6 $PID_7; do
  wait "$pid" || FAIL=1
done

if [ "$FAIL" -ne 0 ]; then
  echo "[Train] One or more experiments failed."
else
  echo "[Train] All experiments completed successfully."
fi

sudo shutdown -h now
