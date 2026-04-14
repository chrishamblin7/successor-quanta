#!/usr/bin/env bash
set -euo pipefail

# Sweep 2: Base x Width experiments on existing 8xA100-40 instance.
# Assumes the repo is already cloned and env is set up from sweep 1.

echo "[sweep2] Starting sweep 2 (base x width grid)"

cd ~/projects/successor-quanta
git pull origin main
source .venv/bin/activate

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

run_experiment 0 "$CONFIGS_DIR/L2_D256_base2_rope.yaml"
PID_0=$!
run_experiment 1 "$CONFIGS_DIR/L2_D512_base2_rope_v2.yaml"
PID_1=$!
run_experiment 2 "$CONFIGS_DIR/L2_D128_base10_rope.yaml"
PID_2=$!
run_experiment 3 "$CONFIGS_DIR/L2_D256_base10_rope.yaml"
PID_3=$!
run_experiment 4 "$CONFIGS_DIR/L2_D512_base10_rope.yaml"
PID_4=$!
run_experiment 5 "$CONFIGS_DIR/L2_D128_base20_rope.yaml"
PID_5=$!
run_experiment 6 "$CONFIGS_DIR/L2_D256_base20_rope.yaml"
PID_6=$!
run_experiment 7 "$CONFIGS_DIR/L2_D512_base20_rope.yaml"
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
