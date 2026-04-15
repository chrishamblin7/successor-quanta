#!/usr/bin/env bash
set -euo pipefail

# Setup-only startup script for successor-quanta.
# Clones repo, installs deps, and syncs experiment data from GCS.
# Does NOT launch any experiments.

echo "[successor-quanta] Starting setup (user: $USER, home: $HOME)"

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

# Sync experiment results from GCS
echo "[successor-quanta] Syncing experiment data from GCS..."
mkdir -p experiments
gsutil -m rsync -r gs://cloud/misc/chris/successor-quanta/ experiments/ || {
  echo "[successor-quanta] GCS sync failed or bucket empty -- continuing"
}

echo "[successor-quanta] Setup complete. Instance is ready."
echo "[successor-quanta] Experiment dirs:"
ls -d experiments/L*/ 2>/dev/null || echo "  (none found)"
