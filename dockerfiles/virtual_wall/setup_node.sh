#!/bin/bash
# Set up a Virtual Wall 2 node for online amortized-DP training.
#
#   setup_node.sh qlever    Docker + the QLever index bundle, swap off, then start one pinned
#                           single-threaded QLever instance per core (index locked in RAM).
#   setup_node.sh trainer   Python venv with requirements.txt + the trainer data bundle.
#
# The bundles are made with make_data_bundles.sh and downloaded from Google Drive (share
# links set to "Anyone with the link"). Re-running is safe: finished steps are skipped.
# Optional overrides: LAN_IP (default: this node's 192.168.x.y address), REPO_DIR, BRANCH.
set -euo pipefail

ROLE="${1:-}"
if [[ "$ROLE" != "qlever" && "$ROLE" != "trainer" ]]; then
  echo "Usage: $0 qlever|trainer" >&2
  exit 1
fi

QLEVER_BUNDLE_URL="https://drive.google.com/file/d/1Y6OCSyGr3KCkfStUZRb_hLLK5QKfEYe4/view?usp=drive_link"
TRAINER_BUNDLE_URL="https://drive.google.com/file/d/1iJc9nJdx-icsumf8cRPR8WnXGgWFkuIM/view?usp=drive_link"
REPO_URL="https://github.com/RubenEschauzier/rl-based-join-optimization.git"
REPO_DIR="${REPO_DIR:-/users/reschauz/rl-based-join-optimization}"
BRANCH="${BRANCH:-master}"
TOOLS_VENV="$HOME/.venv-setup-tools"
export DEBIAN_FRONTEND=noninteractive

# Enable NAT for Virtual Wall internet access
wget -O - -nv --cipher DEFAULT@SECLEVEL=1 https://www.wall2.ilabt.iminds.be/enable-nat.sh | sudo bash

# Grow the root partition first: the default root disk is too small for the data.
if [ ! -f "$HOME/.root_disk_expanded" ]; then
  echo "Expanding root disk..."
  echo "y" | sudo bash -e -c '. <(wget -O - -q https://gitlab.ilabt.imec.be/wvdemeer/wall-public-scripts/-/raw/master/expand-root-disk.sh)'
  touch "$HOME/.root_disk_expanded"
fi

echo "Installing required dependencies..."
sudo apt update
sudo apt install -y ca-certificates curl gnupg git zstd python3 python3-venv python3-pip

if ! command -v docker > /dev/null; then
  echo "Installing Docker..."
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.asc > /dev/null
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt update
  sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  sudo systemctl enable --now docker
fi
sudo usermod -aG docker "$USER"

echo "Cloning repository..."
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi

# gdown handles Google Drive's confirmation page for large files.
if [ ! -x "$TOOLS_VENV/bin/gdown" ]; then
  python3 -m venv "$TOOLS_VENV"
  "$TOOLS_VENV/bin/pip" install -q gdown
fi

fetch_bundle() {   # $1 = share link, $2 = bundle name; extracts into the repository
  local url="$1" name="$2" marker="$REPO_DIR/.bundle_${2%%.*}_done"
  if [ -f "$marker" ]; then
    echo "$name already extracted."
    return
  fi
  if [[ "$url" == *REPLACE_WITH* ]]; then
    echo "Set the Google Drive link for $name at the top of $0." >&2
    exit 1
  fi
  echo "Downloading $name..."
  local download="$HOME/$name"
  "$TOOLS_VENV/bin/gdown" --fuzzy "$url" -O "$download"
  echo "Extracting $name..."
  zstd -dc "$download" | tar -xf - -C "$REPO_DIR"
  rm -f "$download"
  touch "$marker"
}

if [ "$ROLE" = "qlever" ]; then
  fetch_bundle "$QLEVER_BUNDLE_URL" qlever_bundle.tar.zst

  # Locked index pages cannot be swapped, but QLever's own memory can; on a hard disk that
  # would put disk reads back into the measured latencies.
  sudo swapoff -a

  LAN_IP="${LAN_IP:-$(ip -4 -o addr show | awk '{print $4}' | cut -d/ -f1 | grep '^192\.168\.' | head -1 || true)}"
  if [ -z "$LAN_IP" ]; then
    echo "No 192.168.x.y address found: is the node on an experiment LAN with automatic IPv4?" >&2
    exit 1
  fi
  echo "Starting QLever instances on $LAN_IP..."
  # sg: the docker group membership added above does not apply to this shell yet.
  sg docker -c "python3 '$REPO_DIR/data/qlever/deploy_isolated_qlever_instances.py' \
    '$REPO_DIR/data/qlever/qlever_yago' qleverfile_default '$LAN_IP' qlever_yago.env --no-index"
  echo "Setup complete: QLever endpoints listed in $REPO_DIR/data/qlever/qlever_yago/endpoints_$LAN_IP.json"
else
  fetch_bundle "$TRAINER_BUNDLE_URL" trainer_bundle.tar.zst
  # The code needs Python >= 3.10 (int.bit_count, scipy 1.13, pandas 2.2); Ubuntu 20.04
  # ships 3.8, so take 3.10 from the deadsnakes PPA there.
  if ! command -v python3.10 > /dev/null; then
    echo "Installing Python 3.10..."
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.10 python3.10-venv python3.10-dev
  fi
  if [ ! -x "$REPO_DIR/.venv/bin/python" ]; then
    echo "Creating Python environment..."
    python3.10 -m venv "$REPO_DIR/.venv"
  fi
  "$REPO_DIR/.venv/bin/pip" install -q -r "$REPO_DIR/requirements.txt"
  echo "Setup complete: activate with 'source $REPO_DIR/.venv/bin/activate'."
fi
