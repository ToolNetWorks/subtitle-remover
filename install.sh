#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/root/subtitle-remover"
SERVICE_FILE="/etc/systemd/system/subtitle-remover.service"

if ! command -v ffmpeg >/dev/null 2>&1; then
  apt-get update
  apt-get install -y ffmpeg python3-venv
fi

mkdir -p /var/lib/subtitle-remover/jobs
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

cp "$APP_DIR/subtitle-remover.service" "$SERVICE_FILE"
systemctl daemon-reload
systemctl enable --now subtitle-remover.service
systemctl --no-pager --full status subtitle-remover.service || true
