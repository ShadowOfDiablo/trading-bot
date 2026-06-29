#!/usr/bin/env bash
set -euo pipefail

echo "==> Updating system"
sudo apt update
sudo apt upgrade -y

echo "==> Installing base dependencies"
sudo apt install -y python3 python3-pip python3-venv git curl wget build-essential

echo "==> Creating app directory"
APP_DIR="/home/pi/trading-bot"
mkdir -p "$APP_DIR"
cd "$APP_DIR"

if [ ! -d "$APP_DIR/.git" ]; then
  echo "==> Cloning repository"
  sudo rm -rf "$APP_DIR"
  sudo git clone https://github.com/ShadowOfDiablo/trading-bot.git "$APP_DIR"
  cd "$APP_DIR"
else
  echo "==> Updating repository"
  git -C "$APP_DIR" pull || true
fi

echo "==> Creating virtual environment"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f "$APP_DIR/.env" ]; then
  echo "==> Creating .env from example"
  cp .env.example .env
  echo "Please edit $APP_DIR/.env and set your Trading212 and Telegram values."
fi

echo "==> Installing systemd service"
sudo cp "$APP_DIR/scripts/trading-bot.service" /etc/systemd/system/trading-bot.service
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl restart trading-bot

echo "==> Setup complete"
echo "Run: sudo systemctl status trading-bot"
echo "Run: sudo journalctl -u trading-bot -f"
