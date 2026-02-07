#!/bin/bash
set -e

# ═══════════════════════════════════════════════════════════════════════════════
#  AWS Server Setup — Epstein Files Search Engine
#
#  Run this on a fresh Ubuntu 22.04/24.04 EC2 instance:
#    curl -sL <raw-url>/deploy/aws-setup.sh | bash
#
#  Or clone the repo first:
#    git clone <repo-url> /opt/abovea.cloud
#    cd /opt/abovea.cloud && bash deploy/aws-setup.sh
#
#  Requirements:
#    - Ubuntu 22.04 or 24.04 (Amazon Linux 2023 also works with minor changes)
#    - Minimum: t3.medium (2 vCPU, 4 GB RAM) — 8 GB RAM recommended
#    - 30+ GB disk (data + index = ~15 GB)
#    - Security group: allow TCP 80 (HTTP), 443 (HTTPS), 22 (SSH)
# ═══════════════════════════════════════════════════════════════════════════════

APP_DIR="/opt/abovea.cloud"
LOG_DIR="/var/log/epstein-search"

echo ""
echo "═══════════════════════════════════════════════"
echo "  AWS Server Setup — Epstein Files Search"
echo "═══════════════════════════════════════════════"
echo ""

# ─── 1. System packages ─────────────────────────────────────────────────────

echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 python3-venv python3-pip \
    nginx \
    git \
    curl \
    > /dev/null 2>&1
echo "  ✓ System packages installed"

# ─── 2. Create app directory and user permissions ────────────────────────────

echo ""
echo "[2/6] Setting up directories..."

sudo mkdir -p "$APP_DIR"
sudo mkdir -p "$LOG_DIR"

# If repo isn't already cloned here, clone it
if [ ! -f "$APP_DIR/app.py" ]; then
    if [ -f "$(pwd)/app.py" ]; then
        echo "  Copying project from $(pwd) to $APP_DIR..."
        sudo cp -r "$(pwd)/." "$APP_DIR/"
    else
        echo "  ERROR: app.py not found."
        echo "  Clone the repo to $APP_DIR first, or run this script from the repo root."
        exit 1
    fi
fi

sudo chown -R www-data:www-data "$APP_DIR"
sudo chown -R www-data:www-data "$LOG_DIR"
echo "  ✓ Directories ready"

# ─── 3. Python environment + dependencies ────────────────────────────────────

echo ""
echo "[3/6] Setting up Python environment..."
cd "$APP_DIR"

if [ ! -d "venv" ]; then
    sudo -u www-data python3 -m venv venv
fi
sudo -u www-data venv/bin/pip install -q -r requirements.txt
sudo -u www-data venv/bin/pip install -q gunicorn
echo "  ✓ Python environment ready"

# ─── 4. Build the search index (or sync from Drive) ─────────────────────────

echo ""
echo "[4/6] Building search index..."

if [ -f "$APP_DIR/data/index/vectors.faiss" ]; then
    echo "  ✓ Index already exists, skipping build"
else
    echo "  Running full setup (downloads data + builds index)..."
    echo "  This will download ~15 GB of data — grab a coffee."
    sudo -u www-data bash setup.sh
fi

# ─── 5. Configure nginx ─────────────────────────────────────────────────────

echo ""
echo "[5/6] Configuring nginx..."

# Remove default site
sudo rm -f /etc/nginx/sites-enabled/default

# Install our config
sudo cp "$APP_DIR/deploy/nginx.conf" /etc/nginx/sites-available/epstein-search
sudo ln -sf /etc/nginx/sites-available/epstein-search /etc/nginx/sites-enabled/epstein-search

# Test and reload
sudo nginx -t
sudo systemctl reload nginx
echo "  ✓ nginx configured"

# ─── 6. Configure systemd service ───────────────────────────────────────────

echo ""
echo "[6/6] Setting up systemd service..."

sudo cp "$APP_DIR/deploy/epstein-search.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable epstein-search
sudo systemctl start epstein-search

# Wait a moment then check status
sleep 2
if sudo systemctl is-active --quiet epstein-search; then
    echo "  ✓ Service running"
else
    echo "  ✗ Service failed to start. Check logs:"
    echo "    sudo journalctl -u epstein-search -n 20"
    exit 1
fi

# ─── Done ────────────────────────────────────────────────────────────────────

# Get the public IP
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "your-server-ip")

echo ""
echo "═══════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Web UI:  http://${PUBLIC_IP}"
echo ""
echo "  Management:"
echo "    esearch status       # check service status"
echo "    esearch logs         # view logs"
echo "    esearch restart      # restart service"
echo "    esearch search ...   # CLI search"
echo ""
echo "  SSL (optional):"
echo "    sudo apt install certbot python3-certbot-nginx"
echo "    sudo certbot --nginx -d your-domain.com"
echo ""
echo "  Environment variables (edit /opt/abovea.cloud/.env):"
echo "    OPENAI_API_KEY=sk-...    # for AI answers"
echo "    SITE_PASSWORD=...        # optional password gate"
echo "═══════════════════════════════════════════════"
