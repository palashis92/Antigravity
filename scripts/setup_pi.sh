#!/bin/bash
# ==============================================================================
# LUMI AI Companion Robot - Raspberry Pi 5 Environment & Hardware Setup Script
# ==============================================================================

set -e

# Detect project directory (where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ACTUAL_USER="${SUDO_USER:-$USER}"
ACTUAL_HOME="$(eval echo ~$ACTUAL_USER)"

echo "=========================================================="
echo "Starting LUMI Robot Setup on Raspberry Pi 5"
echo "  Project: $PROJECT_DIR"
echo "  User:    $ACTUAL_USER ($ACTUAL_HOME)"
echo "=========================================================="

# 1. Update OS packages
echo "[1/7] Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# 2. Install essential system dependencies and hardware tools
echo "[2/7] Installing system packages (hardware drivers, audio, swig, OpenCV deps)..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-picamera2 \
    python3-opencv \
    i2c-tools \
    libasound2-dev \
    alsa-utils \
    sox \
    libsox-fmt-all \
    stockfish \
    swig \
    cmake \
    libopenblas-dev \
    git

# 3. Configure Hardware Interfaces (/boot/firmware/config.txt)
echo "[3/7] Configuring I2C, SPI, and MAX98357A I2S audio overlays..."
CONFIG_FILE="/boot/firmware/config.txt"
if [ ! -f "$CONFIG_FILE" ]; then
    CONFIG_FILE="/boot/config.txt"
fi

# Enable I2C & SPI
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0

# Check and append MAX98357A I2S DAC overlay if not present
if ! grep -q "dtoverlay=max98357a" "$CONFIG_FILE"; then
    echo "Appending MAX98357A I2S DAC overlay to $CONFIG_FILE..."
    echo "# LUMI MAX98357A I2S Audio" | sudo tee -a "$CONFIG_FILE"
    echo "dtoverlay=max98357a" | sudo tee -a "$CONFIG_FILE"
fi

# 4. Create Python Virtual Environment inside the project directory
echo "[4/7] Setting up Python virtual environment..."
VENV_DIR="$PROJECT_DIR/venv"
python3 -m venv "$VENV_DIR" --system-site-packages
source "$VENV_DIR/bin/activate"

# 5. Install Python Dependencies
echo "[5/7] Installing Python packages..."
pip install --upgrade pip

# Core packages
pip install \
    pyyaml \
    pydantic \
    pillow \
    numpy \
    openai \
    python-chess \
    duckduckgo-search \
    gTTS \
    reportlab \
    pytest

# Hardware packages (Raspberry Pi specific)
pip install \
    adafruit-circuitpython-pca9685 \
    adafruit-circuitpython-servokit \
    spidev \
    RPi.GPIO \
    smbus2

# Optional: face_recognition (heavy, may take 10-15 mins to compile on Pi)
echo "[5b/7] Installing face_recognition (this may take a while on Pi)..."
pip install face_recognition || echo "WARNING: face_recognition install failed. Face ID will use detection-only mode."

# 6. Set correct ownership
echo "[6/7] Setting file permissions..."
sudo chown -R "$ACTUAL_USER:$ACTUAL_USER" "$PROJECT_DIR"

echo "=========================================================="
echo "LUMI environment setup complete!"
echo ""
echo "  venv:    $VENV_DIR"
echo ""
echo "Next steps:"
echo "  1. Create .env file:  nano $PROJECT_DIR/.env"
echo "     Add: GEMINI_API_KEY=your-key"
echo "     Add: MEM0_API_KEY=your-key  (optional)"
echo ""
echo "  2. Run manually:  cd $PROJECT_DIR && source venv/bin/activate && python3 -m lumi.main"
echo "=========================================================="
