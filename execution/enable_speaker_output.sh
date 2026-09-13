#!/bin/bash
# Script to switch ReSpeaker 2-Mics Pi HAT output from 3.5mm Headphone Jack to JST 2.0 Speaker Port

echo "========================================================"
echo "Configuring ReSpeaker 2-Mics Pi HAT: JST Speaker Output"
echo "========================================================"

# Find soundcard index or name
CARD_NAME="seeed2micvoicec"

# Check if card exists
if ! aplay -l | grep -qi "seeed"; then
    echo "⚠️ Warning: seeed-voicecard not found in 'aplay -l'."
    echo "Checking available cards..."
    aplay -l
    CARD_NAME="0"
fi

echo "Unmuting and setting Speaker volume on card: $CARD_NAME..."

# Enable Speaker and PCM master digital output
amixer -c "$CARD_NAME" sset 'PCM' 100% 2>/dev/null || amixer sset 'PCM' 100% 2>/dev/null
amixer -c "$CARD_NAME" sset 'Speaker' 100% unmute 2>/dev/null || amixer sset 'Speaker' 100% unmute 2>/dev/null

# Ensure DAC routing to speaker mixers is active on WM8960
amixer -c "$CARD_NAME" sset 'Left Speaker Mixer Left DAC' on 2>/dev/null
amixer -c "$CARD_NAME" sset 'Right Speaker Mixer Right DAC' on 2>/dev/null
amixer -c "$CARD_NAME" sset 'Left Output Mixer PCM' on 2>/dev/null
amixer -c "$CARD_NAME" sset 'Right Output Mixer PCM' on 2>/dev/null

# Optional: Mute headphone jack so only speaker plays
# amixer -c "$CARD_NAME" sset 'Headphone' 0% mute 2>/dev/null

# Save state permanently across reboots
echo "Saving ALSA mixer state permanently..."
sudo alsactl store

echo "✅ JST Speaker Output enabled!"
echo "Testing speaker output now (Ctrl+C to stop)..."
speaker-test -t wav -c 2 -l 1
