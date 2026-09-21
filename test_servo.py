import time
import board
import busio
from adafruit_pca9685 import PCA9685

print('Initializing I2C Bus 3...')
i2c = busio.I2C(board.D5, board.D4)

print('Connecting to PCA9685...')
pca = PCA9685(i2c, address=0x40)
pca.frequency = 50

def set_angle(channel, angle_deg):
    norm = (angle_deg + 90.0) / 180.0
    pulse_us = 500 + norm * 2000
    period_us = 1_000_000.0 / 50
    duty_cycle = int((pulse_us / period_us) * 65535.0)
    pca.channels[channel].duty_cycle = max(0, min(65535, duty_cycle))
    print(f'Channel {channel} -> {angle_deg} degrees')

try:
    print('Testing Servo on Channel 0 (Press Ctrl+C to stop)...')
    while True:
        set_angle(0, 0)
        time.sleep(1)
        set_angle(0, 45)
        time.sleep(1)
        set_angle(0, 0)
        time.sleep(1)
        set_angle(0, -45)
        time.sleep(1)
except KeyboardInterrupt:
    print('
Stopping...')
    pca.channels[0].duty_cycle = 0
    pca.deinit()
    print('Done.')
