import time

from midware import Midware

midware = Midware("/dev/ttyACM0", 5)

time.sleep(1.0)

while True:
    time.sleep(0.5)
    print(midware.lin_vel, midware.ang_pos[-1])
