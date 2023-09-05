import time

from midware import Midware

drone = Midware("/dev/ttyACM0", 5)

time.sleep(1.0)

while True:
    drone.arm()
    drone.set_guided()
    drone.takeoff()
