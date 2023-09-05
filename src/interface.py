import time

from midware import Midware

if __name__ == "__main__":
    # connect to the drone
    drone = Midware("/dev/ttyACM0", 4, 2)
    time.sleep(1.0)

    # arm, takeoff, land
    while True:
        drone.arm()
        drone.set_guided()
        drone.takeoff()
        time.sleep(10.0)
        drone.land()
