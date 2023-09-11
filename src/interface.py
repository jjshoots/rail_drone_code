import signal
import time

from midware import Midware

if __name__ == "__main__":
    # connect to the drone
    drone = Midware(
        "/dev/ttyACM0", state_update_rate=4, setpoint_update_rate=2, flight_ceiling=5.0
    )
    signal.signal(signal.SIGINT, drone.terminate)
    time.sleep(1.0)

    # arm, takeoff, land
    drone.preflight_setup()
    drone.takeoff()
    time.sleep(10.0)
    drone.land()
