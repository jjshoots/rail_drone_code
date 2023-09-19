from __future__ import annotations

import time

from midware.vehicle import Vehicle

if __name__ == "__main__":
    # connect to the drone
    drone = Vehicle(
        "/dev/ttyACM0",
        state_update_rate=4,
        setpoint_update_rate=2,
        flight_ceiling=5.0,
    )
    time.sleep(1.0)

    # arm, takeoff
    drone.preflight_setup()
    drone.takeoff()

    # count down
    for i in range(5):
        print(f"Going autonomous in {5 - i}...")
        time.sleep(1.0)

    # enable autonomous mode
    drone.enable_autonomous(True)

    # now we wait
    while True:
        time.sleep(100)

    drone.land()
