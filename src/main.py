"""
Middleware for all autonomous flights.
This script is not meant to be run standalone and will fail if there are no setpoints.
"""

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

    # arm, takeoff
    drone.preflight_setup()
    drone.takeoff()

    # enable autonomous mode
    while input("Start autonomous? (y/n)") != "y":
        time.sleep(1)
    drone.enable_autonomous(True)

    # now we wait
    print("Press any key to terminate...")
    try:
        while True:
            time.sleep(100)
    except KeyboardInterrupt:
        pass

    drone.land()
