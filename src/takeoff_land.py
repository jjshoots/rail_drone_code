"""Standalone script for testing takeoff and land functionality."""

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

    # check for land requirement
    while input("Land? (y/n)") != "y":
        time.sleep(1)
    drone.land()
