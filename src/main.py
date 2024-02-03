"""
Main file for autonomous flights.
"""

from __future__ import annotations

import time

from agent import Agent
from midware.vehicle import Vehicle

if __name__ == "__main__":
    # connect to the drone
    drone = Vehicle(
        "/dev/ttyACM0",
        state_update_rate=4,
        setpoint_update_rate=2,
        flight_floor=0.75,
        flight_ceiling=4.0,
    )

    # initialize the agent
    agent = Agent()
    agent.setup()

    # start sending setpoints
    while True:
        agent.set_attitude(drone.attitude)
        drone.set_setpoint(agent.get_setpoint())
        time.sleep(0.3)
