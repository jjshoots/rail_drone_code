"""
Supervisor level script for sending setpoints to fly the drone in a 5m circle at an altitude of 2m.
This script requires `main.py` to first be run in another process.
"""

from __future__ import annotations

import time

import numpy as np
import zmq

if __name__ == "__main__":
    # setpoint publisher
    context = zmq.Context()
    set_pub = context.socket(zmq.PUB)
    set_pub.bind("tcp://127.0.0.1:5556")

    # attitude subscriber
    context = zmq.Context()
    att_sub = context.socket(zmq.SUB)
    att_sub.connect("tcp://127.0.0.1:5555")
    att_sub.setsockopt_string(zmq.SUBSCRIBE, "")

    while True:
        # read the vehicle attitude, this is mainly for altitude data
        altitude = att_sub.recv_pyobj(flags=zmq.NOBLOCK)["altitude"]

        # setpoint is frdy
        # this flies the drone in a clockwise circle
        # with a diameter 5 meters wide
        # with a forward velocity of 1 m/s
        # at a height of 2 meters
        climb_rate = np.clip(2.0 - altitude, a_min=-1.0, a_max=1.0)
        setpoint = np.array([1.0, 0.0, -climb_rate, 0.2])

        # publish the setpoint
        set_pub.send_pyobj(setpoint)

        time.sleep(1)
