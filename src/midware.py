import threading
import time

import numpy as np
from dronekit import VehicleMode, connect


class Midware:
    """Midware."""

    def __init__(self, connection_string: str, update_rate: int) -> None:
        """__init__.

        Args:
            connection_string (str): connection_string
            update_rate (int): update_rate

        Returns:
            None:
        """
        self.vehicle = connect(connection_string, wait_ready=True)

        # constants
        self.update_time = 1.0 / update_rate
        self.snap_limit = np.pi / self.update_time

        # get all parameters
        self.display_base_params()

        # runtime parameters
        self._state_call_time = time.time()
        self._prev_ang_pos = np.zeros((3,), dtype=np.float32)
        self.deg_to_rad = np.pi / 180.0

        # start the state watchdog and its runtime variables
        self.ang_vel = np.zeros((3,), dtype=np.float32)
        self.lin_vel = np.zeros((3,), dtype=np.float32)
        self._compute_state()

        #Create a message listener using the decorator.
        @self.vehicle.on_message("CONTROL_SYSTEM_STATE")
        def listener(self, name, message):
            print(message)

    def display_base_params(self):
        """Displays all the base params, usually called on init."""
        # print(f"Autopilot capabilities (supports ftp): {self.vehicle.capabilities.ftp}")
        # print(f"Global Location: {self.vehicle.location.global_frame}")
        # print(f"Global Location (relative altitude): {self.vehicle.location.global_relative_frame}")
        # print(f"Local Location: {self.vehicle.location.local_frame}")
        # print(f"Attitude: {self.vehicle.attitude}")
        # print(f"Velocity: {self.vehicle.velocity}")
        # print(f"GPS: {self.vehicle.gps_0}")
        # print(f"Gimbal status: {self.vehicle.gimbal}")
        # print(f"Heading: {self.vehicle.heading}")
        print("-----------------------------------------")
        print(f"Autopilot Firmware version: {self.vehicle.version}")
        print(f"Groundspeed: {self.vehicle.groundspeed}")
        print(f"Airspeed: {self.vehicle.airspeed}")
        print(f"{self.vehicle.battery}")
        print(f"EKF OK?: {self.vehicle.ekf_ok}")
        print(f"Last Heartbeat: {self.vehicle.last_heartbeat}")
        print(f"Rangefinder: {self.vehicle.rangefinder}")
        print(f"Rangefinder distance: {self.vehicle.rangefinder.distance}")
        print(f"Rangefinder voltage: {self.vehicle.rangefinder.voltage}")
        print(f"Is Armable?: {self.vehicle.is_armable}")
        print(f"System status: {self.vehicle.system_status.state}")
        print(f"Mode: {self.vehicle.mode.name}")
        print(f"Armed: {self.vehicle.armed}")
        print("-----------------------------------------")

    def set_guided(self):
        """Sets the drone to guided mode."""
        print("-----------------------------------------")
        print("Performing pre-arm checks...")

        while self.vehicle.mode.name == "INITIALISING":
            print("Waiting for vehicle to initialise...")
            time.sleep(1)

        while self.vehicle.gps_0.fix_type and self.vehicle.gps_0.fix_type < 2:
            print(f"Waiting for GPS...:, {self.vehicle.gps_0.fix_type}")
            time.sleep(1)

        print(f"Setting to guided mode...")
        self.vehicle.mode = VehicleMode("GUIDED")
        print(f"Mode set to {self.vehicle.mode}")

        print("Pre-arm checks done, guided mode set")
        print("-----------------------------------------")

    def arm(self) -> bool:
        """Arms the quad.

        Args:

        Returns:
            bool:
        """
        if not self.vehicle.is_armable:
            print("Vehicle not armable!")
            return False

        print("Arming motors!")
        self.vehicle.mode = VehicleMode("GUIDED")
        self.vehicle.armed = True

        while not self.vehicle.armed:
            print(" Waiting for arming...")
            time.sleep(1)

        return True

    def _compute_state(self):
        """Computes the current state of the drone."""
        # record the current time
        elapsed = time.time() - self._state_call_time
        self._state_call_time = time.time()

        # read altitude
        self.altitude = self.vehicle.location.global_relative_frame.alt

        # for angular position, we need to watch for rollover
        self.ang_pos = np.array(
            [
                self.vehicle.attitude.roll,
                -self.vehicle.attitude.pitch,
                -self.vehicle.attitude.yaw,
            ]
        )
        self.ang_pos[(self.ang_pos - self._prev_ang_pos) > self.snap_limit] -= (
            np.pi * 2.0 / elapsed
        )
        self.ang_pos[(self.ang_pos - self._prev_ang_pos) < -self.snap_limit] += (
            np.pi * 2.0 / elapsed
        )

        # compute angular velocity using differences
        temp_ang_vel = (self.ang_pos - self._prev_ang_pos) / elapsed

        # if the angular velocity has not been updates, don't use it
        if np.linalg.norm(temp_ang_vel) != 0.0:
            self.ang_vel = temp_ang_vel

        # store variables for next timestep
        self._prev_ang_pos = self.ang_pos.copy()

        # read linear velocity and rotate to body frame
        gps_vel = np.array(self.vehicle.velocity)
        c, s = np.cos(self.ang_pos[-1]), np.sin(self.ang_pos[-1])
        self.lin_vel[0] = gps_vel[0] * c - gps_vel[1] * s
        self.lin_vel[1] = -gps_vel[0] * s - gps_vel[1] * c
        self.lin_vel[2] = -gps_vel[2]

        # queue the next call
        threading.Timer(self.update_time, self._compute_state).start()
