import threading
import time

import numpy as np
from dronekit import VehicleMode, connect
from pymavlink import mavutil


class Midware:
    """Midware."""

    def __init__(
        self, connection_string: str, state_update_rate: int, setpoint_update_rate: int
    ) -> None:
        """__init__.

        Args:
            connection_string (str): connection_string
            state_update_rate (int): state_update_rate
            setpoint_update_rate (int): setpoint_update_rate

        Returns:
            None:
        """
        self.vehicle = connect(connection_string, wait_ready=True)

        """CONSTANTS"""
        # rates
        self.state_update_period = 1.0 / state_update_rate
        self.setpoint_update_period = 1.0 / setpoint_update_rate

        # conversions from enu to ned and back
        self._frame_conversion = np.array([1.0, -1.0, -1.0, -1.0])
        self.enu2ned = lambda x: x * self._frame_conversion
        self.ned2enu = lambda x: x * self._frame_conversion

        # parameters for the state updates
        self._snap_limit = np.pi / self.state_update_period
        self._deg_to_rad = np.pi / 180.0

        # get all parameters
        self.display_base_params()

        """RUNTIME PARAMETERS"""
        # a flag on whether we are able to fly autonomously
        self.autonomous_enable = False

        # parameters for state
        self._state_call_time = time.time()
        self._prev_ang_pos = np.zeros((3,), dtype=np.float32)
        self.ang_vel = np.zeros((3,), dtype=np.float32)
        self.lin_vel = np.zeros((3,), dtype=np.float32)

        """START DAEMONS"""
        self._state_update_daemon()
        self._send_setpoint_daemon()

        def __del__(self):
            self.vehicle.close()

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
        self.vehicle.armed = True

        while not self.vehicle.armed:
            print(" Waiting for arming...")
            time.sleep(1)

        return True

    def set_guided(self):
        """Sets the drone to guided mode."""
        print("-----------------------------------------")
        print("Performing pre-arm checks...")

        while self.vehicle.mode.name == "INITIALISING":
            print("Waiting for vehicle to initialise...")
            time.sleep(1)

        while self.vehicle.gps_0.fix_type and self.vehicle.gps_0.fix_type < 2:
            print(f"Waiting for GPS...: {self.vehicle.gps_0.fix_type}")
            time.sleep(1)

        print(f"Setting to guided mode...")
        self.vehicle.mode = VehicleMode("GUIDED")
        print(f"Mode set to {self.vehicle.mode}")

        print("Pre-arm checks done, guided mode set")
        print("-----------------------------------------")

        return True

    def takeoff(self, height: float = 1.5):
        """Sends the drone to a hover position.

        Args:
            height (float): height
        """
        if not self.vehicle.armed:
            print("Vehicle is not armed, unable to perform auto-takeoff!")
            return False
        if self.vehicle.mode.name != "GUIDED":
            print("Mode is not set to GUIDED, unable to perform auto-takeoff!")
            return False

        # run the takeoff command
        self.vehicle.simple_takeoff(height)

        # check that the drone has actually reached a stable hover
        # we must maintain a stable hover for 3 seconds before we move on
        step_count = 0
        heights = np.zeros((3,), dtype=np.float32)
        while True:
            # treat the numpy array as a deque of sorts
            heights[
                step_count % heights.shape[0]
            ] = self.vehicle.location.global_relative_frame.alt
            step_count += 1

            # if we've maintained with 5% of the target height for 3 seconds, break
            if abs(np.mean(heights) - height) < abs(heights) * 0.05:
                break

            # otherwise just wait
            time.sleep(1)

        # flag that we can now do autonomous flight
        self.autonomous_enable = True

    def land(self):
        # zero everything, disable autonomous mode
        self.set_velocity_setpoint(np.array([0.0, 0.0, 0.0, 0.0]))
        self.autonomous_enable = False

        # send a land command
        self.vehicle.mode = VehicleMode("LAND")

        # check that the drone has indeed landed
        while True:
            height = self.vehicle.location.global_relative_frame.alt
            if height is not None and height < 0.5:
                break

            # otherwise just wait
            time.sleep(1)

    def set_velocity_setpoint(self, frdy: np.ndarray):
        """Sets a new velocity setpoint.

        Args:
            frdy (np.ndarray): frdy
        """
        setpoint = self.enu2ned(frdy)
        self.setpoint_msg = self.vehicle.message_factory.send_ned_velocity(
            # time_boot_ms (not used)
            0,
            # target system, target component
            0,
            0,
            # frame
            mavutil.mavlink.MAV_FRAME_BODY_FRD,
            # type_mask, addressed in reversed, and used to indicate which components should be IGNORED
            # bit1:PosX, bit2:PosY, bit3:PosZ, bit4:VelX, bit5:VelY, bit6:VelZ, bit7:AccX, bit8:AccY, bit9:AccZ, bit11:yaw, bit12:yaw rate
            0b01111000111,
            # x, y, z positions (not used)
            0,
            0,
            0,
            # x, y, z velocity in m/s
            setpoint[0],
            setpoint[1],
            setpoint[2],
            # x, y, z acceleration (not used)
            0,
            0,
            0,
            # yaw, yaw_rate (only yaw rate used)
            0,
            setpoint[3],
        )

    def _state_update_daemon(self):
        """Updates state in a separate loop."""
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
        self.ang_pos[(self.ang_pos - self._prev_ang_pos) > self._snap_limit] -= (
            np.pi * 2.0 / elapsed
        )
        self.ang_pos[(self.ang_pos - self._prev_ang_pos) < -self._snap_limit] += (
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
        threading.Timer(self.state_update_period, self._state_update_daemon).start()

    def _send_setpoint_daemon(self):
        """Sends setpoints in a separate loop for autonomous mode."""
        # send the setpoint if autonomy is allowed
        if self.autonomous_enable:
            self.vehicle.send_mavlink(self.setpoint_msg)

        # queue the next call
        threading.Timer(self.setpoint_update_period, self._state_update_daemon).start()
