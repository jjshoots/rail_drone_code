import threading
import time

import numpy as np
from dronekit import VehicleMode, connect
from pymavlink import mavutil


class Vehicle:
    """Vehicle.

    The vehicle interface.
    """

    def __init__(
        self,
        connection_string: str,
        state_update_rate: int,
        setpoint_update_rate: int,
        flight_floor: float,
        flight_ceiling: float,
    ) -> None:
        """__init__.

        Args:
            connection_string (str): connection_string
            state_update_rate (int): state_update_rate
            setpoint_update_rate (int): setpoint_update_rate
            flight_ceiling (float): flight ceiling in meters above takeoff point

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

        # flight ceiling
        self.flight_floor = flight_floor
        self.flight_ceiling = flight_ceiling

        # set to stabilized mode first, then get all parameters
        self.base_checks()
        self.set_setpoint(np.array([0.0, 0.0, 0.0, 0.0]))

        """RUNTIME PARAMETERS"""
        self._last_setpoint_time = time.time()
        self._state_call_time = time.time()
        self._prev_ang_pos = np.zeros((3,), dtype=np.float32)
        self.ang_vel = np.zeros((3,), dtype=np.float32)
        self.lin_vel = np.zeros((3,), dtype=np.float32)
        self.attitude = dict()

        """START DAEMONS"""
        self._state_update_daemon()
        self._send_setpoint_daemon()
        self._stale_setpoint_watcher()

        # some sleep helps
        time.sleep(1)

    def __del__(self):
        """__del__."""
        self.vehicle.close()

    def base_checks(self) -> None:
        """Displays all the base params, usually called on init."""

        if self.vehicle.mode.name != "STABILIZE":
            print(f"Vehicle mode is {self.vehicle.mode.name}, setting to STABILIZE...")
            self.vehicle.mode = VehicleMode("STABILIZE")

        # print(f"Autopilot capabilities (supports ftp): {self.vehicle.capabilities.ftp}")
        # print(f"Global Location: {self.vehicle.location.global_frame}")
        # print(f"Global Location (relative altitude): {self.vehicle.location.global_relative_frame}")
        # print(f"Local Location: {self.vehicle.location.local_frame}")
        # print(f"Gimbal status: {self.vehicle.gimbal}")
        # print(f"Heading: {self.vehicle.heading}")
        print("-----------------------------------------")
        print(f"Autopilot Firmware version: {self.vehicle.version}")
        print(f"Groundspeed: {self.vehicle.groundspeed}")
        print(f"Velocity: {self.vehicle.velocity}")
        print(f"Airspeed: {self.vehicle.airspeed}")
        print(f"GPS: {self.vehicle.gps_0}")
        print(f"Attitude: {self.vehicle.attitude}")
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

    def set_setpoint(self, frdy: np.ndarray) -> None:
        """Sets a new velocity setpoint.

        Args:
            frdy (np.ndarray): frdy
        """
        self._last_setpoint_time = time.time()
        setpoint = self.enu2ned(frdy)

        # altitude check
        # height upward is positive, setpoint downward is positive
        vehicle_height = self.vehicle.location.global_relative_frame.alt
        if vehicle_height > self.flight_ceiling:
            setpoint[2] = min(vehicle_height - self.flight_ceiling, 1.0)
        elif vehicle_height < self.flight_floor:
            setpoint[2] = max(vehicle_height - self.flight_floor, -1.0)

        self.setpoint_msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            # time_boot_ms (not used)
            0,
            # target system, target component
            0,
            0,
            # frame
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            # type_mask, addressed in reversed, and used to indicate which components should be IGNORED
            # bit1:PosX, bit2:PosY, bit3:PosZ, bit4:VelX, bit5:VelY, bit6:VelZ, bit7:AccX, bit8:AccY, bit9:AccZ, bit11:yaw, bit12:yaw rate
            0b010111000111,
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

    def _state_update_daemon(self) -> None:
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

        # read linear velocity and rotate to body frame in ENU coordinates
        gps_vel = np.array(self.vehicle.velocity)
        c, s = np.cos(self.ang_pos[-1]), np.sin(self.ang_pos[-1])
        self.lin_vel[0] = gps_vel[0] * c - gps_vel[1] * s
        self.lin_vel[1] = -gps_vel[0] * s - gps_vel[1] * c
        self.lin_vel[2] = -gps_vel[2]

        # publish the attitude to zmq
        self.attitude["lin_vel"] = self.lin_vel
        self.attitude["ang_vel"] = self.lin_vel
        self.attitude["altitude"] = self.altitude

        # queue the next call
        t = threading.Timer(self.state_update_period, self._state_update_daemon)
        t.daemon = True
        t.start()

    def _send_setpoint_daemon(self) -> None:
        """Sends setpoints in a separate loop for autonomous mode."""
        # send the setpoint if autonomy is allowed
        self.vehicle.send_mavlink(self.setpoint_msg)
        print(self.setpoint_msg)

        # queue the next call
        t = threading.Timer(self.setpoint_update_period, self._send_setpoint_daemon)
        t.daemon = True
        t.start()

    def _stale_setpoint_watcher(self) -> None:
        """Watches for stale setpoints and whether to reset."""
        stale_time = time.time() - self._last_setpoint_time
        if stale_time > 3.0:
            self.set_setpoint(np.array([0.0, 0.0, 0.0, 0.0]))
            print(f"Setpoint update stale for {stale_time} seconds.")

        # queue the next call
        t = threading.Timer(1.0, self._stale_setpoint_watcher)
        t.daemon = True
        t.start()
