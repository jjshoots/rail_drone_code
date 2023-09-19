import threading
import time

import numpy as np
import zmq
from dronekit import VehicleMode, connect
from pymavlink import mavutil


class Vehicle:
    """Vehicle.

    The vehicle interface.
    It publishes the vehicle attitude on port 5555.
    It subscribes get vehicle setpoints by subscribing to port 5556.
    """

    def __init__(
        self,
        connection_string: str,
        state_update_rate: int,
        setpoint_update_rate: int,
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
        self.flight_ceiling = flight_ceiling

        # set to stabilized mode first, then get all parameters
        self.base_checks()

        """RUNTIME PARAMETERS"""
        # a flag on whether we are able to fly autonomously
        self.autonomous = False

        # parameters for state
        self._state_call_time = time.time()
        self._prev_ang_pos = np.zeros((3,), dtype=np.float32)
        self.ang_vel = np.zeros((3,), dtype=np.float32)
        self.lin_vel = np.zeros((3,), dtype=np.float32)

        # zmq last update
        self.last_zmq_update = 0.0

        """PYZMQ SOCKETS"""
        # attitude publisher
        self.attitude = dict()
        context = zmq.Context()
        self.att_pub = context.socket(zmq.PUB)
        self.att_pub.bind("tcp://127.0.0.1:5555")

        # setpoint subscriber
        context = zmq.Context()
        self.set_sub = context.socket(zmq.SUB)
        self.set_sub.connect("tcp://127.0.0.1:5556")
        self.set_sub.setsockopt_string(zmq.SUBSCRIBE, "")

        """START DAEMONS"""
        self._state_update_daemon()
        self._send_setpoint_daemon()
        self._zmq_update_watcher()

    def __del__(self):
        """__del__.
        """
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

    def preflight_setup(self) -> None:
        """Sets the drone to guided mode."""
        print("Preflight: Performing pre-arm checks...\n")

        while self.vehicle.mode.name == "INITIALISING":
            print("Preflight: Waiting for vehicle to initialise...")
            time.sleep(1)

        # preflight checks
        clear_time = 5
        while True:
            time.sleep(1)

            # check GPS sanity
            if not self.vehicle.gps_0.fix_type or self.vehicle.gps_0.fix_type < 4:
                print(f"Preflight: Waiting for GPS... {self.vehicle.gps_0}")
                clear_time = 5

            # check that vehicle is armed and ready to go
            if not self.vehicle.armed:
                print(
                    f"Preflight: Vehicle not armed, please arm manually. Waiting for arming..."
                )
                clear_time = 5

            if self.vehicle.mode.name != "GUIDED":
                print(
                    f"Preflight: Please set to GUIDED mode manually. Waiting for mode change..."
                )
                clear_time = 5

            # start counting down if all checks pass
            print("")
            clear_time -= 1
            if clear_time <= 1:
                print(f"Takeover in... {clear_time}")
            if clear_time == 0:
                break

        print("\nPreflight: Ready to rock and roll!")
        print("-----------------------------------------")

    def enable_autonomous(self, enable: bool) -> None:
        """Enables autonomous flight.

        Args:

        Returns:
            None:
        """
        self.autonomous = enable

    def takeoff(self, target_height: float = 1.5) -> None:
        """Sends the drone to a hover position.

        Args:
            height (float): height
        """
        if not self.vehicle.armed:
            print("Vehicle is not armed, unable to perform auto-takeoff! Disarming!")
            self.vehicle.armed = False
            return
        if self.vehicle.mode.name != "GUIDED":
            print(
                "Mode is not set to GUIDED, unable to perform auto-takeoff! Disarming!"
            )
            self.vehicle.armed = False
            return

        # run the takeoff command
        self.vehicle.simple_takeoff(target_height)

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
            if abs(np.mean(heights) - target_height) < abs(target_height) * 0.05:
                break

            # otherwise just wait
            time.sleep(1)

    def land(self) -> None:
        """land.

        Args:

        Returns:
            None:
        """
        # zero everything, disable autonomous mode
        self.update_velocity_setpoint(np.array([0.0, 0.0, 0.0, 0.0]))
        self.autonomous = False

        # send a land command
        self.vehicle.mode = VehicleMode("LAND")

        # check that the drone has indeed landed
        while True:
            height = self.vehicle.location.global_relative_frame.alt
            if height is not None and height < 0.5:
                break

            # otherwise just wait
            time.sleep(1)

    def update_velocity_setpoint(self, frdy: np.ndarray) -> None:
        """Sets a new velocity setpoint.

        Args:
            frdy (np.ndarray): frdy
        """
        setpoint = self.enu2ned(frdy)

        # check the flight ceiling, downward is positive
        vehicle_height = self.vehicle.location.global_relative_frame.alt
        if vehicle_height > self.flight_ceiling:
            setpoint[2] = min(vehicle_height - self.flight_ceiling, 1.0)

        self.setpoint_msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
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
        self.att_pub.send_pyobj(self.attitude)

        # queue the next call
        t = threading.Timer(self.state_update_period, self._state_update_daemon)
        t.daemon = True
        t.start()

    def _send_setpoint_daemon(self) -> None:
        """Sends setpoints in a separate loop for autonomous mode."""
        # update the setpoint if we have a message
        try:
            setpoint = self.set_sub.recv_pyobj(flags=zmq.NOBLOCK)
            self.last_zmq_update = time.time()
            self.update_velocity_setpoint(setpoint)
        except zmq.Again:
            pass

        # send the setpoint if autonomy is allowed
        if self.autonomous:
            self.vehicle.send_mavlink(self.setpoint_msg)

        # queue the next call
        t = threading.Timer(self.setpoint_update_period, self._state_update_daemon)
        t.daemon = True
        t.start()

    def _zmq_update_watcher(self) -> None:
        """A watchdog for the ZMQ updates. Resets the setpoint if stale."""
        stale_time = (time.time() - self.last_zmq_update)
        if stale_time > 3.0:
            self.update_velocity_setpoint(np.array([0.0, 0.0, 0.0, 0.0]))
            print(f"Setpoint update stale for {stale_time} seconds.")

        # queue the next call
        t = threading.Timer(1.0, self._zmq_update_watcher)
        t.daemon = True
        t.start()
