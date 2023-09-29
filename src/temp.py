import time
import dronekit
from pymavlink import mavutil


vehicle = dronekit.connect("/dev/ttyACM0", wait_ready=True)
vehicle.airspeed=3
print("arming")
vehicle.armed = True
print("guiding")
vehicle.mode = dronekit.VehicleMode("GUIDED")
vehicle.simple_takeoff(2.0)
vehicle.wait_for_alt(2.0)
msg = vehicle.message_factory.set_position_target_local_ned_encode(
    0,       # time_boot_ms (not used)
    0, 0,    # target system, target component
    mavutil.mavlink.MAV_FRAME_LOCAL_NED, # frame
    0b110111000111, # type_mask (only speeds enabled)
    0, 0, 5, # x, y, z positions (not used)
    0, 0, 5, # x, y, z velocity in m/s
    0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
    0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)


# send command to vehicle on 1 Hz cycle
while True:
    print("commanding")
    vehicle.send_mavlink(msg)
    time.sleep(1)
