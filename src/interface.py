import time

from midware import Midware

if __name__ == "__main__":
    # connect to the drone
    drone = Midware(
        "/dev/ttyACM0", state_update_rate=4, setpoint_update_rate=2, flight_ceiling=5.0
    )
    time.sleep(1.0)

    _ = input("Press ENTER to arm")
    drone.vehicle.arm(True)
    time.sleep(10.0)
    drone.vehicle.disarm(True)

    # arm, takeoff, land
    # drone.preflight_setup()
    # drone.takeoff()
    # time.sleep(10.0)
    # drone.land()
