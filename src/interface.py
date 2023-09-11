import signal
import time

from midware import Midware

# handler for all the ctrl-c logic
terminate_hooks = []
def terminate_handler(*_):
    print("Crtl-C invoked!")
    for hook in terminate_hooks:
        hook()
    exit(1)

if __name__ == "__main__":
    # connect to the drone
    drone = Midware(
        "/dev/ttyACM0", state_update_rate=4, setpoint_update_rate=2, flight_ceiling=5.0
    )

    # handle terminate handlers
    terminate_hooks.append(drone.terminate)
    signal.signal(signal.SIGINT, terminate_handler)

    time.sleep(1.0)

    # arm, takeoff, land
    drone.preflight_setup()
    drone.takeoff()
    time.sleep(10.0)
    drone.land()
