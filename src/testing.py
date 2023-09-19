import time

import cv2
from wingman import Wingman

from camera import Camera
from midware import Midware

if __name__ == "__main__":
    wm = Wingman(config_yaml="src/settings.yaml")

    if False:
        # connect to the drone
        drone = Midware(
            "/dev/ttyACM0",
            state_update_rate=4,
            setpoint_update_rate=2,
            flight_ceiling=5.0,
        )
        time.sleep(1.0)

        # arm, takeoff, land
        drone.preflight_setup()
        drone.takeoff()
        time.sleep(10.0)
        drone.land()

    if False:
        camera = Camera(wm.cfg.base_resize)
        for image in camera.stream(wm.device):
            cv2.imshow("something", image)
            # cv2.imshow("something", obs["rgba_img"])
            cv2.waitKey(1)
