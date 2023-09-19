import time

import cv2
from wingman import Wingman

from midware.camera import Camera
from midware.vehicle import Vehicle

if __name__ == "__main__":

    if True:
        # connect to the drone
        drone = Vehicle(
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
        wm = Wingman(config_yaml="src/settings.yaml")
        camera = Camera(wm.cfg.base_resize)
        for image in camera.stream(wm.device):
            cv2.imshow("something", image)
            # cv2.imshow("something", obs["rgba_img"])
            cv2.waitKey(1)
