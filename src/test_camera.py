"""Standalone script for testing the camera."""

import cv2
from wingman import Wingman

from midware.camera import Camera

if __name__ == "__main__":
    wm = Wingman(config_yaml="src/settings.yaml")
    camera = Camera(wm.cfg.base_resize)
    image = camera.get_image(device=wm.device)
    print(image.shape)
