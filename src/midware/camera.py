from __future__ import annotations

import time
from typing import Generator

import cv2
import numpy as np
import torch
from prefetch_generator import prefetch
from wingman import gpuize


class Camera:
    def __init__(self, base_resize):
        self.base_resize = base_resize

        # camera initialization
        while True:
            # try to grab the camera
            self.camera = cv2.VideoCapture(-1)
            _, image = self.camera.read()

            # if has image, just break, otherwise we need to retry
            if image is None:
                print(
                    "Camera not detected. Is it setup correctly? Retrying in 5 seconds..."
                )
                time.sleep(5)
            else:
                break

        print("Camera Initialized")

    def get_image(self, device) -> torch.Tensor | np.ndarray:
        _, image = self.camera.read()

        # perform opencv formatting first
        image = cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            [self.base_resize[1], self.base_resize[0]],
        )

        # convert to torch understandable
        image = self.normalize(
            gpuize(
                torch.tensor(
                    image.transpose((2, 0, 1)),
                    dtype=torch.float32,
                ).unsqueeze(0),
                device,
            )
        )
        return image

    @staticmethod
    def normalize(data) -> np.ndarray | torch.Tensor:
        """normalize.

        Args:
            data:
        """
        data = (data.float() - 128.0) / 128.0
        return data

    @staticmethod
    def denormalize(data) -> np.ndarray | torch.Tensor:
        """denormalize.

        Args:
            data:
        """
        if torch.is_tensor(data):
            data = ((data * 128.0) + 128.0).int()
        else:
            data = ((data * 128.0) + 128.0).astype(np.uint8)
        return data
