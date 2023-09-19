from __future__ import annotations

import time

import cv2
import torch
from prefetch_generator import prefetch
from wingman import gpuize


class Camera:
    def __init__(self, base_resize):
        self.base_resize = base_resize

    @prefetch(max_prefetch=2)
    def stream(self, device):
        camera = cv2.VideoCapture(-1)
        print("Camera Initialized")

        while True:
            # get a camera frame
            _, image = camera.read()

            if image is None:
                print(
                    "Camera not detected. Is it setup correctly? Retrying in 5 seconds..."
                )
                time.sleep(5)
                camera = cv2.VideoCapture(-1)
                continue

            # perform opencv formatting first
            image = cv2.flip(
                cv2.resize(
                    image,
                    [self.base_resize[1], self.base_resize[0]],
                ),
                0,
            )

            # convert to torch understandable
            # image = self.normalize(
            #     gpuize(
            #         torch.tensor(
            #             image.transpose((2, 0, 1)),
            #             dtype=torch.float32,
            #         ).unsqueeze(0),
            #         device,
            #     )
            # )

            yield image

    def normalize(self, data):
        """normalize.

        Args:
            data:
        """
        data = (data.float() - 128.0) / 128.0
        return data

    def denormalize(self, data):
        """denormalize.

        Args:
            data:
        """
        if torch.is_tensor(data):
            data = ((data * 128.0) + 128.0).int()
        else:
            data = ((data * 128.0) + 128.0).astype(np.uint8)
        return data
