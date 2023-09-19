from __future__ import annotations

import time

import cv2
import torch
from prefetch_generator import prefetch
from wingman import gpuize


class Camera:
    def __init__(self, quantize_size):
        self.quantize_size = quantize_size

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

            image = self.crop_to_quantizable(
                self.normalize(
                    gpuize(
                        torch.tensor(
                            cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose((2, 0, 1)),
                            dtype=torch.float32,
                        ).unsqueeze(0),
                        device,
                    )
                ),
                self.quantize_size,
            )

            yield image

    def crop_to_quantizable(
        self, data: list[torch.Tensor] | torch.Tensor, quantize_size: int
    ) -> list[torch.Tensor] | torch.Tensor:
        """crop_to_quantizable.

        Args:
            data (list[torch.Tensor] | torch.Tensor): either a list of torch tensors or a single tensor, must be shape (:, H, W)
            quantize_size (int): quantize_size

        Returns:
            list[torch.Tensor] | torch.Tensor:
        """
        # grab the original image size
        base_size = data[0].shape if isinstance(data, list) else data.shape

        # crop to a size that is divisible by the quantization size
        crop_size = [
            base_size[-2] // quantize_size * quantize_size,
            base_size[-1] // quantize_size * quantize_size,
        ]

        # crop the things
        if isinstance(data, list):
            return [d[:, : crop_size[0], : crop_size[1]] for d in data]
        else:
            return data[:, : crop_size[0], : crop_size[1]]

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
