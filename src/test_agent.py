"""
Supervisor level script for sending setpoints generated from the CVRL model.
This script requires `main.py` to first be run in another process.
"""

from __future__ import annotations

import os
from typing import Generator

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from wingman import cpuize, gpuize

from agent import Agent
from midware import Camera


class TestAgent(Agent):
    """The CVRL agent for local testing without camera."""

    def __init__(self):
        """__init__."""
        super().__init__()
        self._setup_nets()

    def read_images(self) -> Generator[torch.Tensor | np.ndarray, None, None]:
        file_path = os.path.dirname(os.path.realpath(__file__))
        images_path = os.path.join(
            file_path, "../../active_learning/datasets/RailwayCVRLV0/train/images"
        )
        for file in os.listdir(images_path):
            # have to read np array and apply preprocessors
            image = cv2.imread(os.path.join(images_path, file))
            image = cv2.resize(image, self.cfg.base_resize)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 0)

            # fix dimensions and normalize
            image = torch.tensor(
                image.transpose((2, 0, 1)), dtype=torch.float32, device=self.cfg.device
            )
            image = Camera.normalize(image)
            image = image.unsqueeze(0)

            yield image
            yield torch.flip(image, dims=[-1])

    def start(self) -> None:
        """The main loop."""
        # start loop, just go as fast as possible for now
        for image in self.read_images():
            # THIS FUNCTION IS COPIED OVER FROM AGENT.py
            # get the camera image and pass to segmentation model, already on gpu
            # mean filter the segmap to remove spurious predictions
            seg_map, _ = self.cv_model(image)
            seg_map = (seg_map.float() == 1.0).float()

            # pass segmap to the rl model to get action, send to cpu
            action = self.rl_model.infer(*self.rl_model(seg_map)).squeeze(0)
            action = cpuize(action)

            # map action, [stop/go, yaw_rate] -> [frdy]
            # linear velocity in the sim is about 1.5
            # yaw rate in the sim is about 1.5
            setpoint = np.array([1.0, 0.0, 0, action[0]])

            print(f"FRDY: {setpoint}")

            # cv2.imshow("something", cpuize(((image + 1.0) / 2.0).squeeze()).transpose(1, 2, 0))
            plt.imshow(cpuize(seg_map.squeeze()))
            plt.show()


if __name__ == "__main__":
    agent = TestAgent()
    agent.start()
