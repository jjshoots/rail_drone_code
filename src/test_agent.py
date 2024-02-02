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
from wingman import Wingman, cpuize, gpuize

from cv import EnsembleAttUNet
from midware import Camera
from rl import GaussianActor


class Agent:
    """The CVRL agent.

    This script hosts the CV and RL models.
    It continuously reads the camera image, and outputs the action.
    """

    def __init__(
        self,
    ):
        """__init__."""
        self.wm = Wingman(config_yaml="src/settings.yaml")
        self.cfg = self.wm.cfg

        """MODELS AND CAMERA"""
        self.cv_model, self.rl_model = self._setup_nets()

        """RUNTIME PARAMETERS"""
        self.setpoint = torch.zeros((4,), dtype=torch.float32, device=self.cfg.device)
        self.obs_att = torch.zeros((1, 8), dtype=torch.float32, device=self.cfg.device)
        self.action = torch.zeros((4,), dtype=torch.float32, device=self.cfg.device)

    def _setup_nets(self) -> tuple[EnsembleAttUNet, GaussianActor]:
        """_setup_nets.

        Args:

        Returns:
            tuple[EnsembleAttUNet, GaussianActor]:
        """
        # cfg up networks
        cv_model = EnsembleAttUNet(
            in_channels=self.cfg.in_channels,
            out_channels=self.cfg.num_labels,
            inner_channels=self.cfg.inner_channels,
            att_num_heads=self.cfg.att_num_heads,
            num_att_module=self.cfg.num_att_module,
            num_ensemble=self.cfg.num_ensemble,
        ).to(self.cfg.device)

        rl_model = GaussianActor(
            act_size=self.cfg.act_size,
            obs_size=self.cfg.obs_size,
        )

        # get weights for CV model
        file_path = os.path.dirname(os.path.realpath(__file__))
        cv_model.load_state_dict(
            torch.load(os.path.join(file_path, "../cv_weights.pth"))
        )

        # get weights for RL model
        rl_model.load_state_dict(
            torch.load(os.path.join(file_path, "../rl_weights.pth"))
        )

        # to eval mode
        cv_model.eval()
        rl_model.eval()

        # to gpu
        cv_model.to(self.cfg.device)
        rl_model.to(self.cfg.device)

        return cv_model, rl_model

    def _update_obs_att(self) -> None:
        self.obs_att = torch.rand((1, 8), dtype=torch.float32, device=self.cfg.device)
        self.obs_att -= 0.5 * 0.5
        self.obs_att[:, 3] = 1.0

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

    def start(self) -> None:
        """The main loop."""
        # start loop, just go as fast as possible for now
        for image in self.read_images():
            # get the camera image and pass to segmentation model, already on gpu
            seg_map, _ = self.cv_model(image)
            seg_map = seg_map.float()

            # update the obs_att
            self._update_obs_att()

            # pass segmap to the rl model to get action, send to cpu
            self.action = self.rl_model.infer(
                *self.rl_model(seg_map)
            ).squeeze(0)
            print(self.action)
            self.action = cpuize(self.action)

            # map action, [stop/go, yaw_rate] -> [frdy]
            # the max linear velocity as defined in the sim is 3.0
            # the max angular velocity as defined in the sim is pi
            self.setpoint = np.array(
                [self.action[0] > 0, 0.0, 0.0, self.action[1]]
            )

            print(f"FRDY: {self.setpoint}")

            # cv2.imshow("something", cpuize(((image + 1.0) / 2.0).squeeze()).transpose(1, 2, 0))
            plt.imshow(cpuize(seg_map.squeeze()[0]))
            plt.show()


if __name__ == "__main__":
    agent = Agent()
    agent.start()
