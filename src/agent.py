from __future__ import annotations

import os
import threading
import time

import numpy as np
import torch
from rl import GaussianActor
from wingman import Wingman, cpuize, gpuize

from cv import EnsembleAttUNet
from midware import Camera


class Agent:
    """The CVRL agent.

    This script hosts the CV and RL models.
    """

    def __init__(self):
        """__init__."""
        self.wm = Wingman(config_yaml="src/settings.yaml")
        self.cfg = self.wm.cfg

        """MODELS AND CAMERA"""
        self.cv_model, self.rl_model = self._setup_nets()
        self.camera = Camera(self.cfg.base_resize)

        """RUNTIME PARAMETERS"""
        self._last_attitude_time = time.time()
        self.obs_att = torch.zeros((1, 8), dtype=torch.float32, device=self.cfg.device)
        self.action = torch.zeros((4,), dtype=torch.float32, device=self.cfg.device)
        self.setpoint = np.zeros((4,), dtype=np.float32)

        self._stale_attitude_watcher()

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
        rl_state_dict = rl_model.state_dict()
        for name, param in torch.load(
            os.path.join(file_path, "../rl_weights.pth")
        ).items():
            if "actor." not in name:
                continue
            rl_state_dict[name.replace("actor.", "")].copy_(param)

        # to eval mode
        cv_model.eval()
        rl_model.eval()

        # to gpu
        cv_model.to(self.cfg.device)
        rl_model.to(self.cfg.device)

        return cv_model, rl_model

    def update_attitude(self, attitude: dict) -> None:
        """Updates the internal attitude."""
        self._last_attitude_time = time.time()
        self.obs_att = torch.tensor(
            [[*attitude["lin_vel"], attitude["altitude"], *self.action]],
            device=self.cfg.device,
        )

    def get_setpoint(self) -> np.ndarray:
        """The main loop."""
        # get the camera image and pass to segmentation model, already on gpu
        seg_map, _ = self.cv_model(self.camera.get_image(self.cfg.device))
        seg_map = seg_map.float()

        # pass segmap to the rl model to get action, send to cpu
        self.action = self.rl_model.infer(
            *self.rl_model(seg_map)
        ).squeeze(0)
        self.action = cpuize(self.action)

        # we want to maintain a height of 1 m, down is +velocity
        climb_rate = -(1.0 - self.obs_att[3]) * 0.5

        # map action, [stop/go, yaw_rate] -> [frdy]
        # the max linear velocity as defined in the sim is 3.0
        # the max angular velocity as defined in the sim is pi
        self.setpoint = np.array(
            [self.action[0] > 0, 0.0, climb_rate, self.action[1]]
        )

        return self.setpoint

    def _stale_attitude_watcher(self) -> None:
        """Watches for stale attitude."""
        print(f"Setpoint FRDY: {self.setpoint}")

        # check the zmq
        stale_time = time.time() - self._last_attitude_time
        if stale_time > 3.0:
            print(f"Attitude estimate stale for {stale_time} seconds.")

        # queue the next call
        t = threading.Timer(1.0, self._stale_attitude_watcher)
        t.daemon = True
        t.start()
