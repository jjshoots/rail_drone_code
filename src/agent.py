from __future__ import annotations

import threading
import time
from os import path

import numpy as np
import torch
from wingman import Wingman, cpuize, gpuize

from cv import EnsembleAttUNet
from cv.mean_filter import MeanFilter
from midware import Camera
from rl import GaussianActor


class Agent:
    """The CVRL agent.

    This script hosts the CV and RL models.
    """

    def __init__(self):
        """__init__."""
        self.wm = Wingman(config_yaml="src/settings.yaml")
        self.cfg = self.wm.cfg

    def setup(self):
        """_default_start."""
        self._setup_nets()
        self._setup_camera()

        self._last_attitude_time = time.time()
        self._current_attitude = np.zeros((4,), dtype=np.float64)

        self._stale_attitude_watcher()

    def _setup_camera(self):
        """_setup_camera."""
        self.camera = Camera(self.cfg.base_resize)

    def _setup_nets(self):
        """_setup_nets."""
        # cfg up networks
        self.cv_model = EnsembleAttUNet(
            in_channels=self.cfg.in_channels,
            out_channels=self.cfg.num_labels,
            inner_channels=self.cfg.inner_channels,
            att_num_heads=self.cfg.att_num_heads,
            num_att_module=self.cfg.num_att_module,
            num_ensemble=self.cfg.num_ensemble,
        ).to(self.cfg.device)
        self.mean_filter = MeanFilter()
        self.rl_model = GaussianActor(
            act_size=self.cfg.act_size,
            obs_size=self.cfg.obs_size,
        )

        # get weights for CV and RL model
        self.cv_model.load_state_dict(
            torch.load(
                path.join(path.dirname(path.realpath(__file__)), "../cv_weights.pth")
            )
        )
        self.rl_model.load_state_dict(
            torch.load(
                path.join(path.dirname(path.realpath(__file__)), "../rl_weights.pth")
            )
        )

        # to eval mode
        self.cv_model.eval()
        self.mean_filter.eval()
        self.rl_model.eval()

        # to gpu
        self.cv_model.to(self.cfg.device)
        self.mean_filter.to(self.cfg.device)
        self.rl_model.to(self.cfg.device)

    def set_attitude(self, attitude: dict) -> None:
        """Updates the internal attitude."""
        self._last_attitude_time = time.time()
        self._current_attitude = np.array([*attitude["lin_vel"], attitude["altitude"]])

    def get_setpoint(self) -> np.ndarray:
        """The main loop."""
        # get the camera image and pass to segmentation model, already on gpu
        seg_map, _ = self.cv_model(self.camera.get_image(self.cfg.device))

        # mean filter the segmap to remove spurious predictions
        seg_map[0] = (self.mean_filter(seg_map[0].float()) == 1.0).float()

        # pass segmap to the rl model to get action, send to cpu
        action = cpuize(self.rl_model.infer(*self.rl_model(seg_map)).squeeze(0))

        # we want to maintain a height of 1 m, down is +velocity
        climb_rate = -(1.0 - self._current_attitude[3]) * 0.5

        # map action, [stop/go, yaw_rate] -> [frdy]
        # the max linear velocity as defined in the sim is 3.0
        # the max angular velocity as defined in the sim is pi
        setpoint = np.array([1.0, 0.0, climb_rate, action[0]])

        return setpoint

    def _stale_attitude_watcher(self) -> None:
        """Watches for stale attitude."""
        # check the zmq
        stale_time = time.time() - self._last_attitude_time
        if stale_time > 3.0:
            print(f"Attitude estimate stale for {stale_time} seconds.")

        # queue the next call
        t = threading.Timer(1.0, self._stale_attitude_watcher)
        t.daemon = True
        t.start()
