from __future__ import annotations

import numpy as np
import torch
from wingman import Wingman, cpuize, gpuize

import zmq
from cv import EnsembleAttUNet
from midware import Camera
from rl import GaussianActor

class Agent():
    """The CVRL agent.

    This script hosts the CV and RL models.
    It continuously reads the camera image, and subscribes to the obs_att on port 5555.
    It publishes the velocity setpoint on port 5556.
    None of the functions are designed to be called after `start()`, which is a blocking loop.
    """
    def __init__(self):
        self.wm = Wingman(config_yaml="src/settings.yaml")
        self.cfg = self.wm.cfg

        """MODELS AND CAMERA"""
        self.cv_model, self.rl_model = self._setup_nets()
        self.camera = Camera(self.cfg.base_resize)

        """RUNTIME PARAMETERS"""
        self.obs_att = np.zeros((4,), dtype=np.float32)
        self.action = np.zeros((4,), dtype=np.float32)

        """CONSTANTS"""
        self.action_scaling = np.array([1.0, 2.0, 2.0, 2.0])

        """PYZMQ SOCKETS"""
        # attitude subscriber
        context = zmq.Context()
        self.att_sub = context.socket(zmq.SUB)
        self.att_sub.bind("tcp://127.0.0.1:5555")
        self.att_sub.setsockopt_string(zmq.SUBSCRIBE, "")

        # setpoint publisher
        context = zmq.Context()
        self.set_pub = context.socket(zmq.PUB)
        self.set_pub.bind("tcp://127.0.0.1:5556")

    def _setup_nets(self) -> tuple[EnsembleAttUNet, GaussianActor]:
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
            obs_att_size=self.cfg.obs_att_size,
            obs_img_size=self.cfg.obs_img_size,
        )

        if False:
            # get weights for CV model
            cv_model.load_state_dict(torch.load("./weights/Version0/weights0.pth"))

            # get weights for RL model
            rl_state_dict = rl_model.state_dict()
            for name, param in torch.load("./weights/Version0/weights0.path"):
                if name not in rl_state_dict:
                    continue
                rl_state_dict[name].copy_(param)

        # to eval mode
        cv_model.eval()
        rl_model.eval()

        # to gpu
        cv_model.to(self.cfg.device)
        rl_model.to(self.cfg.device)

        return cv_model, rl_model

    def _update_obs_att(self):
        # check whether we have state update
        try:
            attitude = self.att_sub.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.Again:
            return

        self.obs_att = torch.tensor([[*attitude["lin_vel"], attitude["altitude"], *self.action]])
        return

    def start(self):
        # start loop, just go as fast as possible for now
        for cam_img in self.camera.stream(self.cfg.device):
            # get the camera image and pass to segmentation model, already on gpu
            seg_map, _ = self.cv_model(cam_img)
            seg_map = seg_map.float()

            # update the obs_att
            self._update_obs_att()

            # pass segmap to the rl model to get action, send to cpu
            self.action = self.rl_model.infer(*self.rl_model(self.obs_att, seg_map)).squeeze(0)
            self.action = cpuize(self.action)

            # map action, [stop/go, right_drift, yaw_rate, climb_rate] -> [frdy]
            # the max linear velocity as defined in the sim is 3.0
            # the max angular velocity as defined in the sim is pi
            self.setpoint = np.array([self.action[0] > 0, -self.action[1], -self.action[3], -self.action[2]])
            self.setpoint *= self.action_scaling

            # publish the setpoint
            self.set_pub.send_pyobj(self.setpoint)

if __name__ == "__main__":
    agent = Agent()
    agent.start()
