from __future__ import annotations

import os
from signal import SIGINT, signal

import numpy as np
import torch
from wingman import Wingman, cpuize, gpuize, shutdown_handler

from cv import EnsembleAttUNet
from midware import Camera, Vehicle
from rl import GaussianActor


def deploy(wm: Wingman):
    # grab the config
    cfg = wm.cfg

    # setup models, camera, midware
    cv_model, rl_model = setup_nets(wm)
    camera = Camera(cfg.base_resize)
    vehicle = Vehicle(
        "/dev/ttyACM0",
        state_update_rate=4,
        setpoint_update_rate=2,
        flight_ceiling=5.0,
    )

    # start loop
    action = np.zeros((4,), dtype=np.float32)
    action_scaling = np.array([1.0, 2.0, 2.0, 2.0])
    for cam_img in camera.stream(cfg.device):
        # get the camera image and pass segmentation model, already on gpu
        seg_map, _ = cv_model(cam_img)
        seg_map = seg_map.float()

        # get the vehicle's attitude, send to gpu
        obs_att = torch.tensor([[*vehicle.lin_vel, vehicle.altitude, *action]])
        obs_att = gpuize(obs_att, cfg.device)

        # pass segmap to the rl model to get action, send to cpu
        action = rl_model.infer(*rl_model(obs_att, seg_map))
        action = cpuize(action)

        # map action, [stop/go, right_drift, yaw_rate, climb_rate] -> [frdy]
        # the max linear velocity as defined in the sim is 3.0
        # the max angular velocity as defined in the sim is pi
        setpoint = np.array([action[0] > 0, -action[1], -action[3], -action[2]])
        vehicle.set_velocity_setpoint(setpoint * action_scaling)
        print(setpoint)


def setup_nets(wm: Wingman) -> tuple[EnsembleAttUNet, GaussianActor]:
    cfg = wm.cfg

    # cfg up networks
    cv_model = EnsembleAttUNet(
        in_channels=cfg.in_channels,
        out_channels=cfg.num_labels,
        inner_channels=cfg.inner_channels,
        att_num_heads=cfg.att_num_heads,
        num_att_module=cfg.num_att_module,
        num_ensemble=cfg.num_ensemble,
    ).to(cfg.device)

    rl_model = GaussianActor(
        act_size=cfg.act_size,
        obs_att_size=cfg.obs_att_size,
        obs_img_size=cfg.obs_img_size,
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
    cv_model.to(cfg.device)
    rl_model.to(cfg.device)

    return cv_model, rl_model


if __name__ == "__main__":
    signal(SIGINT, shutdown_handler)
    wm = Wingman(config_yaml="src/settings.yaml")

    """ SCRIPTS HERE """
    with torch.no_grad():
        deploy(wm)
