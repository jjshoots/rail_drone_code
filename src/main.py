from __future__ import annotations

import os
from signal import SIGINT, signal

import matplotlib
import torch
from wingman import Wingman, cpuize, gpuize, shutdown_handler

from camera import Camera
from cv.att_unet import EnsembleAttUNet
from rl.CCGE import GaussianActor


def test(wm: Wingman):
    # grab the config
    cfg = wm.cfg

    # setup models
    cv_model, rl_model = setup_nets(wm)

    # setup the camera
    camera = Camera(cfg.base_resize)

    for cam_img in camera.stream(cfg.device):
        # pass image through the cv model
        seg_map, _ = cv_model(cam_img)
        seg_map = seg_map.float()

        # pass segmap to the rl model
        obs_att = torch.zeros((1, 10), device=cfg.device)
        action = rl_model.infer(*rl_model(obs_att, seg_map))
        print(action)


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
    try:
        matplotlib.use("TKAgg")
    except Exception as e:
        print(f"Unable to change mpl backend, are you on headless? {e}")
    signal(SIGINT, shutdown_handler)
    wm = Wingman(config_yaml="src/settings.yaml")

    """ SCRIPTS HERE """
    with torch.no_grad():
        test(wm)
