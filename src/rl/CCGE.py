#!/usr/bin/env python3
from __future__ import annotations

import warnings

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as func

from .CCGENet import Actor


class GaussianActor(nn.Module):
    """
    Gaussian Actor
    """

    def __init__(self, act_size, obs_att_size, obs_img_size):
        super().__init__()
        self.net = Actor(act_size, obs_att_size, obs_img_size)

    def forward(self, obs_atti, obs_targ):
        output = self.net(obs_atti, obs_targ)
        return output[0], output[1]

    @staticmethod
    def sample(mu, sigma):
        """
        output:
            actions is of shape B x act_size
            entropies is of shape B x 1
        """
        # lower bound sigma and bias it
        normals = dist.Normal(mu, func.softplus(sigma) + 1e-6)

        # sample from dist
        mu_samples = normals.rsample()
        actions = torch.tanh(mu_samples)

        # calculate log_probs
        log_probs = normals.log_prob(mu_samples) - torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1, keepdim=True)

        return actions, log_probs

    @staticmethod
    def infer(mu, sigma):
        return torch.tanh(mu)
