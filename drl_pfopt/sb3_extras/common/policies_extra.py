#############################################
### DRL-PFOPT - Stable Baselines 3 Extras ###
#############################################
### AJ Zerouali, 23/03/27
# This submodule complements stable_baselines3.common.policies.

###################
### SB3 IMPORTS ###
###################

import collections
import copy
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor

SelfBaseModel = TypeVar("SelfBaseModel", bound="BaseModel")

#########################
### DRL-PFOPT IMPORTS ###
#########################

from stable_baselines3.common.policies import (
    BaseModel,
    BasePolicy,
    ActorCriticPolicy,
    ContinuousCritic,
)

# Create dropout mlp helper function
from drl_pfopt.sb3_extras.common.torch_layers_extra import create_dropout_mlp

#################################
### DROPOUT CONTINUOUS CRITIC ###
#################################
## 23/04/10, AJ Zerouali
## This is an extension of stable_baselines3.common.policies.ContinuousCritic
## class used by TD3/SAC/DDPG.
## Compatible with SB3 v.1.8.0.
class DropOutContinuousCritic(BaseModel):
    """
    Critic network(s) class for DDPG/SAC/TD3 with dropout layers.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        dropout_probs_list: List[float],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        add_input_dropout: bool = False,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = create_dropout_mlp(input_dim = features_dim + action_dim, 
                                       output_dim = 1, 
                                       net_arch = net_arch, 
                                       dropout_probs_list = dropout_probs_list,
                                       activation_fn=activation_fn,
                                       squash_output = False,
                                       add_input_dropout = add_input_dropout,
                                      )
            q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)


    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor) # AJ Zerouali, 23/04/10
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor) # AJ Zerouali, 23/04/10
        return self.q_networks[0](th.cat([features, actions], dim=1))
    