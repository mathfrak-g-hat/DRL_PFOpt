##########################
### TD3 DROPOUT POLICY ###
##########################
### AJ Zerouali, 23/04/11

# common.policies imports
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


### td3.td3.policies imports
from stable_baselines3.common.torch_layers import get_actor_critic_arch
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import (
    BaseModel,
    BasePolicy,
    ActorCriticPolicy,
    ContinuousCritic,
)

# My imports
from stable_baselines3.td3.policies import (
    #Actor, # Have our own actor class here
    TD3Policy,
)


from drl_pfopt.sb3_extras.common.torch_layers_extra import (create_dropout_mlp,
                                                            get_AC_dropout_prob_lists,
                                                            )
from drl_pfopt.sb3_extras.common.policies_extra import DropOutContinuousCritic

'''
###################
### ACTOR CLASS ###
###################
## Goes into drl_pfopt.sb3_extras.td3.td3_dropout_policy.py
## This is an extension of stable_baselines3.td3.policies.Actor class
'''
class Actor(BasePolicy):
    """
    Actor network (policy) for TD3 with dropout layers.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
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
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.net_arch = net_arch
        self.dropout_probs_list = dropout_probs_list # NEW
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.add_input_dropout = add_input_dropout # NEW

        action_dim = get_action_dim(self.action_space)
        # NEW
        actor_net = create_dropout_mlp(input_dim = features_dim, 
                                       output_dim = action_dim, 
                                       net_arch = net_arch, 
                                       dropout_probs_list = dropout_probs_list,
                                       activation_fn = activation_fn,
                                       squash_output=True,
                                       add_input_dropout = add_input_dropout,
                                      )
        
        # Deterministic action
        self.mu = nn.Sequential(*actor_net)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                dropout_probs_list = self.dropout_probs_list, # NEW
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
                add_input_dropout = self.add_input_dropout # NEW
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs, self.features_extractor) # AJ Zerouali, 23/04/10
        return self.mu(features)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self(observation)
'''
##############################
### TD3 WITH DROPOUT CLASS ###
##############################
## Goes into drl_pfopt.sb3_extras.td3.td3_dropout_policy.py
## This is an extension of stable_baselines3.td3.policies.TD3Policy class
'''
class TD3DropOutPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for TD3 that supports dropout layers.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        dropout_probs_list: List[float] = None, # IMPORTANT: Fix initialization somewhere
        activation_fn: Type[nn.Module] = nn.ReLU,
        add_input_dropout: bool = False, # NEW ALSO.
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )
        
        #### ACTOR-CRITIC ARCHITECTURES INITIALIZATION ####

        # Default network architecture, from the original paper
        # Case where net_arch is NONE
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = [256, 256]
            else:
                net_arch = [400, 300]
            # Case when net_arch param AND dropout list are NONE
            if dropout_probs_list is None:
                dropout_probs_list = []
                if add_input_dropout:
                    dropout_probs_list.append(0.1)
                    for x in range(len(net_arch)-1):
                        dropout_probs_list.append(0.5)
                    dropout_probs_list.append(0.0)
                else:
                    for x in range(len(net_arch)-1):
                        dropout_probs_list.append(0.5)
                    dropout_probs_list.append(0.0)

        # Assign actor and critic architecture lists
        actor_arch, critic_arch = get_actor_critic_arch(net_arch) 
        
        # Case where dropout_probs_list is NONE but net_arch != None
        ### Example: net_arch was specified but not dropout_probs_list
        if dropout_probs_list is None:
            
            dropout_probs_list = {"qf":[], "pi":[]}
            
            # Case where we add an input layer dropout
            if add_input_dropout:
                # Actor
                dropout_probs_list["pi"].append(0.1)
                for x in range(len(actor_arch)-1):
                    dropout_probs_list["pi"].append(0.5)
                dropout_probs_list["pi"].append(0.0)
                
                # Critic
                dropout_probs_list["qf"].append(0.1)
                for x in range(len(critic_arch)-1):
                    dropout_probs_list["qf"].append(0.5)
                dropout_probs_list["qf"].append(0.0)
                
            # Case where we do not add an input layer dropout
            else:
                # Actor
                for x in range(len(actor_arch)-1):
                    dropout_probs_list["pi"].append(0.5)
                dropout_probs_list["pi"].append(0.0)
                
                # Critic
                for x in range(len(critic_arch)-1):
                    dropout_probs_list["qf"].append(0.5)
                dropout_probs_list["qf"].append(0.0)

        actor_dropout_probs_list, critic_dropout_probs_list = get_AC_dropout_prob_lists(dropout_probs_list)
        
        #### ASSIGN TD3DropoutPolicy ATTRIBUTES ####
        
        self.net_arch = net_arch
        self.dropout_probs_list = dropout_probs_list # NEW
        self.activation_fn = activation_fn
        self.add_input_dropout = add_input_dropout # NEW
        
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "dropout_probs_list": self.dropout_probs_list, # NEW
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "add_input_dropout": self.add_input_dropout, # NEW
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        # Create actor and target
        # the features extractor should not be shared
        self.actor = self.make_actor(features_extractor=None)
        self.actor_target = self.make_actor(features_extractor=None)
        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Critic target should not share the features extactor with critic
            # but it can share it with the actor target as actor and critic are sharing
            # the same features_extractor too
            # NOTE: as a result the effective poliak (soft-copy) coefficient for the features extractor
            # will be 2 * tau instead of tau (updated one time with the actor, a second time with the critic)
            self.critic_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
        else:
            # Create new features extractor for each network
            self.critic = self.make_critic(features_extractor=None)
            self.critic_target = self.make_critic(features_extractor=None)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        # Target networks should always be in eval mode
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                dropout_probs_list = self.dropout_probs_list, # NEW
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
                add_input_dropout = self.add_input_dropout, # NEW
            )
        )
        return data

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> DropOutContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return DropOutContinuousCritic(**critic_kwargs).to(self.device) # NEW

    def forward(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.actor(observation)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode
        
DropOutMlpPolicy = TD3DropOutPolicy