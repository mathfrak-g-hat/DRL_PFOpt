##########################
### SAC DROPOUT POLICY ###
##########################
### AJ Zerouali, 23/04/10

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
    SquashedDiagGaussianDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import (
    get_action_dim, 
    is_image_space, 
    maybe_transpose, 
    preprocess_obs
)
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor

### sac.sac.policies imports
from stable_baselines3.common.torch_layers import get_actor_critic_arch
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import (
    BaseModel,
    BasePolicy,
    ActorCriticPolicy,
    ContinuousCritic,
)


'''
    Temporary imports (AJ Zerouali)
'''
from SB3_Extras_XP import (create_dropout_mlp, 
                           DropOutContinuousCritic,
                           get_AC_dropout_prob_lists,
                          )

'''
    Package version imports
    =======================

from drl_pfopt.sb3_extras.common.torch_layers_extra import (create_dropout_mlp,
                                                            get_AC_dropout_prob_lists,
                                                            )
from drl_pfopt.sb3_extras.common.policies_extra import DropOutContinuousCritic
'''

'''
###################
### ACTOR CLASS ###
###################
## Goes into drl_pfopt.sb3_extras.sac.sac_dropout_policy.py
## This is an extension of stable_baselines3.sac.policies.Actor class
'''

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(BasePolicy):
    """
    Actor network (policy) for SAC.
    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
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
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.dropout_probs_list = dropout_probs_list # NEW
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.add_input_dropout = add_input_dropout # NEW
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        action_dim = get_action_dim(self.action_space)
        #latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn) # THIS HAS TO BE MODIFIED
        # NEW
        latent_pi_net = create_dropout_mlp(input_dim = features_dim, 
                                           output_dim = -1, # Why? (AJ Zerouali 22/12/02) 
                                           net_arch = net_arch, 
                                           dropout_probs_list = dropout_probs_list,
                                           activation_fn = activation_fn,
                                           add_input_dropout = add_input_dropout,
                                          )
        
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                dropout_probs_list = self.dropout_probs_list, # NEW
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                add_input_dropout = self.add_input_dropout, # NEW
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using gSDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).
        :return:
        """
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.
        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs, self.features_extractor) # AJ Zerouali, 23/04/10
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)


'''
##############################
### SAC WITH DROPOUT CLASS ###
##############################
## Goes into drl_pfopt.sb3_extras.sac.sac_dropout_policy.py
## This is an extension of stable_baselines3.sac.policies.SACPolicy class
'''
class SACDropOutPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for SAC.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
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
        dropout_probs_list: Optional[Union[List[int], Dict[str, List[int]]]] = None, # IMPORTANT: Fix initialization somewhere
        activation_fn: Type[nn.Module] = nn.ReLU,
        add_input_dropout: bool = False, # NEW ALSO.
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
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
        
        # Case where net_arch is NONE
        if net_arch is None:
            net_arch = [256, 256]
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
        
        # Default dropout layers (NEW)
        '''
            We have an issue with this initialization
        
        #len_net_arch = 
        if dropout_probs_list is None:
            dropout_probs_list = []
            if add_input_dropout:
                dropout_probs_list.append(0.1)
                for x in range(len_net_arch-1):
                    dropout_probs_list.append(0.5)
                dropout_probs_list.append(0.0)
            else:
                for x in range(len_net_arch-1):
                    dropout_probs_list.append(0.5)
                dropout_probs_list.append(0.0)
        '''

        
        #### ASSIGN SACDropoutPolicy ATTRIBUTES ####

        self.net_arch = net_arch
        self.dropout_probs_list = dropout_probs_list # NEW
        self.activation_fn = activation_fn
        self.add_input_dropout = add_input_dropout # NEW
        
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "dropout_probs_list": actor_dropout_probs_list, # NEW
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "add_input_dropout": add_input_dropout, # NEW
        }
        
        self.actor_kwargs = self.net_args.copy()

        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
                "dropout_probs_list": critic_dropout_probs_list, # NEW
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic.named_parameters() if "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = self.critic.parameters()

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                dropout_probs_list = self.dropout_probs_list, # NEW
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                add_input_dropout = self.add_input_dropout, # NEW
            )
        )
        return data

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        :param batch_size:
        """
        self.actor.reset_noise(batch_size=batch_size)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return DropOutContinuousCritic(**critic_kwargs).to(self.device) # NEW

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.
        This affects certain modules, such as batch normalisation and dropout.
        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


DropOutMlpPolicy = SACDropOutPolicy
