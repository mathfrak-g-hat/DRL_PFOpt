# Stable Baselines 3 Extras - Experimental
### AJ Zerouali, 22/12/06

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




##########################
### CREATE DROPOUT MLP ###
##########################
## 23/04/10, AJ Zerouali
## This is an extension of stable_baselines3.common.torch_layers.create_mlp()
## to dropout layers.
def create_dropout_mlp(input_dim: int,
                       output_dim: int,
                       net_arch: List[int],
                       dropout_probs_list: List[float],
                       activation_fn: Type[nn.Module] = nn.ReLU,
                       squash_output: bool = False,
                       add_input_dropout: bool = False,
                      )->List[nn.Module]:
    
    '''
        Extension of the helper function stable_baselines3.common.torch_layers.create_mlp().
        Builds an SB3-compatible neural net with dropout layers.
        
        :param input_dim: int, no. of neurons in input layer
        :param output_dim: int, no. of neurons in output layer
        :param net_arch: list of int, no. of neurons in each hidden layer
        :param dropout_probs_list: list of float, dropout probability for each layer
        :param activation_fn: nn.Module, activation function for each layer, 
                              nn.Relu by default
        :param squash_output: bool, whether or not to use tanh as output activation
        :param add_input_dropout: bool, whether or not to add a dropout layer after 
                                  the input layer
        
        :return modules: list of nn.Module objects, to be used as layers of policy nets 
                        (input of nn.Sequential() for policy constructors)
        
    '''
    
    if not all((x<=1) and (0<=x) for x in dropout_probs_list):
        raise ValueError("All entries of dropout_probs_list must be between 0 and 1.")
    else:
        if not add_input_dropout:
            if len(dropout_probs_list) != len(net_arch):
                raise ValueError("dropout_probs_list and net_arch must have the same no. of elts")
        else:
            if len(dropout_probs_list) != (len(net_arch)+1):
                raise ValueError("len(dropout_probs_list) != (len(net_arch)+1)")
    
    if not add_input_dropout:
        if len(net_arch) > 0:
            modules = [nn.Linear(input_dim, net_arch[0]), 
                       activation_fn()]
        else:
            modules = []
            
        for idx in range(len(net_arch) - 1):
            
            modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
            modules.append(nn.Dropout(p=dropout_probs_list[idx]))
            modules.append(activation_fn())
    else:
        if len(net_arch) > 0:
            modules = [nn.Linear(input_dim, net_arch[0]), 
                       nn.Dropout(p=dropout_probs_list[0]),
                       activation_fn()]
        else:
            modules = []
            
        for idx in range(len(net_arch) - 1):
            
            modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
            modules.append(nn.Dropout(p=dropout_probs_list[idx+1]))
            modules.append(activation_fn())            

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
        modules.append(nn.Dropout(p=dropout_probs_list[-1]))
    if squash_output:
        modules.append(nn.Tanh())
    return modules

    
######################################
### GET ACTOR-CRITIC DROPOUT LISTS ###
######################################
## 23/04/10, AJ Zerouali
## This is an extension of stable_baselines3.common.torch_layers.get_actor_critic_arch()
## to dropout layers. 

def get_AC_dropout_prob_lists(dropout_probs_list: Union[List[int], Dict[str, List[int]]],
                             ) -> Tuple[List[int], List[int]]:
    '''
    Get the actor and critic network dropout probabilities for off-policy actor-critic algorithms (SAC, TD3, DDPG). Functions just like get_actor_critic_arch().
    - If dropout_probs_list is a list of floats, then the same dropout probabilities list
      is assigned to both actor_dropout_probs_list and critic_dropout_probs_list
    - If dropout_probs_list is a dictionary lists with keys "pi" and "qf", then:
      actor_dropout_probs_list = dropout_probs_list["pi"]
      and critic_dropout_probs_list = dropout_probs_list["qf"]
      
    :param dropout_probs_list: The specification of the actor and critic networks.
        See above for details on its formatting.
    :return actor_dropout_probs_list: The lists of dropout probabilities for the actor
    :return critic_dropout_probs_list: The lists of dropout probabilities for the critic
    '''
    if isinstance(dropout_probs_list, list):
        actor_dropout_probs_list, critic_dropout_probs_list = dropout_probs_list, dropout_probs_list
        
    else:
        assert isinstance(dropout_probs_list, dict), "Error: the net_arch can only contain be a list of ints or a dict"
        assert "pi" in dropout_probs_list, "Error: no key 'pi' was provided in net_arch for the actor network"
        assert "qf" in dropout_probs_list, "Error: no key 'qf' was provided in net_arch for the critic network"
        actor_dropout_probs_list, critic_dropout_probs_list = dropout_probs_list["pi"], dropout_probs_list["qf"]
    return actor_dropout_probs_list, critic_dropout_probs_list  
    
'''
##############################
### LSTM FEATURE EXTRACTOR ###
##############################
## Should be in common.torch_layers_extra
class LSTMFeatureExtractor(BaseFeaturesExtractor):
    
    
    def __init__(self, 
                 observation_space: gym.Space, 
                 features_dim: int, # This is the output size
                 n_hidden: int = None, # This is the no. of components of the hidden state
                 n_units: int = 1, # This is the no. of LSTM units in extractor
                ):
        super().__init__(observation_space, 
                         features_dim)
        modules = []
        # The fucking loop
        self.net = nn.Sequential(*modules)
        
        
    
    @property
    def features_dim(self)->int:
        return self._features_dim
    
    def forward(self, observations: th.Tensor)->th.Tensor:
        Y, (h, c) = self.net(observations)
        return Y
    
    
        # Constructor
    def __init__(self, input_size, hidden_size,\
                 num_layers=1, nonlinearity="tanh", output_size=1):
        
        # What is this for?
        super().__init__()
        
        # Hyperparams
        self.input_size = input_size # No. of input channels (NOT length)
        self.hidden_size = hidden_size # No. of components of hidden state
        self.num_layers = num_layers # No. of LSTM cells
        self.output_size = output_size # No. of output channels
        
        
        # Layers
        self.rnn_layer = LSTM(input_size = self.input_size, 
                            hidden_size = self.hidden_size,
                            num_layers = self.num_layers)
        
        self.out_layer = nn.Linear(in_features = self.hidden_size, out_features = self.output_size)
    
    # Forward pass
    def forward(self, X, h, c):
        
        X_out, (h, c) = self.rnn_layer(X, (h,c))
        y = self.out_layer(X_out)
        return y, (h.detach(), c.detach())
    '''
        