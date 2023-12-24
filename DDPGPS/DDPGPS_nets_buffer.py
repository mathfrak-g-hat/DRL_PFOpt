#######################################################################
##### Per-Step DDPG Agent - Neural nets and replay buffer classes #####
#######################################################################
## 23/04/13, C Cheng
## 23/05/19, AJ Zerouali

# Basic imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Datetime, deque
from datetime import datetime, timedelta, date
from collections import deque
import random

# PyTorch
import torch as th
from torch import nn
from torch import optim

# DRL_PFOPT
from drl_pfopt import PortfolioOptEnv
from dual_timeframe_XP import PFOptDualTFEnv


'''
        ###############################
        ##### ACTOR NETWORK CLASS #####
        ###############################
        # VERSION 3: 23/05/18
        COMMENTS (23/05/18)
        - The forward() method of this nn.Module computes
          portfolio weights.
        - At time of writing (23/05/18), there is only one
          action normalization implemented, softmax.
        - There are additional parameters, see docstring 
          of constructor.
        
'''
class Actor(nn.Module):
    '''
        Actor class for DDPG per-step algorithm
    '''
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 h1: int, 
                 h2: int,
                 weight_normalization: str = "softmax", 
                 activation_fn: str = "sigmoid", #sigmoid, tanh, relu
                 activation_scale: float = 1.0,
                 init_w: float = 0.003,
                ):
        '''
            :param state_dim: int, no. of units in input layer.
                        Should be the shape obtained from state.view(1,-1)
                        if "state" is an array returned by the environment.
            :param action_dim: int, no. of components of output vector.
                        Should equal env.n_assets if env is the environment
                        used.
            :param h1: int, no. of neurons in 1st hidden layer.
            :param h2: int, no. of neurons in 2nd hidden layer.
            :param weight_normalization: str, normalization type for portfolio weights.
                        default is "softmax"
            :param activation_fn: str, 
            :param activation_scale: float, multiplication factor for pre-normalization 
                        activation. Default: 1.0.
            :param init_w: float, weight initialization. Default: 0.003.
            
        '''
        
        super(Actor, self).__init__()
        
        # Layers
        self.linear1 = nn.Linear(state_dim, h1)
        self.linear1.weight.data = fanin_(self.linear1.weight.data.size())
        self.linear2 = nn.Linear(h1, h2)
        self.linear2.weight.data = fanin_(self.linear2.weight.data.size())
        self.linear3 = nn.Linear(h2, action_dim)
        self.linear3.weight.data.uniform_(-init_w, init_w)

        # Activation functions
        self.activation_scale = activation_scale
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)
        if activation_fn == "sigmoid":
            self.activation_fn = self.sigmoid
        elif activation_fn == "tanh":
            self.activation_fn = self.tanh
        elif activation_fn == "relu":
            self.activation_fn = self.relu
        elif activation_fn == "identity":
            self.activation_fn = self.identity_activation
        else:
            raise ValueError(f"Activation function {activation_fn} is not supported")
        
        # Normalization to portfolio weights
        if weight_normalization == "softmax":
            self.normalize = self.softmax_normalization
        elif weight_normalization == "relu":
            self.normalize = self.relu_normalization
        else:
            raise ValueError(f"Weight normalization{weight_normalization} is not supported")
        
                
        # Device
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def identity_activation(self,
                            x: th.FloatTensor,
                           ):
        return x
    
    def softmax_normalization(self,
                              x: th.FloatTensor,
                             ):
        return self.softmax(x)
    
    def relu_normalization(self,):
        raise NotImplementedError("relu normalization not supported")
    
    def forward(self, state):
        
        # Forward pass thru layers
        x = self.linear1(state)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        # Pre-normalization activation
        x = self.activation_fn(x)
        x = self.activation_scale*x
        # Normalization to portfolio weights
        x = self.normalize(x)
        return x

def fanin_(size):
    '''
        Helper function
    '''
    fan_in = size[0]
    weight = 1./np.sqrt(fan_in)
    return th.Tensor(size).uniform_(-weight, weight)
    
    

'''
        ################################
        ##### CRITIC NETWORK CLASS #####
        ################################
        # VERSION 3: 23/05/18
        COMMENTS (23/05/18)
'''
class R_Critic(nn.Module):
    '''
        Pseudo-critic module for DDPGPS agent.
        Uses the last available close price returns
        in the environment as a prediction of the
        portfolio asset returns.
        The forward function outputs the prediction
        of the portfolio return based on the
        actions interpreted as portfolio weights.
    '''
    
    def __init__(self, env: PortfolioOptEnv):
        '''
          :param env: PFOptDualTFEnv, environment object
        
        '''
        super(R_Critic, self).__init__()
        # Device
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # Input shape param
        self.n_assets = env.n_assets
                
        # Close returns hist
        self.CloseReturns = self.get_returns_array(env)
    
    def get_returns_array(self, env: PortfolioOptEnv,):
        '''
            Function to compute the close returns array 
            from the dataset contained in an environment.
            Here, the env parameter is assumed to be a
            PFOptDualTFEnv object.
            These are not exactly the same close returns as 
            the ones computed in the env state features.
        '''
        # Get close timestamp list
        close_timestamp_list = []
        trade_timestamp_list = env.trade_timeframe_info_dict["trade_timestamp_list"]
        trading_data_schedule = env.trade_timeframe_info_dict["trading_data_schedule"]
        for trade_timestamp in trade_timestamp_list:
            close_timestamp_list.append(trading_data_schedule[trade_timestamp][-1])
        
        # Get reduced close prices dataframe and array, compute close returns
        df_close = env.df_X[env.df_X["date"].isin(close_timestamp_list)]
        df_close = df_close.pivot_table(index='date',columns = 'tic',values = 'close')
        np_close = df_close.to_numpy()
        X_close_ret = np.zeros(shape=(len(trade_timestamp_list),env.n_assets))
        X_close_ret[1:,:] = (np_close[1:,:]-np_close[:-1,:])/np_close[:-1,:]
        
        # Return output
        return X_close_ret

    def forward(self,
                action_t: th.FloatTensor, 
                t: np.ndarray,): 
        '''
          Pseudo-critic forward function. Inputs are batches of 
          actions and time indices. The action batch must have
          shape (batch_size, n_assets), where:
          - n_assets is one of the R_Critic attributes obtained
            from the env in the constructor.
          - batch_size is the length of the time index batch.
          The time index batch is used to locate the close price
          returns in the CloseReturns attribute computed from 
          environment (see constructor).
          The computation of the predicted portfolio return
          uses the torch.bmm() method:
          https://pytorch.org/docs/stable/generated/torch.bmm.html

          :param action_t: th.FloatTensor, batch of portfolio weights.
          :param t_idx: np.ndarray, batch of trading timestamp indices.
          
          :return prod_batch: th.FloatTensor, batch of predicted 
                              portfolio returns.
        '''
        # Required input shape
        batch_size = len(t)
        input_shape = (batch_size, self.n_assets)
    
        # Ensure that the shape is correct
        if (action_t.shape != input_shape):
            raise ValueError("action_t must have shape (len(t), n_assets)")

        # COMMENT: For th.bmm(batch_1, batch_2), you will need:
        ## batch_1.shape = (batch_size, n, m), batch_2.shape = (batch_size, m, l),
        ## so that the (n,m)x(m,l) matrix multiplication is defined.
        returns_t = th.FloatTensor(self.CloseReturns[t,:]).to(self.device)
        prod_batch = th.bmm(action_t.unsqueeze(1),
                            returns_t.unsqueeze(2)
                            ).squeeze(1)

        return prod_batch

    
'''
        ###############################
        ##### REPLAY BUFFER CLASS #####
        ###############################
'''
class replayBuffer(object):
    def __init__(self, buffer_size, name_buffer=''):
        self.name_buffer = name_buffer
        self.buffer_size = buffer_size  #choose buffer size
        self.num_exp = 0
        self.buffer = deque()

    def add(self, s, a, r, s2, t):
        experience=(s, a, r, s2, t)
        if self.num_exp < self.buffer_size:
            self.buffer.append(experience)
            self.num_exp +=1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.buffer_size

    def count(self):
        return self.num_exp

    def sample(self, batch_size):
        if self.num_exp < batch_size:
            batch = random.sample(self.buffer, self.num_exp)
        else:
            batch = random.sample(self.buffer, batch_size)

        s, a, r, s2, t = map(np.stack, zip(*batch))

        return s, a, r, s2, t

    def clear(self):
        self.buffer = deque()
        self.num_exp=0