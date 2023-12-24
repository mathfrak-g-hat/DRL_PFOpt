#############################################
##### Per-Step DDPG Agent - Agent class #####
#############################################
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
from drl_pfopt import PFOpt_Agent, PortfolioOptEnv
from dual_timeframe_XP import PFOptDualTFEnv

# Actor, Critic and replay buffer classes
from DDPGPS.DDPGPS_nets_buffer import Actor, R_Critic, replayBuffer 


'''
        #####################################
        ##### PER-STEP DDPG AGENT CLASS #####
        #####################################
''' 
        
class DDPGPS_Agent(PFOpt_Agent):
    """
        Portfolio optimization deep RL agent class.
        
    """
    ##################
    ### Contructor ###
    ##################
    ### 23/05/18
    def __init__(self, 
                 train_env: PortfolioOptEnv,
                 test_env: PortfolioOptEnv=None,
                 model_fname: str = None,
                 ):
        '''
            Constructor. Initializes the attributes of
            the agent class to none.
            All these attributes must be instantiated in
            the set_model() method.
        '''
        
        super().__init__(train_env = train_env, 
                         test_env = test_env,
                         model_fname = model_fname,
                        )
        
        # Environment attributes
        self.reward_type = None
        self.train_close_returns = None
        self.test_close_returns = None
        
        # Actor attributes
        self.actor = None
        self.policy_arch = None
        self.policy_optimizer = None
        # self.actor_kwargs       
        
        # Critic attributes
        self.critic = None
        
        # Replay buffer attributes
        self.replay_buffer = None
        self.buffer_size = None
        
        # Learning (hyper)parameters
        self.gamma = None
        self.learn_rate = None
        self.batch_size = None
        self.buffer_size = None
        
        # Supplementary learning attributes
        #self.
        
    ############################
    ### Model initialization ###
    ############################
    ### 23/05/18    
    def set_model(self,
                  actor_n_hidden_1: int = 400,
                  actor_n_hidden_2: int = 200,
                  weight_normalization: str = "softmax",
                  activation_fn: str = "sigmoid",
                  activation_scale: float = 1.0,
                  buffer_size: int = 1000,
                  learn_rate: float = 0.001,
                 ):
                
        # Build actor
        dim_in = self.train_env.obs_space_shape[0]*self.train_env.obs_space_shape[1]
        dim_out = self.train_env.n_assets
        self.actor = Actor(dim_in,
                           dim_out,
                           actor_n_hidden_1,
                           actor_n_hidden_2,
                           weight_normalization = weight_normalization,
                           activation_fn = activation_fn,
                           activation_scale = activation_scale,
                          )
        # Make actor architecture dictionary
        self.policy_arch = {}
        self.policy_arch["state_dim"] = dim_in
        self.policy_arch["action_dim"] = dim_out
        self.policy_arch["h1"] = actor_n_hidden_1
        self.policy_arch["h2"] = actor_n_hidden_2
        self.policy_arch["weight_normalization"] = weight_normalization
        self.policy_arch["activation_fn"] = activation_fn
        self.policy_arch["activation_scale"] = activation_scale
        
        
        # Build pseudo-critic
        ## Note: The critic stores the training dataset close price returns
        self.critic = R_Critic(env = self.train_env)
        
        # Build replay buffer
        self.replay_buffer = replayBuffer(buffer_size)
        self.buffer_size = buffer_size
        
        # Build policy optimizer
        self.learn_rate = learn_rate
        self.policy_optimizer = th.optim.Adam(self.actor.parameters(),
                                              lr=learn_rate)
                
        
        print(f"Per-step DDPG agent with given parameters has been successfully created.")
        return True
        
    ###################################
    ### Evaluate actions from state ###
    ###################################
    # 23/05/18, AJ Zerouali
    def get_action(self, 
                   state: np.ndarray,
                  ):
        '''
            Method to compute actions from a state.
            Called in the model_train() and run_backtest()
            methods.
            
            :param state: np.ndarray, state for action computation
            
            :return action: np.ndarray, portfolio weights computed
                        by actor net.
        '''
        state_  = th.FloatTensor(state).to(self.actor.device)
        action = self.actor.forward(state_.view(1,-1))
        #return action.cpu().detach().numpy()[0]
        return action.cpu().detach().numpy()
        
    #############################
    ### Agent learn method ###
    #############################
    # 23/05/18, AJ Zerouali
    def learn(self):
        '''
            MAIN LEARNING ALGORITHM
            Implementation of the DDPGPS algorithm
            
            :return policy_loss: np.ndarray, negative of return computed
                        by the "critic" network.
        '''
        
        # Check no. of available transitions in buffer
        if self.replay_buffer.count() < self.batch_size:
            # Is this really a good idea
            # DEBUG
            #print(f"Skipped learning. replay_buffer.count() = {replay_buffer.count()}")
            return
        
        
        # Sample batch transitions
        ## NOTE: The states have been flattened here.
        s_batch, a_batch, r_batch, s_next_batch, t_batch = \
            self.replay_buffer.sample(self.batch_size)
        
        # Convert batch objs to th.FloatTensor objects
        ### Note: We only use the current states and time index batches below
        s_batch_ = th.FloatTensor(s_batch).to(self.actor.device)
        
        # Set gradients to 0 for policy optimizer
        self.policy_optimizer.zero_grad()
        
        # Compute actor loss
        policy_loss = -self.critic(self.actor(s_batch_), t_batch).mean()
        #policy_loss = -th.pow(100.0*critic(actor(s_batch_), t_batch).mean(),2)
        
        # Gradient step
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Return policy loss
        return policy_loss.cpu().detach().numpy()

    #############################
    ### Agent training method ###
    #############################
    # 23/05/18, AJ Zerouali
    def train_model(self,
                    n_train_rounds: int = 8,
                    batch_size: int = 32,
                    show_env_plots: bool = True,
                   ):
        '''
            Training method for DDPGPS agent.
            Collects sample transitions and calls the learn() method.
            
            :param n_train_rounds: int, number of training episodes, 
                        i.e. sweeps over the dataset in train_env. Default = 8.
            :param batch_size: int, size of training batch of transitions, default = 32.
            :param show_env_plots: bool, True by default. Whether or not
                        we let the environment display the end of episode plots.
                        
            :return average_reward_hist, average_policy_loss_hist: Lists
                        of average rewards and policy losses over each episode.
        '''
        
        # Set batch size
        self.batch_size = batch_size
        
        # Disable env plots during training
        if not show_env_plots:
            train_env_plot_prds_end_episode = self.train_env.plot_prds_end_episode
            self.train_env.plot_prds_end_episode = 0
        
        # Init. average reward hist
        average_reward_hist = []
        average_policy_loss_hist = []
        
        # Training loop
        for episode in range(n_train_rounds):
            
            # Reset train_env
            s = self.train_env.reset()
            done = False
            
            # Reward averaging variables
            episode_reward_list = []
            episode_policy_loss_list = []
            episode_avg_reward = 0.0
            
            # DEBUG
            action_is_nan = False
            
            # Episode loop
            while not done and not action_is_nan:
                
                # Compute action from current state
                ## NOTE - CLASS VERSION: Only input is the state
                action = self.get_action(s)
                action_is_nan = np.isnan(action).all()
                # DEBUG
                #print(f"action = {action}")
                if action_is_nan:
                    print(f"ACTION IS NAN")
                    print(f"TIME INDEX: train_env.t_idx = {train_env.t_idx}")
                    print(f"LAST STATE: s = {s}")
                    break

                # Step in train_env
                s_next, reward, done, info = self.train_env.step(action)
                
                ## Add current reward to hist
                episode_reward_list.append(reward)
                
                # Store transition in replay_buffer
                ## (Reshape the states in the buffer for convenience)
                self.replay_buffer.add(s.flatten(), action, reward, 
                                       s_next.flatten(), self.train_env.t_idx)
                
                # Run DDPG per step
                ## The policy loss below in a numpy array
                policy_loss = self.learn()
                
                # Add current policy loss to list
                if self.replay_buffer.count() >= self.batch_size:
                    episode_policy_loss_list.append(policy_loss)
                
                # Update current state
                s = s_next
                
            # END EPISODE LOOP
            
            # DEBUG
            if action_is_nan:
                break
            
            # Store last episode's average reward
            ## Compute mean reward
            if self.train_env.reward_type == "portfolio_delta_val":
                episode_avg_reward = np.mean(episode_reward_list)
            elif self.train_env.reward_type == "portfolio_return":
                episode_log_returns = np.log(np.ones(shape = (len(episode_reward_list),))\
                                            +np.array(episode_reward_list)
                                            )
                episode_avg_reward = -1+np.exp(episode_log_returns.mean())
            ## Store
            average_reward_hist.append(episode_avg_reward)
            
            # Store last episode's average policy loss
            if self.replay_buffer.count() >= self.batch_size:
                episode_avg_policy_loss = np.mean(episode_policy_loss_list)
                average_policy_loss_hist.append(episode_avg_policy_loss)
            
            # Verbose for progress       
            if (episode+1) % 4 == 0:
                print(f"######################################################")
                print(f"              TRAINING EPISODE: {episode+1}")
                print(f"######################################################")
                if self.train_env.reward_type == "portfolio_delta_val":
                    average = np.mean(average_reward_hist)
                    print(f"Average reward = {average} (portfolio delta val.)")
                elif self.train_env.reward_type == "portfolio_return":
                    log_returns = np.log(np.ones(shape = (len(average_reward_hist),))\
                                        +np.array(average_reward_hist)
                                        )
                    average = -1 + np.exp(log_returns.mean())
                    print(f"Average reward = {average} (portfolio return)")
                
                # before: policy loss isn't computed before buffer is filled
                if self.replay_buffer.count() >= self.batch_size:
                    avg_policy_loss = np.mean(average_policy_loss_hist)
                    print(f"Average policy loss = {avg_policy_loss}")
                
                print(f"######################################################")
            
    
        # END TRAINING LOOP
        
        # Plot policy loss and mean reward histories
        
        if len(average_reward_hist) == len(average_policy_loss_hist):
            print(f"Plotting average reward and policy loss histories...")
            df_train_hist = pd.DataFrame({"Episode":range(n_train_rounds),
                                          "Avg_Reward":average_reward_hist,
                                          "Avg_Policy_Loss":average_policy_loss_hist,
                                         }).set_index("Episode")
            df_train_hist.plot()
        '''
        else:
            df_reward_hist = 
        '''
        
        # Done training: add something here?
        
        # End of episode n_plots of environment
        if not show_env_plots:
            self.train_env.plot_prds_end_episode = train_env_plot_prds_end_episode
        
        # Update training Boolean
        self.model_is_trained = True
        
        return average_reward_hist, average_policy_loss_hist

    ####################################
    ###     CRUCIAL: Run Backtest    ###
    ####################################
    ## 23/05/18, AJ Zerouali
    # Modified the end of backtest results.
    def run_backtest(self, 
                     test_env: PortfolioOptEnv = None, 
                     deterministic=True,
                     ):
        '''
            Runs backtest in the specified test environment.
            If the agent already has a test environment, 
            it is overwritten by the given test_env parameter.
            Outputs 2 pd.DataFrames: The history of portfolio values
            and the history of portfolio weights, at end of each
            trading period 
            Note (22/10/19): The only period supported for now is daily.
            Note (22/11/15): Backtest results are now returned as a dictionary.
        '''
        # Check if agent is trained and update test environment
        if not self.model_is_trained:
            print("ERROR: Cannot run a backtest with an untrained agent")
            return None
        else:
            if test_env != None:
                self.set_test_env(test_env=test_env)
            
        # Check if there's a test environment
        if self.test_env == None:
            print("ERROR: No test environment provided to run a backtest.")
            return None
        
        else: # This is where backtest starts
            
            # Reset SB3 test environment, get initial state
            state = self.test_env.reset()

            # IMPORTANT NOTE: SB3's get_attr() and method_env() return lists b/c of stacked envs.
            N_periods = self.test_env.N_periods
            
            # Main loop
            for i in range(N_periods): # Needs to be changed
                
                # Compute action corresponding to current observation
                action = self.get_action(state)
                
                # Execute the step() function
                ### WARNING (22/10/19): I don't see state_ being reused here. C'est normal ça?
                state, rewards, done, info = self.test_env.step(action)
                
                
                # Store history
                if i == (N_periods - 1): 
                    # Important: i = (N_periods-2) is when the environment's t_idx is updated to (N_periods-1)
                    # The assignment is done here because SB3 automatically resets the environment to t_idx = 0
                    pf_value_hist = self.test_env.get_pf_value_hist()
                    pf_return_hist = self.test_env.get_pf_return_hist()
                    pf_weights_hist = self.test_env.get_pf_weights_hist()
                    agt_action_hist = self.test_env.get_agt_action_hist()
                 
                if done:
                    # These are 
                    print("Finished running backtest. Storing results...")
                    self.ran_backtest = True
                    self.pf_value_hist = pf_value_hist
                    self.pf_return_hist = pf_return_hist
                    self.pf_weights_hist = pf_weights_hist
                    self.agt_action_hist = agt_action_hist 
                    self.pf_performance_stats = self.get_performance_stats()
            
            # Prepare output dictionary
            backtest_results_dict = {}
            backtest_results_dict["value_hist"] = self.pf_value_hist
            backtest_results_dict["return_hist"] = self.pf_return_hist
            backtest_results_dict["weights_hist"] = self.pf_weights_hist
            backtest_results_dict["agt_actions_hist"] = self.agt_action_hist
            backtest_results_dict["performance_stats"] = self.pf_performance_stats
            
            return backtest_results_dict
        
        
    ############################
    ### Save a trained model ###
    ############################
    def save_model(self, model_fname: str = None):
        '''
            Method to save a trained model. Wraps stable_baselines3's BaseAlgorithm.save() method.
            Returns a Boolean for saving success.
            ## Note (22/10/19): Generalize code for path?
        
        if not self.model_is_trained:
            print("ERROR: Cannot save an untrained agent.")
            return False
        else:
            if model_fname == None:
                if self.model_fname == None:
                    # If no filename is provided, default to "pwd/Unnamed_Model_YYMMDDHHMM.zip".
                    fname_suffix = self.get_fname_date_suffix()
                    model_fname = "Unnamed_Model"+fname_suffix
                else: 
                    model_fname = self.model_fname
            
            self.model.save(path = model_fname)
            print(f"Model successfully saved under {model_fname}.zip")
            return True
        '''
        raise NotImplementedError()
        
    ############################
    ### Load a trained model ###
    ############################
    def load_model(self, model_name, new_model_fname: str = None):
        '''
            Load a saved model.
            Note (22/10/19): Use context manager?
        
        # Call SB3 load() method
        self.model = MODELS[model_name].load(new_model_fname)
        # Set loaded model's environment to train_env
        ## NOTE: train_env gets updated during retraining.
        self.model.set_env(self.train_env)
        # Update agent attributes
        self.model_is_trained = True
        self.ran_backtest = False
        
        print(f"Successfully loaded {model_name} model from {new_model_fname}.")
        return True
        '''
        raise NotImplementedError()