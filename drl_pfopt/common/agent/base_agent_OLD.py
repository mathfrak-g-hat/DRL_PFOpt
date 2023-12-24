############################################
### Deep RL portfolio optimization agent ###
############################################
### 2022/11/24, A.J. Zerouali
# A modification of FinRL's DRLAgent class
'''
TO DO (22/10/27): 
    - Add docstring to PFOpt_DRL_Agent
    - Comment the code. Describe inputs and outputs of the methods.
    - Implement retraining functionalities
    
'''


from __future__ import annotations, print_function, division
import time
from builtins import range
from datetime import datetime


# np, pd plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gym imports
import gym

# Stable Baselines 3 imports
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from sb3_contrib import RecurrentPPO
from sb3_contrib import TRPO

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BaseModel, BasePolicy

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

# PyFolio imports
import pyfolio
from pyfolio import create_full_tear_sheet
from pyfolio.timeseries import perf_stats

# DRL_PFOpt imports
from drl_pfopt.common.envs import PortfolioOptEnv 
from drl_pfopt.common.data.data_utils import format_returns
from drl_pfopt.common.data.data_utils import get_str_date_format



# Global variables (borrowed from FinRL)
MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO, 
          "recurrent_ppo": RecurrentPPO, "trpo": TRPO,
         }
NOISE = {"normal": NormalActionNoise,
        "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
        }
# Model Parameters (borrowed from FinRL)
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
} 
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
# MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}
MODEL_KWARGS = {"a2c":A2C_PARAMS, "ppo":PPO_PARAMS,
                "ddpg":DDPG_PARAMS, "td3":TD3_PARAMS,
                "sac":SAC_PARAMS,}

## AJ Zerouali
class PFOpt_DRL_Agent:
    """
        Portfolio optimization deep RL agent class.
        
    """
    ##################
    ### Contructor ###
    ##################
    def __init__(self, 
                 train_env: PortfolioOptEnv,
                 test_env: PortfolioOptEnv=None,
                 model_fname: str = None,
                 ):
        
        # Environment attributes
        self.train_env = train_env
        self.test_env = None # Init. by set_test_env()
        
        # Model attributes
        self.model = None # Init. by set_model()
        self.model_fname = model_fname # Init. by contructor or save_model()
        self.model_is_trained = None # Init. by set_model()
        self.model_tb_log_name = None # Init. by train_model()
        self.model_callback = None # Init. by train_model()
        
        # Backtest results
        self.ran_backtest = False 
        self.pf_value_hist = None # Init. by run_backtest()
        self.pf_return_hist = None # Init. by run_backtest()
        self.pf_weights_hist = None # Init. by run_backtest()
        self.agt_action_hist = None # Init. by run_backtest()
        self.pf_performance_stats = None # Init. by run_backtest()
        
        # If test environment passed as parameter, init
        if test_env != None:
            self.set_test_env(test_env)
        
    ####################################
    ### Test environment initializer ###
    ####################################
    ## AJ Zerouali, 22/11/16
    def set_test_env(self, test_env: PortfolioOptEnv):
        '''
            Set test environment. Param. should be a PortfolioOptEnv object,
            sets the self.test_env attribute to an SB3 DummyVecEnv wrapper obj.
            Reinitializes the portfolio history and performance.
        '''
        self.test_env = DummyVecEnv([lambda: test_env])
        
        if self.ran_backtest:
            print("Deleting previous backtest results and history...")
            # Re-initialize portfolio history attributes
            ## Comment: Idea is to avoid mistakes if the test environment 
            ##          is modfied right after running a backtest.
            self.ran_backtest = False # Init. by run_backtest()
            self.pf_value_hist = None # Init. by run_backtest()
            self.pf_return_hist = None # Init. by run_backtest()
            self.pf_weights_hist = None # Init. by run_backtest()
            self.agt_action_hist = None # Init. by run_backtest()
            self.pf_performance_stats = None # Init. by run_backtest()
        
        '''
        # If we only want to set a test_env after training the agent
        if not self.model_is_trained:
            print("ERROR: Cannot set a test environment for an untrained agent.")
        else:
            self.test_env = DummyVecEnv([lambda: test_env])
        '''
        
    
    #################################
    ### Deep RL agent initializer ###
    #################################
    ### 22/11/15, AJ Zerouali
    def set_model(self,
                  model_name,
                  policy="MlpPolicy",
                  policy_kwargs=None,
                  model_kwargs=None,
                  verbose=1,
                  seed=None,
                  tensorboard_log=None,
                  model_logger = None,
                 ):
        
        '''
            Initialize the agent's self.model attribute with an SB3 algorithm.
            Wraps the stable_baselines3 model constructor.
        '''
    
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        if "action_noise" in model_kwargs:
            n_actions = self.train_env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(model_kwargs)
        
        # Instantiate SB3 Algorithm
        self.model = MODELS[model_name](policy=policy,
                                        env=self.train_env,
                                        tensorboard_log=tensorboard_log,
                                        verbose=verbose,
                                        policy_kwargs=policy_kwargs,
                                        seed=seed,
                                        **model_kwargs,
                                       )
        # Set logger
        if model_logger != None:
            self.model.set_logger(model_logger)
        
        # Initialize training/testing Boolean attributes
        self.model_is_trained = False
        self.ran_backtest = False
        
        print(f"{model_name} with given parameters has been successfully created.")
        return True
    
    #############################
    ### Agent training method ###
    #############################
    # 22/11/13, AJ Zerouali
    # Updated param to n_train_rounds instead of total timesteps
    # total_timesteps = n_train_rounds*train_env.N_periods.
    def train_model(self, tb_log_name: str, 
                    n_train_rounds: int = 20,
                    save_model = False, 
                    model_fname: str = None,
                    progress_bar: bool = False,
                    model_callback: BaseCallback = None,
                   ):
        '''
            Trains the model attribute according to the given parameters.
            Wrapper for stable_baselines3's BaseAlgorithm.learn() method.
            Notes (22/10/19): 
            1) Add more training attributes to this agent class.
            2) Use context manager below for training?
            
            :param tb_log_name: Log name for TensorBoard
            :param n_train_rounds: Number of episodes for training.
                Determines total number of times that train_env.step() is
                called, with total_timesteps = n_train_rounds*train_env.N_periods.
            :param save_model: Whether or not to call save_model() at end of training.
            :param model_fname: Filename for saved zip file containing SB3 model.
            :param progress_bar: Whether or not to display the progress bar during training.
            :param model_callback: BaseCallback object (or list thereof) to call during training.
            
        '''
        # Init. training attributes
        self.model_tb_log_name = tb_log_name
        self.model_callback = model_callback # Note (22/11/11): Made this into a BaseCallback param
        
        # Get total timesteps from no. of training rounds
        total_timesteps = n_train_rounds*self.train_env.N_periods
        
        # Call learn()
        self.model.learn(total_timesteps=total_timesteps,
                         tb_log_name=self.model_tb_log_name,
                         progress_bar = progress_bar,
                         callback=self.model_callback,
                         )
        self.model_is_trained = True
        
        if save_model:
            self.save_model(model_fname= model_fname)
    
    ############################
    ### Save a trained model ###
    ############################
    def save_model(self, model_fname: str = None):
        '''
            Method to save a trained model. Wraps stable_baselines3's BaseAlgorithm.save() method.
            Returns a Boolean for saving success.
            ## Note (22/10/19): Generalize code for path?
        '''
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
        
    ############################
    ### Load a trained model ###
    ############################
    def load_model(self, model_name, new_model_fname: str = None):
        '''
            Load a saved model.
            Note (22/10/19): Use context manager?
        '''
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
            

    ####################################
    ###     CRUCIAL: Run Backtest    ###
    ####################################
    ## 22/11/15, AJ Zerouali
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
            N_periods = self.test_env.get_attr(attr_name = "N_periods")[0]
            
            # Main loop
            for i in range(N_periods): # Needs to be changed
                
                # Compute action corresponding to current observation
                action, state_ = self.model.predict(state, deterministic=deterministic)
                
                # Execute the step() function
                ### WARNING (22/10/19): I don't see state_ being reused here. C'est normal ça?
                state, rewards, dones, info = self.test_env.step(action)
                
                
                # Store history
                if i == (N_periods - 1): # 23/05/04
                    # Important: i = (N_periods-2) is when the environment's t_idx is updated to (N_periods-1)
                    # The assignment is done here because SB3 automatically resets the environment to t_idx = 0
                    pf_value_hist = self.test_env.env_method(method_name="get_pf_value_hist")[0]
                    pf_return_hist = self.test_env.env_method(method_name="get_pf_return_hist")[0]
                    pf_weights_hist = self.test_env.env_method(method_name="get_pf_weights_hist")[0]
                    agt_action_hist = self.test_env.env_method(method_name="get_agt_action_hist")[0]
                 
                if dones[0]:
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
        
    ######################################
    ### Get Backtest performance stats ###
    ######################################
    ## AJ Zerouali, 22/11/15
    def get_performance_stats(self):
        '''
            Wrapper for PyFolio's pyfolio.timeseries.perf_stats.
            Called by run backtest, outputs a pd.DataFrame instead of 
            a pd.Series.
        '''
        
        # Format the returns for PyFolio
        df_backtest_returns_ = self.pf_return_hist.copy()
        str_date_format = get_str_date_format(list(df_backtest_returns_.date.unique()))
        df_backtest_returns_["date"] = pd.to_datetime(df_backtest_returns_["date"], 
                                                      format=str_date_format) 
        df_backtest_returns_ = df_backtest_returns_.set_index('date')
        
        # Get the performance stats series from PyFolio's perf_stats
        series_performance_stats = perf_stats(returns=df_backtest_returns_["daily_return"])
        del df_backtest_returns_
        
        # Format output
        df_performance_stats = pd.DataFrame(series_performance_stats)
        df_performance_stats = df_performance_stats.rename(columns = {0:"Value"})
        
        return df_performance_stats
 
        
    
    #############################
    ### Save Backtest results ###
    #############################
    ##  AJ Zerouali, 22/11/16
    def save_backtest_results(self, pf_vals_fname: str = None, 
                              pf_returns_fname: str = None, 
                              pf_weights_fname: str = None,
                              pf_performance_stats_fname: str = None,
                             ):
        '''
            Saves results of last backtest performed by the agent.
            Exports portfolio value and portfolio weight histories to
            CSV files.
            Date in YYMMDDHHMM format is automatically appended to the
            filenames provided, as well as the .csv extension.
            Unless filename strings and path are provided, the CSV
            files are saved as:
            - "Backtest_PFValue_hist_YYMMDDHHMM.csv"
            - "Backtest_PFReturns_hist_YYMMDDHHMM.csv"
            - "Backtest_PFWeights_hist_YYMMDDHHMM.csv"
            - "Backtest_Perf_Stats_YYMMDDHHMM.csv"
            
            
            :param pf_vals_fname: Portfolio values CSV filename
            :param pf_returns_fname: Portfolio returns CSV filename
            :param pf_weights_fname: Portfolio weights CSV filename
            :param pf_performance_stats_fname: Portfolio performance stats filename.
            
        ''' 
        if not self.ran_backtest:
            print("ERROR: No results to save, no backtest was run.")
            return False
        else:
            # Get fname suffix
            fname_suffix = self.get_fname_date_suffix()
            # Check and modify filenames
            # Portfolio values history
            if pf_vals_fname == None:
                pf_vals_fname = "Backtest_PFValue_hist"+fname_suffix+".csv"
            else:
                pf_vals_fname += fname_suffix+".csv"
            # Portfolio returns history
            if pf_returns_fname == None:
                pf_returns_fname = "Backtest_PFReturns_hist"+fname_suffix+".csv"
            else:
                pf_returns_fname += fname_suffix+".csv"
            # Portfolio weights history
            if pf_weights_fname == None:
                pf_weights_fname = "Backtest_PFWeights_hist"+fname_suffix+".csv"
            else:
                pf_weights_fname += fname_suffix+".csv"
            # Portfolio performance stats
            if pf_performance_stats_fname == None:
                pf_performance_stats_fname = "Backtest_Perf_Stats"+fname_suffix+".csv"
            else:
                pf_performance_stats_fname += fname_suffix+".csv"
            
            # Export portfolio value and weight histories to CSV
            self.pf_value_hist.to_csv(pf_vals_fname)
            self.pf_return_hist.to_csv(pf_returns_fname)
            self.pf_weights_hist.to_csv(pf_weights_fname)
            self.pf_performance_stats.to_csv(pf_performance_stats_fname)
            
            return True
        
    
    #############################
    ### Plot Backtest results ###
    #############################
    ## 22/11/16, AJ Zerouali
    ## Wraps pyfolio.create_full_tearsheet
    def plot_backtest_results(self, df_benchmark_returns: pd.DataFrame, benchmark_name: str = "Benchmark_returns"):
        '''
            Wrapper for pyfolio.create_full_tear_sheet().
            Requires a pd.DataFrame of benchmark returns with exactly two columns: one labelled "date" and the other containing the returns.
            ### Note (22/11/16): Thought of adding code to download 
            ###       benchmark returns using a ticker. Might not be a good
            ###       idea because of data frequency.
            ###       See following helper function DRL_PFOpt_Utils submodule:
            ###       - get_benchmark_prices_and_returns() to download a ticker benchmark.
            ###       - get_eq_wts_benchmark() to obtain equal weights strategy benchmark results.
            
            :param df_benchmark_returns: pd.DataFrame of benchmark returns by date.
            :param benchmark_name: str. Name of benchmark to display in plots.
        '''
        if not self.ran_backtest:
            print("ERROR: No results to plot, no backtest was run.")
            return False
        else:
            # Check if benchmark dataframe has correct format
            if len(list(df_benchmark_returns.columns))!=2 or ("date" not in list(df_benchmark_returns.columns)):
                print("ERROR: df_benchmark_returns does not have the correct format.")
                print("df_benchmark_returns must have exactly two columns, one \n labeled \"date\" and the other containing benchmark returns")
                return False
            
            else:
                # Format backtest and benchmark returns according to pyfolio's format
                df_pf_returns_, df_benchmark_returns_ = format_returns(self.pf_return_hist, df_benchmark_returns, benchmark_name)
                # Call pyfolio's tearsheet plotting 
                with pyfolio.plotting.plotting_context(font_scale=1.1):
                    '''22/10/20: Temporary, should be changed.'''
                    create_full_tear_sheet(returns=df_pf_returns_['daily_return'], 
                                           benchmark_rets=df_benchmark_returns_[benchmark_name],
                                           set_context=False)
                return True
        
    ###################################
    ### Create filename date suffix ###
    ###################################
    def get_fname_date_suffix(self):
        '''
            Creates date suffix for filenames in YYMMDDHHMM format.
            Called when saving models and backtest results.
            
            :return fname_suffix: "_YYMMDDHHMM" str
        '''
        # Set formatted date str
        time_now_str = str(datetime.now())
        fname_suffix = "_"
        char_idxs = [2,3,5,6,8,9,11,12,14,15]
        for i in char_idxs: fname_suffix+=time_now_str[i]
        
        return fname_suffix


# Borrowed from FinRL...
class TensorboardCallback(BaseCallback):
    
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except BaseException:
            self.logger.record(key="train/reward", value=self.locals["reward"][0])
        return True
