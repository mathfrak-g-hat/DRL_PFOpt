##################################################
### Deep RL portfolio optimization environment ###
##################################################
### 2022/11/24, A.J. Zerouali. Additional comments at reset() and step() on 23/04/28
# A modification of FinRL's StockPortfolioEnv class
'''
    To do (22/11/02):
    - Tests for basic PortfolioOptEnv class.
    - Modify computation of weights in step().
    - Modify computation of transaction expenses
      in compute_pf_val_rets().
    - Subclass for dictionary observation spaces,
      important when using distinct feature extraction
      layers.
    - Subclass for live paper-trading with Alpaca.
'''

from __future__ import print_function, division
from builtins import range

# NumPy, Pandas, matplotlib, PyTorch, Gym, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Pytorch imports
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# Gym imports
import gym
from gym import spaces

# Stable Baselines 3
from stable_baselines3.common.vec_env import DummyVecEnv

# 
from drl_pfopt.common.data.data_utils import get_timeframe_info

########################
### Global variables ###
########################
REWARD_TYPES = ["portfolio_return", "portfolio_value", "log_portfolio_return", 
                "portfolio_delta_val"]

DF_X_COLUMNS = ["date", "open", "high", "low", "close", "volume", "tic"]

STATE_STRUCT_KWDS = ["open", "high", "low", "close", "volume", "weights", 
                     "actions", "close_returns", "returns_cov"]

TECHNICAL_INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", 
                        "cci_30", "dx_30", "close_30_sma", "close_60_sma",
                        "turbulence"]

################################################
### Portfolio optimization environment class ###
################################################
### 22/10/25, AJ Zerouali

class PortfolioOptEnv(gym.Env):
    """
        A dynamic portfolio allocation environment.
        
        IMPORTANT (22/10/20): This implementation is made to train SB3 models and run backtests.
                It takes fixed input data and uses the number of trading periods as an attribute.
                For live paper-trading, this class should be fundamentally modified.

        Attributes
        ----------
        :param df_X: pd.DataFrame. 
            Main input dataframe of processed financial data.
        
        :param ret_X: np.ndarray.
             Array of asset returns (close prices) over n_lookback_period. Default: np.empty(shape=0).
        
        :param cov_X: np.ndarray.
            Array of return covariances over n_lookback_prds. Default: np.empty(shape=0).
        
        :attr X: np.ndarray. 
            Main dataset for the environment transitions. Obtained from _set_state_struct() using df_X, ret_X, and cov_X.
        
        :attr state_idx_dict: dict. 
            Dictionary of indices in second dim. of X for the features in state_struct_list and tech_ind_list.
        
        :param n_lookback_prds: int. 
            Number of lookback periods for feature computations (e.g. ret_X, cov_X). Default val: 60.
        
        :param pf_value_ini: float. 
            Initial portfolio value. Default value: 25000.0.
        
        :param reward_type: str. 
            Reward type keyword, specifies the reward function used by the RL agent. Default value: "portfolio_return".
        
        :param state_struct_list: list. 
            List of features to use in environment states, excluding features in tech_ind_list. Default val: ["close", "weights"].
        
        :param tech_ind_list: list. 
            List of technical indicators added to state features. Empty list by default.
        
        :param transaction_cost_pct: float. 
            Transaction cost in percentage. Default val: 0.
            
        :param plot_prds_end_episode: int. 
            Number of trading periods to include in portfolio value and weights
            plots at end of one episode. No plot will be made if plot_prds_end_episode=0,
            and otherwise the number of periods must be lower than N_periods. Default: 0.
        
        * See __init___() for full list and detailed description of class attributes.


        Methods
        -------
        _check_input_data()
            Check that inputs are coherent. See help for details.
        
        _set_state_struct()
            Setup main dataset array X and feature index dictionary. See help for details.
            
        step()
            Usual gym.Env.step() method for state transitions used by agent.
            
        compute_pf_val_rets()
            Compute portfolio rate of return, portfolio value, and other reward functions
            supported by the class. Returns dictionary of reward values.
            
        reset()
            Reset the environment to first date in X.
            
        render()
            Display current state array.
            
        softmax_normalization()
            Compute softmax normalization of input array.
        
        get_pf_value_hist()
            Get dataframe of portfolio values by date.
        
        get_pf_return_hist()
            Get dataframe of portfolio (rates of) returns by date.
        
        get_pf_weights_hist()
            Get dataframe of portfolio asset weights by date.

    """
    
    metadata = {'render.modes': ['human']}

    ###################
    ### Constructor ###
    ###################
    ## AJ Zerouali, 22/11/16
    def __init__(self, 
                 df_X: pd.DataFrame, # Main input dataframe of processed financial data
                 ret_X: np.ndarray = np.empty(shape=0), # Array of asset returns (close prices) over n_lookback_period
                 cov_X: np.ndarray = np.empty(shape=0), # Array of return covariances over n_lookback_prds
                 n_lookback_prds: int = 60, # Number of lookback periods for feature computations (e.g. ret_X, cov_X)
                 pf_value_ini: float = 25000.0, # Initial portfolio value 
                 reward_type: str = "portfolio_return", # Reward type string, 
                 state_struct_list: list = ["close", "weights"], # List of state features other than tech. ind.
                 tech_ind_list: list = [], # List of technical indicators added to state features
                 transaction_cost_pct: float = 0, # Transaction cost in percentage
                 weight_normalization: str = "relu", # Weight computation. Softmax normalization or sum of ReLU outputs
                 plot_prds_end_episode = 0, # Number of trading periods to include in the plot of portfolio value and weights at the end of 
                 ):
        
        '''
        #### I - Assign environment parameters ####
        '''
        ##### This sections assigns the attributes corresponding to the constructor parameters, which are used to:
        ##### (1) Check that the input data is coherent and has the correct format;
        ##### (2) Format the intput data into a single NumPy array.
        
        # Input data attributes
        self.df_X = df_X.copy()
        self.df_X = self.df_X.sort_values(by=['date', 'tic']).reset_index(drop = True) # Sort df_X by date and then by
        self.ret_X = ret_X
        self.cov_X = cov_X
        self.date_hist = df_X.date.unique() # Portfolio trading dates history.
        self.N_periods = len(self.date_hist) # No. of trading periods in input data (i.e. no. of unique dates)
        self.n_lookback_prds = n_lookback_prds # Number of lookback periods for stat computations of state features.
        self.weight_normalization = weight_normalization
        self.timeframe_info_dict = get_timeframe_info(self.date_hist)
        
        # Portfolio global attributes
        self.pf_assets_list = list(self.df_X.tic.unique()) # List of portfolio asset tickers
        self.n_assets = len(self.pf_assets_list) # No. of portfolio assets (i.e. no. of unique tickers)
        self.pf_value_ini = pf_value_ini # Initial portfolio/account value
        self.transaction_cost_pct = transaction_cost_pct # Transaction cost percentage
        self.plot_prds_end_episode = plot_prds_end_episode # 
        
        # State structure attributes
        self.tech_ind_list = tech_ind_list
        self.state_struct_list = state_struct_list
        self.state_incl_actions = ("actions" in state_struct_list) # Include agent actions weights in states
        self.state_incl_weights = ("weights" in state_struct_list) # Include portfolio weights in states
        self.state_incl_clrets = ("close_returns" in state_struct_list) # Include close returns in states (over n_lookback prds)
        self.state_incl_retscov = ("returns_cov" in state_struct_list) # Include return covariances in states (over n_lookback prds)
        
        # Reward type and reward
        self.reward_type = reward_type
        
        
        # Check that input data is coherent         
        data_is_coherent, error_message_list = self._check_input_data()
        if not data_is_coherent: # ABORT CONDITION: Print error messages and raise ValueError().
            print(f"Failed to create portfolio optimization environment:"\
                  f"\n {len(error_message_list)} errors detected.")
            for error in error_message_list: print(error)
            raise ValueError("Input data is incoherent. Please check error messages.")
            #### DEBUG ####
            #return None
            #pass
        
        # Convert data to array X, get state feature index dict. for X, and compute shape of observation space
        self.X, self.state_idx_dict, self.obs_space_shape, _ = self._set_state_struct()
        ### gym.Env attributes
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = self.obs_space_shape) # Required obs space
        self.action_space = spaces.Box(low = 0, high = 1, shape = (self.n_assets,) ) # Required action space
        
        '''
        #### II - Attributes for environment transitions  ####
        '''
        ##### This section records ALL the remaining attributes used by the environment methods.
        ##### These attributes:
        ##### (1) Are assigned the value None in the contructor;
        ##### (2) Are initialized in reset();
        ##### (3) Are updated by step().
        
        # Time period indexing
        self.t_idx = None # Time index for input dataframe periods. Initialized in reset()
        self.at_terminal_prd = None # "At terminal state" Boolean indicator. Init. in reset()
        
        # Portfolio transition attributes
        self.action_cur = None # Action at period t_idx.
        self.pf_value_cur = None # Portfolio value at period t_idx.
        self.pf_return_cur = None # Portfolio return at period t_idx.
        self.pf_weights_cur = None # Portfolio weights at period t_idx.
        self.pf_rwds_dict = None  # Portfolio rewards dict. at time t_idx. See compute_pf_ret_val().
        self.reward = None # Reward at time t_idx.
        self.state = None # State np.ndarray of environment at period t_idx.
               
        # History attributes - Portfolio values, returns, and weights
        self.pf_value_hist = None # Portfolio value history.
        self.pf_return_hist = None # Portfolio return history.
        self.pf_weights_hist = None # Portfolio weights history.
        self.agt_action_hist = None # Agent actions hist.
                
        # For a future version of the environment...
        #self.hmax = hmax # Max no. of stocks to trade
        #self.reward_scaling = reward_scaling # Scaling for training
        
        
    ############################
    ###   Check input data   ###
    ############################
    ## 22/11/16, AJ Zerouali
    def _check_input_data(self):
        '''
            Helper method to check coherence of the input data parameters.
            Goes through the next steps:
            
            1 - Check that df_X contains "date", "open", "high", "low", "close", "volume", and "tic" columns.
            2 - Check that entries of state_struct_list are admissible
            3 - Check that df_X contains the required state features.
            4 - Check that entries of the tech_ind_list are admissible
            5-  Check that df_X contains the required technical indicators.
            6 - If state_struct_list contains "close_returns", check ret_X not None and ret_X.shape.
            7 - If state_struct_list contains "returns_cov", check cov_X not None and cov_X.shape.
            8 - Check reward_type is in list of supported reward types REWARD_TYPES.
            9* - Check N_periods >= 10*n_lookback_prds (should be much larger in fact). Suspended on 22/10/31 (see below)
            10 - Check plot_prds_end_episode < N_periods.
            
            If data_is_coherent is False, PortfolioOptenv.__init__() will print all messages in error_message_list
            and raise a ValueError() exception.
            
            
            :return data_is_coherent: bool. Returns true if and only if error_message_list is empty.
            
            :return error_message_list: list of str. Error messages for aborting the constructor.
            
        '''
        # Initializations
        data_is_coherent = True
        error_message_list = []
        
        df_X_input_cols = list(self.df_X.columns)
        check_close_returns = self.state_incl_clrets
        check_returns_cov = self.state_incl_retscov
        state_struct_list_ = self.state_struct_list.copy()  
                
        ### Remove exceptional keywords in state_struct_list_
        if self.state_incl_actions:
            state_struct_list_.remove("actions")
        if self.state_incl_weights:
            state_struct_list_.remove("weights")
        if check_close_returns:
            state_struct_list_.remove("close_returns")
        if check_returns_cov:
            state_struct_list_.remove("returns_cov")
        
        # 1) Financial data columns
        for x in DF_X_COLUMNS:
            if x not in df_X_input_cols:
                data_is_coherent=False
                error_message = "ERROR: Input dataframe has incorrect format.\n"\
                                "Column \'"+str(x)+"\' is missing from df_X.\n"
                error_message_list.append(error_message)
                
        # Steps (2) and (3)
        for x in state_struct_list_:
            # 2) State structure list and accepted keywords
            if x not in STATE_STRUCT_KWDS:
                data_is_coherent=False
                error_message = "ERROR: State feature \'"+str(x)+"\' in state_struct_list is not "\
                                "supported.\n (See DRL_PFOpt_gymEnv.STATE_STRCT_KWDS)\n"
                error_message_list.append(error_message)
                
            # 3) State structure list and input dataframe
            if x not in df_X_input_cols:
                data_is_coherent=False
                error_message = "ERROR: State feature \'"+str(x)+"\' in state_struct_list missing "\
                                "from input dataframe.\n (See attribute df_X.columns)\n"
                error_message_list.append(error_message)
        
        # Steps (4) and (5) 
        for x in self.tech_ind_list:
            # 4) Technical indicator list and supported keywords
            if x not in TECHNICAL_INDICATORS:
                data_is_coherent=False
                error_message = "ERROR: Technical indicator \'"+str(x)+"\' in tech_ind_list is not "\
                                "supported.\n (See DRL_PFOpt_gymEnv.TECHNICAL_INDICATORS)\n"
                error_message_list.append(error_message)
            
            # 5) Technical indicator list and input dataframe
            if x not in df_X_input_cols:
                data_is_coherent=False
                error_message = "ERROR: Technical indicator \'"+str(x)+"\' in tech_ind_list missing "\
                                "from input dataframe.\n (See attribute df_X.columns)\n"
                error_message_list.append(error_message)
            
        # 6) If required, check close returns array not empty and has correct shape:
        if check_close_returns:
            #if self.ret_X == None:
            if np.array_equal(self.ret_X, np.empty(shape=0)):
                data_is_coherent=False
                error_message = "ERROR: Required close returns argument ret_X is None.\n"
                error_message_list.append(error_message)
            else:
                ret_X_shape = (self.N_periods, self.n_lookback_prds, self.n_assets)
                if self.ret_X.shape != ret_X_shape:
                    data_is_coherent = False
                    error_message = "ERROR: Close returns ret_X array shape is incorrect.\n"\
                                    "Required shape: "+str(ret_X_shape)
                    error_message_list.append(error_message)
        
        # 7) If required, check return covariances array not empty and has correct shape:
        if check_returns_cov:
            # np.empty(shape=0)
            #if self.cov_X == None:
            if np.array_equal(self.cov_X, np.empty(shape=0)):
                data_is_coherent=False
                error_message = "ERROR: Required return covariances argument cov_X is None.\n"
                error_message_list.append(error_message)
            else:
                cov_X_shape = (self.N_periods, self.n_assets, self.n_assets)
                if self.cov_X.shape != cov_X_shape:
                    data_is_coherent = False
                    error_message = "ERROR: Return covariances cov_X array shape is incorrect.\n"\
                                    "Required shape: "+str(cov_X_shape)
                    error_message_list.append(error_message)
                    
        # 8) Reward type supported
        if self.reward_type not in REWARD_TYPES:
            data_is_coherent = False
            error_message = "ERROR: Reward type \'"+self.reward_type+"\' is not supported.\n"\
                            "(See DRL_PFOpt_gymEnv.REWARD_TYPES).\n"
            error_message_list.append(error_message)
            
        # 9) N_periods vs n_lookback_prds # 22/10/28 AJ Zerouali: I'll relax the 10% condition for now
        ### (22/10/31, AJ Zerouali) Relaxing this requirement for now because it conflicts with
        ###   retraining agents when creating retraining environments.
        '''
        if ((self.N_periods -10*self.n_lookback_prds)<0):
            data_is_coherent = False
            error_message = f"ERROR: Number of trading periods is too short compared to no. of lookback"\
                            f"periods.\n N_periods = {self.N_periods} and n_lookback_prds = {self.n_lookback_prds}.\n"\
                            f"n_lookback_prds should not exceed 10% of N_periods.\n"
            error_message_list.append(error_message)
        
        if (self.N_periods <= self.n_lookback_prds):
            data_is_coherent = False
            error_message = f"ERROR: n_lookback_prds >= N_periods"\
                            f"periods.\n N_periods = {self.N_periods} and n_lookback_prds = {self.n_lookback_prds}."
            error_message_list.append(error_message)
        '''
        # 10) Number of trading periods for end of episode plot
        if self.plot_prds_end_episode > self.N_periods:
            data_is_coherent = False
            error_message = f"ERROR: Reward plot_prds_end_episode must be lower than\n"\
                            f"total no. of trading periods N_periods.\n"\
                            f"N_periods = {self.N_periods}< plot_prds_end_episode = {self.plot_prds_end_episode}"
            error_message_list.append(error_message)
        
        return data_is_coherent, error_message_list
    
    ############################
    ### Set state attributes ###
    ############################
    def _set_state_struct(self):
        '''
           Helper function to setup state structure and compute the following env attributes:
           self.X, self.state_idx_dict, and self.obs_space_shape (see return description for details).
           If the actions or portfolio weights are included in the state array, the corresponding
           vectors are added to self.X when calling reset() and step().
           
           CONVENTION: The following convention is used for this class. A state array has shape
           (n_state_rows, n_assets).
           - The columns of a state state array follow the alphabetical order of the tickers.
             n_assets is obtained from df_X['tic'].unique().
           - The rows of a state array are determined by the features in state_struct_list and
             in tech_ind_list. They are organized in the following order:
                 [state_struct_list_, tech_ind_list, actions, weights, close_returns, returns_cov],
             where state_struct_list_ are the features in state_struct_list other than close_returns,
             returns_cov, actions, and weights.
             
            ## IMPORTANT NOTE FOR CODE MODIFICATIONS (22/10/26) ##
            Should any custom state features be added to the dataset self.X, first modify
            _check_input_data() data and the global variable STATE_STRUCT_KWDS. 
           
           
           :return X: np.ndarray of formatted dataset obtained from df_X, ret_X, and cov_X.
                               Has shape (N_periods, n_state_rows, n_assets).
           :return state_idx_dict: dict. Stores indices for each of the state features
                               specified in state_struct_list and tech_ind_list.
           :return obs_space_shape: tuple. Shape of the observation space, equals (n_state_rows, n_assets) 
                               (i.e. shape of np.ndarray of a state at a given period).
           :return state_list: List of state features with the same order as in X.
        '''
        
        # Initializations
        state_idx_dict = {}
        
        add_actions = self.state_incl_actions
        add_weights = self.state_incl_weights
        add_close_returns = self.state_incl_clrets
        add_returns_cov = self.state_incl_retscov
        
        # Ordered state features list
        #### This iterator is used to form the array X and the dict. state_idx_dict
        #### Its ordering is crucial
        
        ## Init. state_features
        state_features = self.state_struct_list.copy()
        
        ### Remove exceptional keywords in state_features
        if add_actions:
            state_features.remove("actions")
        if add_weights:
            state_features.remove("weights")
        if add_close_returns:
            state_features.remove("close_returns")
        if add_returns_cov:
            state_features.remove("returns_cov")
        
        ### Add technical indicators to state_features
        for x in self.tech_ind_list: state_features.append(x)
        
        # Compute no of rows in a state array
        n_state_rows = len(self.state_struct_list)+len(self.tech_ind_list)
        if add_close_returns:
            n_state_rows += self.n_lookback_prds - 1
        if add_returns_cov:
            n_state_rows += self.n_assets - 1
            
        # Get obs_space_shape
        obs_space_shape = (n_state_rows, self.n_assets)
        
        # Init. X
        X = np.zeros(shape = (self.N_periods, n_state_rows, self.n_assets), 
                     dtype = np.float64)
        
        
        # Fill state indexes dictionary and array X
        ### Features from df_X columns
        j_i = 0 # This index is very important for the exceptions below
        for i in range(len(state_features)):
            state_idx_dict[state_features[i]] = j_i
            df_temp = self.df_X.pivot_table(index = "date", columns = "tic", values = state_features[i])
            X[:,j_i,:] = df_temp.to_numpy()
            j_i += 1
            
        ### Actions (already initialized to 0 in X)
        if add_actions:
            state_idx_dict["actions"] = j_i
            state_features.append("actions")
            j_i += 1
        ### Weights (already initialized to 0 in X)
        if add_weights:
            state_idx_dict["weights"] = j_i
            state_features.append("weights")
            j_i += 1
        ### Close returns
        if add_close_returns:
            state_idx_dict["close_returns"] = range(j_i,j_i+self.n_lookback_prds)
            X[:, state_idx_dict["close_returns"], :] = self.ret_X
            state_features.append("close_returns")
            j_i += self.n_lookback_prds
        ### Return covariances
        if add_returns_cov:
            state_idx_dict["returns_cov"] = range(j_i,j_i+self.n_assets)
            X[:, state_idx_dict["returns_cov"], :] = self.cov_X
            state_features.append("returns_cov")
            j_i += self.n_assets
            
        '''
            DEBUG
        '''
        # Check
        if j_i != n_state_rows:
            print("ERROR: There's an index counting issue:")
            print(f"j_i = {j_i},")
            print(f"n_state_rows = {n_state_rows}.")
            print("Should have j_i = n_state_rows")
        
        return X, state_idx_dict, obs_space_shape, state_features
        
    #############################
    ### Gym's render() method ###
    #############################
    def render(self, mode='human'):
        return self.state
    
    ############################
    ### Gym's reset() method ###
    ############################
    # AJ Zerouali, 22/11/15
    def reset(self):
        '''
            PortfolioOptEnv's implementation of Gym's reset() method.
            
            :return state: np.ndarray of current state in specified format.
        '''
        
        # Comment: For coherence, the attribute updates should be in the same
        #          order as in step().
        
        # Reset action and weights. NOTE: Portfolio is initialized with equal weights, although they aren't used.
        self.action_cur = np.zeros(shape = (self.n_assets,), dtype=np.float64)
        self.pf_weights_cur = np.ones(shape=(self.n_assets,), dtype=np.float64)/self.n_assets # TEMPORARY. Use is unclear
        
        
        # Reset time index and "done" attributes
        self.t_idx = 0
        self.at_terminal_prd = (self.t_idx >= (self.N_periods-1))
        
        # Reward
        self.reward = 0.0
        
        # Reset current portfolio value, return and weights
        self.pf_value_cur = self.pf_value_ini 
        self.pf_return_cur = 0 

        # Reset actions and weights in self.X if required
        '''
            23/04/28
            This is incorrect. This part should completely reset
            the actions and weights in the dataset array X, such that:
            - At t_idx the weights/actions should be those of an equally
              weighted portfolio.
            - For t_idx=1,...,(N_periods-1), the weights/actions should be 
              reset to 0.
            The correct instructions are as follows:
        if self.state_incl_actions:
            self.X[self.t_idx,self.state_idx_dict["actions"],:] = self.action_cur
            self.X[self.t_idx+1:,self.state_idx_dict["actions"],:] = 0
        if self.state_incl_weights:
            self.X[self.t_idx+1:,self.state_idx_dict["weights"],:] = 0
            self.X[self.t_idx+1:,self.state_idx_dict["weights"],:] = 0
        '''
        if self.state_incl_actions:
            self.X[self.t_idx,self.state_idx_dict["actions"],:] = 0
        if self.state_incl_weights:
            self.X[self.t_idx,self.state_idx_dict["weights"],:] = 0
        
        # Reset the state
        '''
            23/04/28
            At this very first timestamp t_idx = 0, the environment
            returns the first batch of data.
            The state returned by env.reset() IS USED by the neural
            network.
        '''
        self.state = self.X[self.t_idx,:,:]
        
        # Reset portfolio history attributes
        ## Value history
        self.pf_value_hist = np.zeros(shape = (self.N_periods,), dtype = np.float64)
        self.pf_value_hist[self.t_idx] = self.pf_value_ini
        ## Return history
        self.pf_return_hist = np.zeros(shape = (self.N_periods,), dtype = np.float64)
        ## Asset weights history. NOTE: Column order is that of tickers sorted alphabetically
        self.pf_weights_hist = np.zeros(shape = (self.N_periods, self.n_assets), dtype = np.float64)
        self.pf_weights_hist[self.t_idx,:] = self.pf_weights_cur
        ## Agent actions hist.
        self.agt_action_hist = np.zeros(shape = (self.N_periods, self.n_assets), dtype = np.float64)
        self.agt_action_hist[self.t_idx,:] = self.action_cur
        
        return self.state
    
    
    ###########################################
    ### Compute portfolio value and returns ###
    ###########################################
    def compute_pf_val_rets(self, weights):
        '''
            Helper function to compute the rewards dictionary given the 
            current portfolio weights (softmax of output of action network). 
            Among the quantities computed are the portfolio return and its value
            according to the 'close' price column in the dataframe df_X.
            The full list of values computed is specified in: 
                REWARD_TYPES=["portfolio_return", "portfolio_value", "log_portfolio_return", 
                              "portfolio_delta_val"]
            This method is called by step() AFTER t=self.t_idx is updated, so formally,
            it takes w_(t-1) as input to compute r_t = r(s_(t-1), a_(t-1), s_t),
            where the state s_t contains the close prices x_t = self.X[t_idx, close_idx,:].
            
            ### Notes (22/10/26): 
            ### 1) Should be modified to account for additional
            ###    possible rewards if warranted.
            ### 2) The portfolio rewards are computed according to the close
            ###    prices in this version. This should be modified if the
            ###    intra-day trading periods are used.
            ###    (e.g. when the 'date' col. of df_X accounts for hour/min.)
            
            :param weights: Portfolio weights at period (t_idx-1)
            :return pf_rwds_dict: Dictionary of rewards at period t_idx. Keys
                        of this dictionary are in the REWARD_TYPES list.
        '''
        # Initializations
        date = self.date_hist[self.t_idx] # Date at period t_idx = t
        date_old = self.date_hist[self.t_idx-1] # date at period (t-1)
        X_close = self.df_X[self.df_X['date']==date].close.to_numpy() # Close prices at period t_idx
        X_close_old = self.df_X[self.df_X['date']==date_old].close.to_numpy()# Close prices at period (t_idx-1)
        
        pf_rewards_dict = {}
                
        # Portfolio (rate of) return at period t (no units)
        pf_rewards_dict["portfolio_return"] = np.dot(weights,(X_close-X_close_old)/X_close_old)*(1-self.transaction_cost_pct)
        
        # Log portfolio return at period t (no units).
        pf_rewards_dict["log_portfolio_return"] = np.log(1+pf_rewards_dict["portfolio_return"])
        
        # Portfolio value at period t (in $)
        pf_rewards_dict["portfolio_value"] = self.pf_value_hist[self.t_idx-1]*(1+pf_rewards_dict["portfolio_return"])
        
        # Portfolio value change at period t (in $). The  $ amount gained/lost when rebalancing.
        pf_rewards_dict["portfolio_delta_val"] = pf_rewards_dict["portfolio_value"] - self.pf_value_hist[self.t_idx-1]
        
        
        return pf_rewards_dict
    
    ###########################
    ### Gym's step() method ###
    ###########################
    ## AJ Zerouali, 22/11/16
    def step(self, actions):
        '''
            Gym's step() method for PortfolioOptEnv.
                       
            :return : state, reward, done, {}
        '''
        # Update self.at_terminal_prd attribute
        self.at_terminal_prd = (self.t_idx >= (self.N_periods-1))
        
        # Record current action
        self.action_cur = actions
        
        ## If currently at terminal state ##
        if self.at_terminal_prd:
            
            # Chceck if these are really necessary
            weights = self.softmax_normalization(actions)
            self.pf_rwds_dict = self.compute_pf_val_rets(weights)
            self.reward = self.pf_rwds_dict[self.reward_type]
            
            print("=================================")
            print(f"Initial portfolio value: {self.pf_value_ini}")
            print(f"End portfolio value: {self.pf_value_cur}")
            
            # Compute the Sharpe ratio at last step of episode.
            if self.pf_return_hist.std() != 0:
                sharpe_ratio = ((252**0.5)*self.pf_return_hist.mean())/(self.pf_return_hist.std())
                print(f"Yearly Sharpe ratio at last period: {sharpe_ratio}")
            
            print("=================================")
            
            # Plot portfolio value and weights 
            if self.plot_prds_end_episode>0:
                
                print(f"Plotting portfolio values and weights for last {self.plot_prds_end_episode} periods:")
                # Visualize the decisions made by agents to see if things are changing -- Cheng 11/10/2022 
                figure, axis = plt.subplots(2, 1)
                axis[0].plot(self.pf_value_hist[-self.plot_prds_end_episode:])
                axis[0].set_title("Portfolio value")
                axis[1].plot(self.pf_weights_hist[-self.plot_prds_end_episode:,:])
                axis[1].set_title("Portfolio weights")
                figure.tight_layout()
                  
                # Combine all the operations and display
                plt.show()
            
            return self.state, self.reward, self.at_terminal_prd, {}

        ## If not at end of episode ##
        else:
            # Compute weights w_t at period t
            weights = self.compute_weights(actions) 
            self.pf_weights_cur = weights # NOTE (22/10/21): TEMPORARY. Unclear if needed
            
            # Increment period index to period (t+1)
            self.t_idx += 1
                        
            # Compute portfolio returns and value at period (t+1)
            ### Remark: Formally, r_(t+1) = r(s_t, a_t, s_(t+1))
            self.pf_rwds_dict = self.compute_pf_val_rets(weights)
            self.pf_value_cur = self.pf_rwds_dict["portfolio_value"]
            self.pf_return_cur = self.pf_rwds_dict["portfolio_return"]
            
            # Get reward
            self.reward = self.pf_rwds_dict[self.reward_type]
            
            # Update actions and weights in self.X if required
            ### NOTE: When weights are included in states, the state at period (t+1)
            ###     is written: s_(t+1) = (x_(t+1), w_t, y_(t+1)), where w_t are the 
            ###     PREVIOUS weights, x_(t+1) is the close price at period (t+1),
            ###     and y_(t+1) are the other state features.
            '''
                23/04/28
                This block of instructions indeed assigns the action taken to 
                
            '''
            if self.state_incl_actions:
                self.X[self.t_idx,self.state_idx_dict["actions"],:] = actions # Add actions of period t
            if self.state_incl_weights:
                self.X[self.t_idx,self.state_idx_dict["weights"],:] = weights # Add weights of period t
            
            # Get state at time t_idx (i.e. at period (t+1))
            '''
                23/04/28
                Start the reasoning from here...
            '''
            self.state = self.X[self.t_idx,:,:] # For agent, not used by the env
            
            # Update portfolio history attributes
            self.pf_value_hist[self.t_idx] = self.pf_value_cur 
            self.pf_return_hist[self.t_idx] = self.pf_return_cur
            self.pf_weights_hist[self.t_idx,:] = weights
            self.agt_action_hist[self.t_idx,:] = self.action_cur
            
            return self.state, self.reward, self.at_terminal_prd, {}
    
        
    ###############
    ### Softmax ###
    ###############
    ### 22/09/28, AJ Zerouali
    '''
      This is to compute the portfolio weights. 
      The actions in this environment correspond
      to portfolio weights, but the output of the policy
      networks used is not necessarily normalized.
    '''
    def softmax_normalization(self, actions):
        '''
            :param actions: np.ndarray of actions computed by policy network.
            :return: Softmax normalization of actions.
        '''
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator/denominator
        return softmax_output

    #####################
    ### ReLU function ###
    #####################
    # 22/11/15, AJ Zerouali
    ## Quick computation of ReLU found on StackExchange.
    def ReLU(self, X:np.ndarray):
        return (X+np.abs(X))/2
    
    #######################
    ### Compute weights ###
    #######################
    # 22/11/15, AJ Zerouali
    def compute_weights(self, actions:np.ndarray):
                
        if self.weight_normalization == "relu":
            relu_actions = self.ReLU(actions)
            if relu_actions.sum() == 0.0:
                weights = np.ones(shape = (self.n_assets,), dtype = actions.dtype)/self.n_assets
            else:
                weights = relu_actions/relu_actions.sum()
                
        elif self.weight_normalization == "softmax":
            weights = self.softmax_normalization(actions)
        
        return weights
    
    #############################################
    ### Get portfolio value history dataframe ###
    #############################################
    def get_pf_value_hist(self):
        """
            Get portfolio value history.
            Format: - index: int from 0 to self.N_periods-1
                    - columns = ["date", "daily_value"]
            
            :return df_pf_value_hist: pd.DataFrame of daily portfolio returns over dates in input.
        """ 
        date_list = self.date_hist
        pf_value_hist = self.pf_value_hist
        df_pf_value_hist = pd.DataFrame({'date':date_list,'daily_value':pf_value_hist})
        return df_pf_value_hist
    
    #############################################
    ### Get portfolio return history dataframe ###
    #############################################
    def get_pf_return_hist(self):
        """
            Get portfolio return history.
            Format: - index: int from 0 to self.N_periods-1
                    - columns = ["date", "daily_return"]
            
            :return df_pf_return_hist: pd.DataFrame of daily portfolio returns over dates in input.
        """ 
        date_list = self.date_hist
        pf_return_hist = self.pf_return_hist
        df_return_hist = pd.DataFrame({'date':date_list,'daily_return':pf_return_hist})
        return df_return_hist
    
    ###############################################
    ### Get portfolio weights history dataframe ###
    ###############################################
    def get_pf_weights_hist(self):
        """
            Get portfolio return history.
            Format: - index: int from 0 to self.N_periods-1
                    - columns = ["date", "ticker_1", ..., "ticker_(self.n_assets)"]
            
            :return df_pf_weights_hist: pd.DataFrame of daily weights returns over dates in input.
        """ 
        # Init.
        date_hist = self.date_hist
        pf_assets_list = self.pf_assets_list
        pf_weights_hist = self.pf_weights_hist.copy()
        
        # Make "date" column
        df_pf_weights_hist = pd.DataFrame({"date":date_hist})
        # Add portfolio weight history
        df_pf_weights_hist[pf_assets_list] = pd.DataFrame(pf_weights_hist)
        
        return df_pf_weights_hist
    
    ##########################################
    ### Get agent action history dataframe ###
    ##########################################
    ## AJ Zerouali, 22/11/15
    def get_agt_action_hist(self):
        """
            Get agent actions history.
            Format: - index: int from 0 to self.N_periods-1
                    - columns = ["date", "ticker_1", ..., "ticker_(self.n_assets)"]
            
            :return df_agt_action_hist: pd.DataFrame of agent actions at each period in input dataframe.
        """ 
        # Init.
        date_hist = self.date_hist
        pf_assets_list = self.pf_assets_list
        agt_action_hist = self.agt_action_hist.copy()
        
        # Make "date" column
        df_agt_action_hist = pd.DataFrame({"date":date_hist})
        # Add portfolio weight history
        df_agt_action_hist[pf_assets_list] = pd.DataFrame(agt_action_hist)
        
        return df_agt_action_hist

    ###################
    ### Seed method ###
    ###################
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    #######################
    ### SB3 environment ###
    #######################
    def get_SB3_env(self):
        sb3_env = DummyVecEnv([lambda: self])
        return sb3_env