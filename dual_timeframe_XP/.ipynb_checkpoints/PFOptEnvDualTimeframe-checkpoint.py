##################################################
### Deep RL portfolio optimization environment ###
##################################################
### 2023/05/18, A.J. Zerouali

'''
    A gym environment for portfolio optimization
    with reinforcement learning.
    This version supports data organized according
    to two timeframes, one for trading and one for 
    downloaded data.
    
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
from datetime import datetime, timedelta

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
from drl_pfopt.common.envs import PortfolioOptEnv

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
                        "turbulence", "vix"] # Added vix. Does it even belong there?

################################################
################################################


class PFOptDualTFEnv(PortfolioOptEnv):
    """
        A dynamic portfolio allocation environment.
        
        IMPORTANT (22/10/20): This implementation is made to train SB3 models and run backtests.
                It takes fixed input data and uses the number of trading periods as an attribute.
                For live paper-trading, this class should be fundamentally modified.

        Attributes
        ----------
        :param df_X: pd.DataFrame. 
            Main input dataframe of processed financial data. ### TO BE MODIFED
        
        :param ret_X: np.ndarray.
             Array of asset returns (close prices) over n_lookback_period. Default: np.empty(shape=0). ### TO BE MODIFED
        
        :param cov_X: np.ndarray.
            Array of return covariances over n_lookback_prds. Default: np.empty(shape=0). ### TO BE MODIFED
        
        :attr X: np.ndarray. 
            Main dataset for the environment transitions. Obtained from _set_state_struct() using df_X, ret_X, and cov_X.
        
        :attr state_idx_dict: dict. 
            Dictionary of indices in second dim. of X for the features in state_struct_list and tech_ind_list. ### TO BE MODIFED
        
        :param n_lookback_prds: int. 
            Number of lookback periods for feature computations (e.g. ret_X, cov_X). Default val: 60. ### TO BE MODIFED
        
        :param pf_value_ini: float. 
            Initial portfolio value. Default value: 25000.0.
        
        :param reward_type: str. 
            Reward type keyword, specifies the reward function used by the RL agent. Default value: "portfolio_return".
        
        :param state_struct_list: list. 
            List of features to use in environment states, excluding features in tech_ind_list. Default val: ["close", "weights"].
        
        :param tech_ind_list: list. 
            List of technical indicators added to state features. Empty list by default. ### TO BE MODIFED
        
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
    ## AJ Zerouali, 23/04/22
    '''
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
    def __init__(self, 
                 data_dict: dict, # 23/04/22 NEW
                 pf_value_ini: float = 25000.0, # Initial portfolio value 
                 reward_type: str = "portfolio_return", # Reward type string, 
                 state_struct_list: list = ["close", "weights"], # List of state features other than tech. ind.
                 tech_ind_list: list = [], # List of technical indicators added to state features
                 transaction_cost_pct: float = 0, # Transaction cost in percentage
                 weight_normalization: str = "softmax", # Weight computation. 
                 plot_prds_end_episode: int = 0, # Number of trading periods to include in the plot of portfolio value and weights at the end of 
                 ):
        
        '''
        #### I - Assign environment parameters ####
        '''
        ##### This sections assigns the attributes corresponding to the constructor parameters, which are used to:
        ##### (1) Check that the input data is coherent and has the correct format;
        ##### (2) Format the intput data into a single NumPy array.
        
        # Input data attributes
        self.df_X = data_dict["df"].copy()  ### 23/04/22, NEW
        self.df_X = self.df_X.sort_values(by=['date', 'tic']).reset_index(drop = True) # Sort df_X by date and then by ticker
        self.ret_X = data_dict["np_close_returns"]  ### 23/04/22, NEW
        self.cov_X = data_dict["np_returns_covs"]  ### 23/04/22, NEW
        self.data_timeframe_info_dict = data_dict["data_timeframe_info"] ### 23/04/22, NEW
        self.trade_timeframe_info_dict = data_dict["trade_timeframe_info"] ### 23/04/22, NEW
        self.n_lookback_prds = self.trade_timeframe_info_dict["N_lookback_trade_prds"] #23/04/22, Number of lookback trading periods for stat computations of state features. ### NEW
        self.n_intratrading_timestamps = self.trade_timeframe_info_dict["n_intratrading_timestamps"] # No. of data timestamps between 2 trading prds 23/04/22, NEW
        self.trading_data_schedule = self.trade_timeframe_info_dict["trading_data_schedule"]
        self.date_hist = self.trade_timeframe_info_dict["trade_timestamp_list"] # Portfolio trading dates history. ### 23/04/22, NEW
        #self.date_hist = [self.trading_data_schedule[self.date_hist[0]][-self.n_intratrading_timestamps]]+ self.date_hist # Add initial, 23/05/03
        ## I don't know if this will work, will have to correct the date index below (reset, step, state struct)
        self.N_periods = len(self.trade_timeframe_info_dict["trade_timestamp_list"]) # No. of trading periods in input data ### 23/05/03, NEW, MIGHT NOT WORK
        self.tech_ind_list = tech_ind_list
        self.weight_normalization = weight_normalization
        # Assign weight computation function
        if self.weight_normalization == "identity":
            self.compute_weights = self.identity_normalization
        elif self.weight_normalization == "softmax":
            self.compute_weights = self.softmax_normalization
        elif self.weight_normalization == "relu":
            self.compute_weights = self.relu_normalization
        else:
            raise NotImplememntedError(f"Weight normalization {weight_normalization} is not supported.")
        
        
        '''
            ### IMPORTANT: See also the list of technical indicators
        '''
        
        # Portfolio global attributes
        self.pf_assets_list = list(self.df_X.tic.unique()) # List of portfolio asset tickers
        self.n_assets = len(self.pf_assets_list) # No. of portfolio assets (i.e. no. of unique tickers)
        self.pf_value_ini = pf_value_ini # Initial portfolio/account value
        self.transaction_cost_pct = transaction_cost_pct # Transaction cost percentage
        self.plot_prds_end_episode = plot_prds_end_episode # 
        
        # State structure attributes
        #self.tech_ind_list = tech_ind_list  ### TO BE MODIFED - compare with technical indicators of data_dict
        self.state_struct_list = state_struct_list
        self.state_incl_actions = ("actions" in state_struct_list) # Include agent actions weights in states
        self.state_incl_weights = ("weights" in state_struct_list) # Include portfolio weights in states
        self.state_incl_clrets = ("close_returns" in state_struct_list) # Include close returns in states (over n_lookback prds)
        self.state_incl_retscov = ("returns_cov" in state_struct_list) # Include return covariances in states (over n_lookback prds)
        
        # Reward type and reward
        self.reward_type = reward_type
        
        
        # Check that input data is coherent       ### TO BE MODIFED   
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
        '''
            ####################################
            ### CRUCIAL: SET STATE STRUCTURE ###
            ####################################
            23/04/22
            
            One of the keys of the data dictionary produced by
            the FeatureEngDualTF class is "feature_shape_dict",
            which contains the shape of each feature (OHLCV,
            technical indicators, close returns and return covs).
            The purpose of this dictionary is to have an iterator
            to easily build the state_idx_dict (state index dictionary).
            This attribute is used compute obs_space_shape and to
            compose the main data array X.
        '''
        self.X, self.state_idx_dict, self.obs_space_shape, _\
            = self._set_state_struct(data_dict["feature_shape_dict"])  ### 23/04/22, NEW
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
            9 - Check plot_prds_end_episode < N_periods.
            
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
            #if self.ret_X == None: ### 23/04/30 New feature eng. always returns a close returns array
            if np.array_equal(self.ret_X, np.empty(shape=0)):
                data_is_coherent=False
                error_message = "ERROR: Required close returns argument ret_X is None.\n"
                error_message_list.append(error_message)
            else:
                #ret_X_shape = (self.N_periods, self.n_lookback_prds, self.n_assets) ### 23/04/30 I want this to return an error
                ret_X_shape = (self.N_periods, self.n_lookback_prds*self.n_intratrading_timestamps, self.n_assets) ### 23/04/30 Correct assignment
                if self.ret_X.shape != ret_X_shape:
                    data_is_coherent = False
                    error_message = "ERROR: Close returns ret_X array shape is incorrect.\n"\
                                    "Required shape: "+str(ret_X_shape)
                    error_message_list.append(error_message)
        
        # 7) If required, check return covariances array not empty and has correct shape:
        if check_returns_cov:
            # np.empty(shape=0)
            #if self.cov_X == None: ### 23/04/30 New feature eng. always returns a returns covariances array
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
            
        # 9) Number of trading periods for end of episode plot
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
    # AJ Zerouali 23/04/26
    ## 
    def _set_state_struct(self, feature_shape_dict: dict):
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
           
           :param feature_shape_dict: Dictionary of feature shapes built by the 
                   feature engineer. Called using data_dict["feature_shape_dict"]
                   in the constructor.
           
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

        df_X = self.df_X
        date_hist = self.trade_timeframe_info_dict["trade_timestamp_list"]#self.date_hist #23/05/03
        data_schedule =  self.trade_timeframe_info_dict["trading_data_schedule"] 
        n_intratrading_timestamps = self.n_intratrading_timestamps
        N_periods = self.N_periods
        n_assets = self.n_assets
        np_close_returns = self.ret_X
        np_returns_covs = self.cov_X

        # Ordered state features list
        #### This iterator is used to form the array X and the dict. state_idx_dict
        #### the next instructions re-order this list

        ## Init. state_features
        df_state_features = self.state_struct_list.copy()
        except_state_features = []

        ### Remove exceptional keywords from state_features
        if self.state_incl_actions:
            df_state_features.remove("actions")
            except_state_features.append("actions")
            # There is only one action per trading timestamp
            feature_shape_dict["actions"] = (1, n_assets)
        if self.state_incl_weights:
            df_state_features.remove("weights")
            except_state_features.append("weights")
            # There is only one action per trading timestamp
            feature_shape_dict["weights"] = (1, n_assets)
        if self.state_incl_clrets:
            df_state_features.remove("close_returns")
            except_state_features.append("close_returns")
        if self.state_incl_retscov:
            df_state_features.remove("returns_cov")
            except_state_features.append("returns_cov")

        ### Add technical indicators to df_state_features
        df_state_features = df_state_features+self.tech_ind_list
        #for x in self.tech_ind_list: df_state_features.append(x)

        # Init. main array
        X = np.empty(shape=(N_periods, 0, n_assets))
        
        # Fill state_idx_dict and X using feature_shape_dict
        ## Init. n_state_rows. Final value of this variable is known AFTER adding all features
        n_state_rows = 0
        ## Loop over features in dataframe 
        for feature in df_state_features:
            # Number of rows for current feature
            n_rows_temp = feature_shape_dict[feature][0]
            # Init. temp array
            X_temp = np.empty(shape=(N_periods, n_rows_temp, n_assets))
            # Init. temp dataframe
            df_temp = df_X.pivot_table(index = "date", columns = "tic", values = feature)
            # Indices of current feature
            state_idx_dict[feature] = \
            range(n_state_rows, n_rows_temp+n_state_rows)
            # Update number of rows in data array
            n_state_rows += n_rows_temp

            # Fill X_temp
            for i in range(N_periods):
                # Get last n_rows_temp data timestamps from trading_schedule_dict
                ## data_schedule[date_hist[i]] contains n_intratrading_timestamps*N_lookback_trade_prds
                ## timestamps, and only the values of the previous trading period is needed here.
                data_timestamp_list_i = data_schedule[date_hist[i]][-n_rows_temp:]
                # Reshape 
                X_temp[i,:,:] = df_temp[df_temp.index.isin(data_timestamp_list_i)].to_numpy()

            # Concatenate X_temp and X
            X = np.concatenate([X,X_temp], axis = 1)

        ## Loop over "exceptional" features. There are only 4 of those.
        for feature in except_state_features:
            # Number of rows for current feature
            n_rows_temp = feature_shape_dict[feature][0]
            # Init. temp array with zeros (for weights and actions in particular)
            X_temp = np.zeros(shape=(N_periods, n_rows_temp, n_assets))
            # Indices of current feature
            state_idx_dict[feature] = \
            range(n_state_rows, n_rows_temp+n_state_rows)
            # Update number of rows in data array
            n_state_rows += n_rows_temp

            if feature == "close_returns":
                X_temp = np_close_returns
            elif feature == "returns_cov":
                X_temp = np_returns_covs

            # Concatenate X_temp and X
            X = np.concatenate([X,X_temp], axis = 1)

        # Get obs_space_shape
        obs_space_shape = (n_state_rows, n_assets)#self.n_assets) ### CLASS VERSION

        # Ordered feature list
        for x in except_state_features: df_state_features.append(x) 

        # DEBUG
        if n_state_rows != X.shape[1]:
            print(f"ERROR: There's a mismatch between obs_state_shape and X.shape:")
            print(f"X.shape = {X.shape}; obs_space_shape = {obs_space_shape}")
            print(f"n_state_rows = {n_state_rows} should be equal to obs_state_shape[0]"\
                  f" and X.shape[1].")


        return X, state_idx_dict, obs_space_shape, df_state_features
        
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

        '''
        ################################################
        ##### INITIAL ACTION AND PORTFOLIO WEIGHTS #####
        ################################################
        '''
        self.action_cur = np.zeros(shape = (self.n_assets,), dtype=np.float64)
        self.pf_weights_cur = np.ones(shape=(self.n_assets,), dtype=np.float64)/self.n_assets
        
        '''
        #######################################################
        ##### RESET TIME INDEX AND TERMINAL STATE BOOLEAN #####
        #######################################################
        '''
        # Reset time index and "done" attributes
        self.t_idx = 0
        self.at_terminal_prd = (self.t_idx == (self.N_periods-1))

        '''
        ################################
        ##### RESET CURRENT REWARD #####
        ################################
        # Reward
        '''
        self.reward = 0.0

        '''
        ####################################################
        ##### RESET CURRENT PORTFOLIO VALUE AND RETURN #####
        ####################################################
        # Reset current portfolio value and return
        '''
        self.pf_value_cur = self.pf_value_ini
        self.pf_return_cur = 0.0
        
        '''
        ###################################################
        ##### RESET WEIGHTS AND ACTIONS IN DATA ARRAY #####
        ###################################################
        '''
        if self.state_incl_actions:
            # Store actions at beginning of period t_idx = 0
            self.X[self.t_idx, self.state_idx_dict["actions"], :] = self.action_cur
            # Reset actions to 0 for periods i>t_idx
            self.X[self.t_idx+1:, self.state_idx_dict["actions"], :] = 0.0
        if self.state_incl_weights:
            # Store weights at beginning of t_idx = 0
            self.X[self.t_idx, self.state_idx_dict["weights"], :] = self.pf_weights_cur
            # Reset weights to 0 for periods i>t_idx
            self.X[self.t_idx+1:, self.state_idx_dict["weights"], :] = 0.0
        
        '''
        ###################################
        ##### INITIAL PORTFOLIO STATE #####
        ###################################
        '''
        self.state = self.X[self.t_idx,:,:]

        '''
        ##############################################
        ##### RESET PORTFOLIO HISTORY ATTRIBUTES #####
        ##############################################
        COMMENT: The self.pf_xxx_hist attributes have an additional
                entry to account for the initialization of the
                portfolio.
        
        ## Value history
        self.pf_value_hist = np.zeros(shape = (self.N_periods+1,), dtype = np.float64)
        self.pf_value_hist[self.t_idx] = self.pf_value_cur
        ## Return history
        self.pf_return_hist = np.zeros(shape = (self.N_periods+1,), dtype = np.float64)
        self.pf_return_hist[self.t_idx] = self.pf_return_cur
        ## Agent action(s) history
        self.agt_action_hist =  np.zeros(shape = (self.N_periods+1, self.n_assets), dtype = np.float64)
        self.agt_action_hist[self.t_idx,:] = self.action_cur
        ## Asset weights history. NOTE: Column order is that of tickers sorted alphabetically
        self.pf_weights_hist = np.zeros(shape = (self.N_periods+1, self.n_assets), dtype = np.float64)
        self.pf_weights_hist[self.t_idx,:] = self.pf_weights_cur
        '''
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
        # Function version
        df_X = self.df_X
        date_hist = self.date_hist # 23/05/03
        #trade_timestamp_list = self.trade_timeframe_info_dict["trade_timestamp_list"] # 23/05/03, Equivalent
        n_intratrading_timestamps = self.n_intratrading_timestamps
        trading_data_schedule = self.trade_timeframe_info_dict["trading_data_schedule"]
        
        # Initializations
        '''
            For t_idx = t, date is the timestamp of CURRENT portfolio rebalancing,
            that is, the timestamp at which period (t+1) BEGINS.
        '''
        date = date_hist[self.t_idx] # 23/05/03
        '''
            For t_idx = t, date_old is the timestamp of PREVIOUS portfolio rebalancing,
            that is, the timestamp at which period t BEGINS.
        '''
        date_old = trading_data_schedule[date][-n_intratrading_timestamps] # date at period (t-1)
        X_close_old = df_X[df_X['date']==date_old].close.to_numpy()# Close prices at period (t_idx-1)
        '''
            For t_idx = (N_periods-1), there are no close prices for date_hist[t_idx],
            so we take the last close prices of period (N_periods-1).
            The portfolio value, return, weights and actions are not added to the
            portfolio history.
        '''
        if self.at_terminal_prd:
            last_timestamp = trading_data_schedule[date][-1]
            X_close = df_X[df_X['date']==last_timestamp].close.to_numpy() # Close prices at period t_idx
        else:
            X_close = df_X[df_X['date']==date].close.to_numpy() # Close prices at period t_idx
        
        pf_rewards_dict = {}
                
        # Portfolio (rate of) return at period t (no units)
        pf_rewards_dict["portfolio_return"] = np.dot(weights,(X_close-X_close_old)/X_close_old)\
                                                *(1-self.transaction_cost_pct)
        
        # Log portfolio return at period t (no units).
        pf_rewards_dict["log_portfolio_return"] = np.log(1+pf_rewards_dict["portfolio_return"])
        
        # Portfolio value at period t (in $)
        pf_rewards_dict["portfolio_value"] = self.pf_value_cur*(1+pf_rewards_dict["portfolio_return"])
        
        # Portfolio value change at period t (in $). The  $ amount gained/lost when rebalancing.
        pf_rewards_dict["portfolio_delta_val"] = pf_rewards_dict["portfolio_value"] - self.pf_value_cur
        
        
        return pf_rewards_dict
    
    ###########################
    ### Gym's step() method ###
    ###########################
    ## AJ Zerouali, 22/11/16
    ## C Chi, 22/10/11
    def step(self, actions):
        '''            
            Gym's step() method for PortfolioOptEnv.
                       
            :return : state, reward, done, {}
        '''
        
        '''
            ###################################
            ##### UPDATE TERMINAL BOOLEAN #####
            ###################################
            CRUCIAL: Be careful with the index shift.
            OLD VERSION: date_hist[t_idx] 
        '''
        # Update self.at_terminal_prd attribute
        self.at_terminal_prd = (self.t_idx == (self.N_periods-1))
        if self.t_idx == 0: self.reset_hist_attr()
        
        '''
            #################################
            ##### UPDATE CURRENT ACTION #####
            #################################
        # Record current action
        '''
        self.action_cur = actions
        
        ## If currently at terminal state ##
        #if self.at_terminal_prd: ### CLASS VERSION
        if self.at_terminal_prd:
            '''
            # DEBUG
            print(f"End of pf_date_hist: t_idx = {t_idx};")
            timestamp = data_dict["trade_timeframe_info"]["trade_timestamp_list"][t_idx]
            print(f"pf_date_hist[t_idx] = {timestamp}")
            '''
            '''
                ############################
                #### WEIGHT COMPUTATION ####
                ############################
            # Compute weights w_t at period t
            '''
            weights = self.compute_weights(actions)
            self.pf_weights_cur = weights
                
            '''
                ##############################################
                #### COMPUTE PF VALUE, RETURN AND REWARDS ####
                ##############################################
            # Compute portfolio returns and value at en of period (N_periods-1)
            # These values are NOT stored in the history attributes.
            '''
            self.pf_rwds_dict = self.compute_pf_val_rets(weights)
            self.pf_value_cur = self.pf_rwds_dict["portfolio_value"] # Pf value at the END of period (N_periods-1)
            self.pf_return_cur = self.pf_rwds_dict["portfolio_return"] # Pf return at the END beginning of period (N_periods-1)
            self.reward = self.pf_rwds_dict[self.reward_type]
            # Record last pf value, returns, weights and actions, 
            self.pf_value_hist[self.N_periods] = self.pf_value_cur # Shifted index, pf value at beginning of period
            self.pf_return_hist[self.N_periods] = self.pf_return_cur # Shifted index, pf return at beginning of period
            self.pf_weights_hist[self.N_periods,:] = weights # Previous period
            self.agt_action_hist[self.N_periods,:] = self.action_cur # Previous period
         
            # We do not modify the state 
            
            
            ########################
            #### RESULTS ####
            ########################
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
        ### This is valid for t_idx = 0, ..., (N_periods-2)
        else:
            
            '''
                ############################
                #### WEIGHT COMPUTATION ####
                ############################
            # Compute weights w_t at period t
            '''
            weights = self.compute_weights(actions)
                
            '''
                #####################################################
                #### COMPUTE PF VALUE, REWARD, AND STORE CURRENT ####
                #####################################################
            # Compute portfolio returns and value at period (t+1)
            ### Remark: Formally, r_(t+1) = r(s_t, a_t, s_(t+1))
            '''
            self.pf_rwds_dict = self.compute_pf_val_rets(weights)
            self.pf_value_cur = self.pf_rwds_dict["portfolio_value"]
            self.pf_return_cur = self.pf_rwds_dict["portfolio_return"]
            
            # Get reward
            self.reward = self.pf_rwds_dict[self.reward_type]            
            
            '''
                ##############################
                #### INCREMENT TIME INDEX ####
                ##############################
            '''
            self.t_idx += 1
            
            
            '''
                ##############################
                #### UPDATE DATASET ARRAY ####
                ##############################
            # Update actions and weights in self.X if required
            ### NOTE: When weights are included in states, the state at period (t+1)
            ###     is written: s_(t+1) = (x_(t+1), w_t, y_(t+1)), where w_t are the 
            ###     PREVIOUS weights, x_(t+1) is the close price at period (t+1),
            ###     and y_(t+1) are the other state features.
            ###     At this stage, t_idx still indexes the previous period.
            '''
            if self.state_incl_actions:
                self.X[self.t_idx, self.state_idx_dict["actions"], :] = actions
            if self.state_incl_weights:
                self.X[self.t_idx, self.state_idx_dict["weights"], :] = weights
            
            '''
                #############################################
                #### UPDATE PORTFOLIO HISTORY ATTRIBUTES ####
                #############################################
            # Update portfolio history attributes
            '''
            self.pf_value_hist[self.t_idx] = self.pf_value_cur # Shifted index, pf value at beginning of period
            self.pf_return_hist[self.t_idx] = self.pf_return_cur # Shifted index, pf return at beginning of period
            self.pf_weights_hist[self.t_idx,:] = weights # Previous period
            self.agt_action_hist[self.t_idx,:] = self.action_cur # Previous period

            
            '''
                ##################################
                #### UPDATE ENVIRONMENT STATE ####
                ##################################
            ### IMPORTANT: The state returned by step() is indexed by the incremented t_idx
            # Get state at time t_idx (i.e. at period (t+1))
            '''
            self.state = self.X[self.t_idx, :, :]
            
            return self.state, self.reward, self.at_terminal_prd, {}
    
    
    ################################
    ### Reset history attributes ###
    ################################
    # AJ Zerouali, 23/05/04
    def reset_hist_attr(self):
        '''
            Reset history attributes. 
            Added to step() to avoid the loss of the history attributes 
            when the stable_baselines3 wrapper resets.
            :return state: np.ndarray of current state in specified format.
        '''
        if self.t_idx == 0:
            ## Value history
            self.pf_value_hist = np.zeros(shape = (self.N_periods+1,), dtype = np.float64)
            self.pf_value_hist[self.t_idx] = self.pf_value_cur
            ## Return history
            self.pf_return_hist = np.zeros(shape = (self.N_periods+1,), dtype = np.float64)
            self.pf_return_hist[self.t_idx] = self.pf_return_cur
            ## Agent action(s) history
            self.agt_action_hist =  np.zeros(shape = (self.N_periods+1, self.n_assets), dtype = np.float64)
            self.agt_action_hist[self.t_idx,:] = self.action_cur
            ## Asset weights history. NOTE: Column order is that of tickers sorted alphabetically
            self.pf_weights_hist = np.zeros(shape = (self.N_periods+1, self.n_assets), dtype = np.float64)
            self.pf_weights_hist[self.t_idx,:] = self.pf_weights_cur
    
    #####################
    ### ReLU function ###
    #####################
    # 22/11/15, AJ Zerouali
    ## Quick computation of ReLU found on StackExchange.
    def ReLU(self, X:np.ndarray):
        return (X+np.abs(X))/2
    
    ###############################
    ### ReLU normalized weights ###
    ###############################
    # 23/05/18, AJ Zerouali
    def relu_normalization(self, actions):
        '''
            Function that computes portfolio weights from actions.
            Computes the ReLU output of actions and
            normalizes by the sum of its components.
            
            :param actions: np.ndarray, action vector obtained from agent.
            
            :return weights: np.ndarray, where weights = ReLU(actions)/sum(ReLU(actions))
        '''
        relu_actions = self.ReLU(actions)
        if relu_actions.sum() == 0.0:
            weights = np.ones(shape = (self.n_assets,), dtype = actions.dtype)/self.n_assets
        else:
            weights = relu_actions/relu_actions.sum()
        
        return weights
    
    ##################################
    ### Softmax normalized weights ###
    ##################################
    # 23/05/18, AJ Zerouali
    def softmax_normalization(self, actions):
        '''
            Function that computes portfolio weights from actions.
            Returns the softmax normalization of actions.
            
            :param actions: np.ndarray, action vector obtained from agent.
            
            :return weights: np.ndarray, where weights = SoftMax(actions) = exp(actions)/sum(exp(actions))
        '''
        exp_actions = np.exp(actions)
        weights = exp_actions/np.sum(exp_actions)
        return weights
    
    ##################################
    ### Identity normalized weights ###
    ##################################
    # 23/05/18, AJ Zerouali
    def identity_normalization(self, actions):
        '''
            Function that assigns the agent actions as portfolio weights.
            
            IMPORTANT NOTE: Ensure that the input of step() is indeed 
                            a vector whose components sum to 1 and 
                            are each in [0,1].
        '''
        return actions
    
    '''
    #######################
    ### Compute weights ###
    #######################
    # 22/11/15, AJ Zerouali
    # 23/05/18, DEPRECATED FUNCTION. See constructor for
    # assignment of compute_weights.
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
    '''
    
    #############################################
    ### Get portfolio value history dataframe ###
    #############################################
    def get_pf_value_hist(self):
        """
            Get portfolio value history.
            Format: - index: int from 0 to self.N_periods-1
                    - columns = ["date", "daily_value"]
            COMMENT: The list of dates here includes one timestamp more than 
                    self.date_hist, in order to account for the initial and
                    final portfolio values.
                    The entries self.date_hist are the effective portfolio rebalancing
                    timestamps, and this list does not include the initialization
                    timestamp of the portfolio.
                    
            :return df_pf_value_hist: pd.DataFrame of daily portfolio returns over dates in input.
        """ 
        # Add initialization timestamp to the date_hist
        date_list = [self.trading_data_schedule[self.date_hist[0]][-self.n_intratrading_timestamps]]\
                    + self.date_hist
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
            COMMENT: The list of dates here includes one timestamp more than 
                    self.date_hist, in order to account for the initial and
                    final portfolio values.
                    The entries self.date_hist are the effective portfolio rebalancing
                    timestamps, and this list does not include the initialization
                    timestamp of the portfolio.
            
            :return df_pf_return_hist: pd.DataFrame of daily portfolio returns over dates in input.
        """ 
        # Add initialization timestamp to the date_hist
        date_list = [self.trading_data_schedule[self.date_hist[0]][-self.n_intratrading_timestamps]]\
                    + self.date_hist
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
            COMMENT: The list of dates here includes one timestamp more than 
                    self.date_hist, in order to account for the initial and
                    final portfolio values.
                    The entries self.date_hist are the effective portfolio rebalancing
                    timestamps, and this list does not include the initialization
                    timestamp of the portfolio.
            
            :return df_pf_weights_hist: pd.DataFrame of daily weights returns over dates in input.
        """ 
        # Init.
        # Add initialization timestamp to the date_hist
        date_list = [self.trading_data_schedule[self.date_hist[0]][-self.n_intratrading_timestamps]]\
                    + self.date_hist
        pf_assets_list = self.pf_assets_list
        pf_weights_hist = self.pf_weights_hist.copy()
        
        # Make "date" column
        df_pf_weights_hist = pd.DataFrame({"date":date_list})
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
            COMMENT: The list of dates here includes one timestamp more than 
                    self.date_hist, in order to account for the initial and
                    final portfolio values.
                    The entries self.date_hist are the effective portfolio rebalancing
                    timestamps, and this list does not include the initialization
                    timestamp of the portfolio.
            
            :return df_agt_action_hist: pd.DataFrame of agent actions at each period in input dataframe.
        """ 
        # Add initialization timestamp to the date_hist
        date_list = [self.trading_data_schedule[self.date_hist[0]][-self.n_intratrading_timestamps]]\
                    + self.date_hist
        pf_assets_list = self.pf_assets_list
        agt_action_hist = self.agt_action_hist.copy()
        
        # Make "date" column
        df_agt_action_hist = pd.DataFrame({"date":date_list})
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