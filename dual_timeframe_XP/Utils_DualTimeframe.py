################################################
##### DRL PFOpt - Dual Timeframe Utilities #####
################################################
### AJ Zerouali, 23/05/03


from __future__ import print_function, division
from builtins import range

# NumPy, Pandas, matplotlib, PyTorch, Gym, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# pyfolio
from pyfolio import create_full_tear_sheet
from pyfolio.timeseries import perf_stats

# drl_pfopt
from drl_pfopt import PortfolioOptEnv
from drl_pfopt.common.data.data_utils import get_str_date_format
from drl_pfopt.common.data.data_utils import format_returns

'''
    ##########################################
    ##### DATA DICTIONARY SPLIT FUNCTION #####
    ##########################################
    AJ Zerouali, 23/05/03
    A version of the usual data_split() function
    tailored to the output of FeatureEngDualTF.
'''
def data_dict_split(data_dict: dict,
                    split_date: str,
                   ):
    '''
        Function to split data dictionary produced by
        FeatureEngDualTF.preprocess_data().
        Returns two dictionaries. First one is typically
        the training dataset and the second one is the
        testing/validation dataset.
        
        :param data_dict: dict of original data.
        :param split_date: str of end_date of (non-inclusive) first date of first dataset in %Y-%m-%d format
        
        :return data_dict_A, data_dict_B:
    '''
    # Check that split_date parameter occurs strictly after start_date and end_date of 
    # the trading days
    split_date_ = datetime.strptime(split_date,"%Y-%m-%d")
    if (split_date_<=data_dict["trade_timeframe_info"]["start_date"])\
        or (split_date_>=data_dict["trade_timeframe_info"]["end_date"]):
        trade_start_date = data_dict["trade_timeframe_info"]["start_date"].strftime("%Y-%m-%d")
        trade_end_date = data_dict["trade_timeframe_info"]["end_date"].strftime("%Y-%m-%d")
        raise ValueError(f"ERROR: split_date = {split_date} must be stricly after\n"\
                         f"start_date = {trade_start_date} and strictly before\n"\
                         f"end_date = {end_start_date}."
                        )
    
    # Init. output
    data_dict_A = {}
    data_dict_B = {}
    data_dict_A["feature_shape_dict"] = data_dict["feature_shape_dict"]
    data_dict_B["feature_shape_dict"] = data_dict["feature_shape_dict"]
    data_timeframe_info_A = {}
    data_timeframe_info_B = {}
    trade_timeframe_info_A = {}
    trade_timeframe_info_B = {}
    n_intratrading_timestamps = data_dict["trade_timeframe_info"]["n_intratrading_timestamps"]
    N_lookback_trade_prds = data_dict["trade_timeframe_info"]["N_lookback_trade_prds"]
    trade_timestamp_list = data_dict["trade_timeframe_info"]["trade_timestamp_list"]
    trading_data_schedule = data_dict["trade_timeframe_info"]["trading_data_schedule"]
    df = data_dict["df"]
        
    '''
        #############################
        ##### IMPORTANT INDICES #####
        #############################
    '''
    ## This index is used to make the day_list attributes 
    split_day_idx = data_dict["trade_timeframe_info"]["day_list"].index(split_date_)
    ## This next index is used to make the following attributes:
    ## trading_timestamp_list attributes, trading_data_schedule, np_close_returns, np_returns_covs.
    ## We recover it by searching for the first trading timestamp of the trading timeframe info dict.
    found_timestamp_idx = False
    i = 0
    while not found_timestamp_idx and i<len(trade_timestamp_list):
        if trade_timestamp_list[i][:10] == split_date:
            split_timestamp_idx = i
            found_timestamp_idx = True
        else:
            i+=1
    ### Check found_timestamp_idx that and that split_day_idx is larger than N_lookback_trade_prds
    if not found_timestamp_idx:
        raise ValueError("ERROR: Could not find the first timestamp idx corresponding\n"\
                         f"to split_date = {split_date}. Check the trading_timestamp_list."
                        )
    else:
        if split_timestamp_idx < N_lookback_trade_prds:
            raise ValueError(f"ERROR: The first timestamp index of {split_date}\n"\
                             f"must be larger or equal N_lookback_trade_prds={N_lookback_trade_prds}.\n"
                             f"Here: split_timestamp_idx = {split_timestamp_idx}."
                            )
    
    '''
        ###########################################################################
        ##### Invariant keys for data_timeframe_info and trade_timeframe_info #####
        ###########################################################################
    '''
    
    inv_keys = ["str_date_format", "timeframe", "extended_trading_hours", "n_intraday_timestamps"]
    #inv_keys_trade = ["n_intratrading_timestamps", "N_lookback_trade_prds"]
    ## Record invariant attributes for output data dictionaries
    for key in inv_keys:
        #data_dict_A[key] = data_dict[key]
        #data_dict_B[key] = data_dict[key]
        data_timeframe_info_A[key] = data_dict["data_timeframe_info"][key]
        trade_timeframe_info_A[key] = data_dict["trade_timeframe_info"][key]
        data_timeframe_info_B[key] = data_dict["data_timeframe_info"][key]
        trade_timeframe_info_B[key] = data_dict["trade_timeframe_info"][key]
    ## Trade timeframe intratrading periods and lookback periods
    trade_timeframe_info_A["n_intratrading_timestamps"] = n_intratrading_timestamps
    trade_timeframe_info_A["N_lookback_trade_prds"] = N_lookback_trade_prds
    trade_timeframe_info_B["n_intratrading_timestamps"] = n_intratrading_timestamps
    trade_timeframe_info_B["N_lookback_trade_prds"] = N_lookback_trade_prds
    ## Start dates for data_dict_A
    data_timeframe_info_A["start_date"] = data_dict["data_timeframe_info"]["start_date"]
    trade_timeframe_info_A["start_date"] = data_dict["trade_timeframe_info"]["start_date"]
    ## End dates for data_dict_B
    data_timeframe_info_B["end_date"] = data_dict["data_timeframe_info"]["end_date"]
    trade_timeframe_info_B["end_date"] = data_dict["trade_timeframe_info"]["end_date"]
        
    '''
        ##########################################
        ##### TRADING TIMEFRAME DICTIONARIES #####
        ##########################################
        Here we fill the day_list, trade_timestamp_list,
        start_date, end_date, trading_data_schedule
        entries of trade_timeframe_info dicts
        of the outputs.
    '''
    ## Trading timeframe info A
    ### Day list
    trade_timeframe_info_A["day_list"] = data_dict["trade_timeframe_info"]["day_list"][:split_day_idx+1]
    ### Trading timestamp list
    trade_timeframe_info_A["trade_timestamp_list"] = trade_timestamp_list[:split_timestamp_idx+1]
    ### End date
    trade_timeframe_info_A["end_date"] = split_date_
    ### Trading/data schedule
    trade_timeframe_info_A["trading_data_schedule"] = {}
    for trade_timestamp in trade_timeframe_info_A["trade_timestamp_list"]:
        trade_timeframe_info_A["trading_data_schedule"][trade_timestamp]\
            = trading_data_schedule[trade_timestamp]
    
    ## Trading timeframe info B
    ### Day list
    trade_timeframe_info_B["day_list"] = data_dict["trade_timeframe_info"]["day_list"][split_day_idx:]
    ### Trading timestamp list
    trade_timeframe_info_B["trade_timestamp_list"] = trade_timestamp_list[split_timestamp_idx+1:]
    ### End date
    trade_timeframe_info_B["start_date"] = split_date_
    ### Trading/data schedule
    trade_timeframe_info_B["trading_data_schedule"] = {}
    for trade_timestamp in trade_timeframe_info_B["trade_timestamp_list"]:
        trade_timeframe_info_B["trading_data_schedule"][trade_timestamp]\
            = trading_data_schedule[trade_timestamp]
    
    '''
        ################################################
        ##### CLOSE RETURNS AND COVARIANCES ARRAYS #####
        ################################################
        Use the split_timestamp_idx to split
        np_close_returns and np_returns_covs.
    '''
    # Close returns
    np_close_returns_A = data_dict["np_close_returns"][:split_timestamp_idx+1,:,:]
    np_close_returns_B = data_dict["np_close_returns"][split_timestamp_idx+1:,:,:]
    # Returns covariances
    np_returns_covs_A = data_dict["np_returns_covs"][:split_timestamp_idx+1,:,:]
    np_returns_covs_B = data_dict["np_returns_covs"][split_timestamp_idx+1:,:,:]
    
    '''
        ###########################################
        ##### PRICES AND TECH IND. DATAFRAMES #####
        ###########################################
        Use the split_timestamp_idx to split
        the df in data_dict. This is a little delicate
        because we still want to keep some lookback data
        for data_dict_B, meaning there could be overlap
        between the date columns of df_A and df_B.
    '''
    # Get end date for df_A and start date for df_B
    final_A_trade_timestamp = trade_timeframe_info_A["trade_timestamp_list"][-1]#trade_timestamp_list[split_timestamp_idx]
    dict_A_end_data_timestamp = trading_data_schedule[final_A_trade_timestamp][-1]
    first_B_trade_timestamp = trade_timeframe_info_B["trade_timestamp_list"][0] #trade_timestamp_list[split_timestamp_idx+1]
    dict_B_start_data_timestamp = trading_data_schedule[first_B_trade_timestamp][0]
    # Dataframe for data_dict_A
    df_A = df[df.date<=dict_A_end_data_timestamp]
    df_A.sort_values(["date", "tic"], ignore_index=True)
    # Dataframe for data_dict_B
    df_B = df[df.date>=dict_B_start_data_timestamp]
    df_B.sort_values(["date", "tic"], ignore_index=True)
    
    '''
        #######################################
        ##### DATA TIMEFRAME DICTIONARIES #####
        #######################################
        Here we fill the day_list, start_date, 
        and end_date of data_timeframe_info dicts
        of the outputs.
        This time, these attributes depend on the
        content of the trade_timeframe_info
        dictionaries.
    '''
    split_data_date = trade_timeframe_info_A["trade_timestamp_list"][-1][:10]
    split_data_date_ = datetime.strptime(split_data_date,"%Y-%m-%d")
    split_data_idx = data_dict["data_timeframe_info"]["day_list"]\
            .index(split_data_date_)
    # Data timeframe info A
    ## Day_list
    data_timeframe_info_A["day_list"] = data_dict["data_timeframe_info"]["day_list"][:split_data_idx]
    ## End date
    data_timeframe_info_A["end_date"] = split_data_date_
    # Data timeframe info B
    ## Day list
    data_timeframe_info_B["day_list"] = data_dict["data_timeframe_info"]["day_list"][split_data_idx:]
    ## Start date
    data_timeframe_info_B["start_date"] = split_data_date_
    
    '''
        ##########################
        ##### PREPARE OUTPUT #####
        ##########################
    '''    
    # Add timeframe info dictionaries to output
    data_dict_A["data_timeframe_info"] = data_timeframe_info_A
    data_dict_A["trade_timeframe_info"] = trade_timeframe_info_A
    data_dict_B["data_timeframe_info"] = data_timeframe_info_B
    data_dict_B["trade_timeframe_info"] = trade_timeframe_info_B
    
    # Add pd.DataFrame and np.array objs to output
    data_dict_A["df"] = df_A
    data_dict_A["np_close_returns"] = np_close_returns_A
    data_dict_A["np_returns_covs"] = np_returns_covs_A
    data_dict_B["df"] = df_B
    data_dict_B["np_close_returns"] = np_close_returns_B
    data_dict_B["np_returns_covs"] = np_returns_covs_B
    
    return data_dict_A, data_dict_B

'''
    ###################################
    ##### RANDOM WEIGHTS FUNCTION #####
    ###################################
    AJ Zerouali, 23/05/03
    A version of the usual data_split() function
    tailored to the output of FeatureEngDualTF.
'''

def exec_random_weights(env: PortfolioOptEnv):
    '''
        Random weights policy for a PortfolioOptEnv
        
        :param env: DRL_PFOpt_gymEnv.PortfolioOptEnv object. 
        
        :return rw_benchmark_dict: dict of random weights strategy results
                    over data in env. Keys are as follows:
                    * "value_hist": History of portfolio values.
                    * "return_hist": History of portfolio returns.
                    * "Sharpe_ratio": Sharpe ratio of the strategy over the period.
                    * "performance_stats": Performance stats computed by
                                        pyfolio.timeseries.perf_stats
    '''
    # Initializations
    np.random.seed(0)
    env.reset()
    actions_mu = np.ones(shape = (env.n_assets,))/env.n_assets
    actions_sigma = 0.05
    str_date_format = get_str_date_format(env.date_hist)
    
    # Main loop
    for i in range(env.N_periods):
        # Notice that the initial weights have already been assigned
        actions_rand_wts = np.random.normal(loc = actions_mu, scale=actions_sigma, size = (env.n_assets,))
        env.step(actions_rand_wts)
        '''
            23/04/28
            COMMENT: See lines 360-361 of base_agent.py for an important comment
                    on the (N_periods-2) condition.
                    In the new implementation, this number of periods is of crucial
                    importance.
        '''
        if i == env.N_periods - 1:#2: # Why is the condition (N_periods-2)
            df_rw_value_hist = env.get_pf_value_hist()
            df_rw_return_hist = env.get_pf_return_hist()
            df_rw_weights_hist = env.get_pf_weights_hist()
            rw_Sharpe_ratio = (np.sqrt(252)*df_rw_return_hist["daily_return"].to_numpy().mean())\
                            /(df_rw_return_hist["daily_return"].to_numpy().std())
    
    env.reset()
    
    # Make performance stats dataframe from pyfolio's perf_stats
    df_returns_ = df_rw_return_hist.copy()
    df_returns_["date"] = pd.to_datetime(df_returns_["date"], 
                                         format=str_date_format)
    df_returns_ = df_returns_.set_index('date')
    series_performance_stats = perf_stats(returns=df_returns_["daily_return"])
    del df_returns_
    df_rw_perf_stats = pd.DataFrame(series_performance_stats)
    df_rw_perf_stats = df_rw_perf_stats.rename(columns = {0:"Value"})
    
    # Prepare output - "rw" for "random weights"
    rw_results_dict = {}
    rw_results_dict["value_hist"] = df_rw_value_hist
    rw_results_dict["return_hist"] = df_rw_return_hist
    rw_results_dict["Sharpe_ratio"] = rw_Sharpe_ratio
    rw_results_dict["performance_stats"] = df_rw_perf_stats
    rw_results_dict["weight_hist"] = df_rw_weights_hist
    
    return rw_results_dict