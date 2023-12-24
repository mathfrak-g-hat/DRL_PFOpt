##################################################################
### Deep RL portfolio optimization - Data processing utilities ###
##################################################################
### 2023/05/02, A.J. Zerouali
from __future__ import annotations, print_function, division

import time
from datetime import datetime, timedelta
from builtins import range

# np, pd plt
import numpy as np
import pandas as pd
import pandas_market_calendars as pd_mkt_cals
import matplotlib.pyplot as plt


'''
    ###############################
    ###  FinRL Helper Functions ###
    ###############################
'''
def get_daily_return(df, value_col_name="account_value"):
    df = df.copy()
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)

###############################
##### Data split function #####
###############################
## 22/10/28 Cheng C. Updated helper function to split dataframe and return/covariance arrays according to specified dates.
def data_split(df_X: pd.DataFrame, 
               start: str, 
               end:str, 
               target_date_col: str ="date",
               use_returns_cov: bool = False, 
               ret_X: np.ndarray = np.empty(shape=0),
               cov_X: np.ndarray = np.empty(shape=0)):
    '''
    Function to impute a dataset given a start and an end date.
    Returns only a pd.DataFrame if use_returns_cov is False,
    and a pd.DataFrame along with two np.ndarray's if True.
    
    ### Note (22/10/27): The inputs of this function are
    ###     intended to be outputs of the FeatureEngineer.preprocess_data()
    ###     function of the present submodule.
    
    :param df_X, start, end: pd.DataFrame of financial data
    :param use_returns_cov: bool to split close returns and covariances arrays 
                            ## this bool should be enough for both return and cov (i remove use_cl_ret) - Cheng C. 22/10/28 
    
    :param ret_X: np.ndarray of close returns over dates in df_X
    :param cov_X: np.ndarray of return covariances over dates in df_X
    
    :return df_Y: pd.DataFrame of imputed dataset.
    :return ret_Y: np.ndarray of close returns over dates in df_Y
    :return cov_Y: np.ndarray of return covariances over dates in df_Y
    
    '''
    
    
    df_Y = df_X[(df_X[target_date_col] >= start) & (df_X[target_date_col] < end)]
    df_Y = df_Y.sort_values([target_date_col, "tic"], ignore_index=True)
    df_Y.index = df_Y[target_date_col].factorize()[0]
    
    
    if use_returns_cov:
        ### check if df_X has same number of dates as return and cov 
        if len(df_X.date.unique()) == ret_X.shape[0] and len(df_X.date.unique()) == cov_X.shape[0]:
            print("Dataframe, returns and cov have same number of dates")
        else:
            raise Exception("Dataframe, returns and cov DO NOT have same number of dates")
        
        
        Y_index = list(df_Y.date.unique())
        X_index = list(df_X.date.unique())

        start_index = X_index.index(Y_index[0])
        end_index = X_index.index(Y_index[-1])
        
        ret_Y = ret_X[start_index:end_index+1,:,:] #### note: end_index+1 because we need to include the last element 
        cov_Y = cov_X[start_index:end_index+1,:,:]
        
        return df_Y, ret_Y, cov_Y
    
    else:
        return df_Y
    
    
'''
    ################################
    ##### FORMATTING FUNCTIONS #####
    ################################
    * get_str_date_format()
    * format_returns()
'''

#####################################################
#### Get date format string from a list of dates ####
#####################################################

# Updated 23/03/26
## Supports more intra-hour timestamps
def get_str_date_format(date_list):
    '''
        Helper function to extract string date format used in a list of dates.
        Checks if the timeframes used in said list are coherent with the ones
        used in this package (i.e. "Day", "Hour", "1min", "2min", "3min", "4min",
        "5min", "6min","10min", "12min", "15min", "20min", "30min")
    '''
    all_list_elts_str = all(isinstance(x, str) for x in date_list)
    all_list_elts_datetime = all(isinstance(x, datetime) for x in date_list)
    
    # Case where all timestamps in the list are strings
    if all_list_elts_str:
        
        # Get date format
        ## Case of "%Y-%m-%d"
        if all(len(x)==10 for x in date_list):
            str_date_format = "%Y-%m-%d"
        ## Case of "%Y-%m-%d %H:%M"
        elif all(len(x)==16 for x in date_list):
            # Old version, 22/11/24
            #if all((int(x[14:16]) in [0,15,30,45]) and (int(x[11:13]) in range(8,24)) for x in date_list):
            # 23/03/24 update
            if all((int(x[14:16]) in range(60)) and (int(x[11:13]) in range(8,24)) for x in date_list):
                str_date_format = "%Y-%m-%d %H:%M"
        else:
            raise ValueError("Unrecognized date format or timeframe in date_list.\n"\
                             "Accepted date formats: \"%Y-%m-%d\" or \"%Y-%m-%d %H:%M\"\n."\
                             "Accepted timeframes: \"Day\", \"Hour\", \n"\
                             "                    \"1min\", \"2min\", \"3min\", \"4min\", \"5min\", \"6min\",\n"\
                             "                    \"10min\", \"12min\", \"15min\", \"20min\", \"30min\"")
            
    elif all_list_elts_datetime:
        
        # Get date format    
        if all( (x.hour, x.minute, x.second) == (0,0,0) for x in date_list):
            str_date_format = "%Y-%m-%d"
        # Old version, 22/11/24
        #elif all( (x.minute in [0,15,30,45]) and (x.hour in range(8,24)) for x in date_list):
        # 23/03/26 update
        elif all((x.hour in range(8,24)) for x in date_list):
            str_date_format = "%Y-%m-%d %H:%M"
        else:
            raise ValueError("Unrecognized date format or timeframe in date_list.\n"\
                             "Accepted date formats: \"%Y-%m-%d\" or \"%Y-%m-%d %H:%M\"\n."\
                             "Accepted timeframes: \"Day\", \"Hour\", \n"\
                             "                    \"1min\", \"2min\", \"3min\", \"4min\", \"5min\", \"6min\",\n"\
                             "                    \"10min\", \"12min\", \"15min\", \"20min\", \"30min\"")
            
    elif (not all_list_elts_str) and (not all_list_elts_datetime):
        raise ValueError("All entries of the date_list parameter must be either str or datetime instances.")
    
    return str_date_format

###########################################################
#### Format backtest and benchmark returns for pyfolio ####
###########################################################
### 22/11/15, AJ Zerouali
def format_returns(df_backtest_returns, df_benchmark_returns, benchmark_name: str = "Benchmark_returns"):
    '''
        Function to unify format of backtest and benchmark returns.
        The outputs of this function are compatible with PyFolio's
        create_full_tear_sheet() function.
        ### NOTE (22/11/15): Temporary implementation. Used in
        ###    PFOpt_DRAL_Agent.plot_backtest_results().
        ###    See that method for expected format.
        
        :param df_backtest_returns: Backtest returns history dataframe
                        obtained from PortfolioOptEnv and PFOpt_DRL_Agent.
        :param df_benchmark_returns: Benchmark returns history dataframe
                        obtained from get get_benchmark_prices_and_returns().
                        
        :return df_backtest_returns_: Formatted pd.DataFrame of backtest returns
                                        index = 'date'
                                        columns = ['daily_return']
        :return df_benchmark_returns_: Formatted pd.DataFrame of benchmark returns
                                        index = 'date'
                                        columns = ['daily_return']
    '''
    # Copy input
    df_backtest_returns_ = df_backtest_returns.copy()
    df_benchmark_returns_ = df_benchmark_returns.copy()
    
    # Get string date format
    str_date_format = get_str_date_format(list(df_backtest_returns_.date.unique()))
    
    # Convert return dates to datetime objects
    df_backtest_returns_["date"] = pd.to_datetime(df_backtest_returns_["date"], 
                                                  format=str_date_format)  
    df_benchmark_returns_["date"] = pd.to_datetime(df_benchmark_returns_["date"], 
                                                   format=str_date_format)
    
    # Set 'date' column as index
    df_backtest_returns_ = df_backtest_returns_.set_index('date')
    df_benchmark_returns_ = df_benchmark_returns_.set_index('date')
    
    # Rename benchmark returns column
    df_benchmark_returns_ = df_benchmark_returns_.rename(columns={list(df_benchmark_returns_.columns)[0] : benchmark_name})
    
    return df_backtest_returns_, df_benchmark_returns_

'''
    ##################################################
    ##### DATES AND TIMEFRAMES RELATED FUNCTIONS #####
    ##################################################
    * get_market_calendar_day_list()
    * get_timeframe_info()
    * make_intraday_timsetamp_list()
    * make_timestamp_list()
'''
####################################
##### Get Market Calendar Days #####
####################################
## AJ Zerouali, 22/11/21
def get_market_calendar_day_list(start_date: datetime,
                                 end_date: datetime,
                                 calendar_name: str = "NASDAQ"):
    '''
        Function to get the list of days from a calendar.
        
        :param start_date: datetime object.
        :param end_date: datetime object.
        
        :return day_list: list of (datetime) calendar days.
    '''
    calendar = pd_mkt_cals.get_calendar(name = calendar_name)
    # End date is inclusive in pandas-market-calendars
    schedule = calendar.schedule(start_date = start_date, 
                                 end_date = end_date-timedelta(days=1))
    day_list = [x.to_pydatetime() for x in list(schedule.index)]
    del calendar, schedule
    
    return day_list

# Updated 23/05/02
## Supports more intra-hour timestamps. Fixed an issue with end_date using
## get_market_calendar_day_list().
def get_timeframe_info(date_list: list):
    '''
        Function to compute several timeframe variables associated to a 
        sorted list of dates in string format.
        This function is taylored to recover the parameters of the
        AlpacaDownloader.make_timestamp_list() method.        
        
        :param date_list: List of dates in str format.
        
        :return timeframe_info_dict: dict of timeframe info.
                timeframe_info_dict.keys() = ["str_date_format", "start_date", "end_date", 
                                            "timeframe", "extended_trading_hours", 
                                            "day_list", "n_intraday_timestamps"]
   
    '''
    # Get string date format
    str_date_format = get_str_date_format(date_list)
    
    all_list_elts_str = all(isinstance(x, str) for x in date_list)
    all_list_elts_datetime = all(isinstance(x, datetime) for x in date_list)
    
    # Case where all timestamps in the list are strings
    if all_list_elts_str:
        # Get start_date and end_date
        ## Compute the start date of the list
        start_date = datetime.strptime(date_list[0], str_date_format)
        start_date = start_date - timedelta(hours= start_date.hour,
                                            minutes = start_date.minute,)
        ## Compute the end date of the date list (the day AFTER the last calendar date in the list)
        #end_date = datetime.strptime(date_list[-1], str_date_format)
        #end_date = end_date + timedelta(days = 1) - timedelta(hours= end_date.hour,
        #                                                      minutes = end_date.minute,))
        ## Temporary end_date. The above may not be a business day.
        last_day_list_ = datetime.strptime(date_list[-1], str_date_format)
        last_day_list_ = last_day_list_ - timedelta(hours= last_day_list_.hour,
                                                    minutes = last_day_list_.minute,)
        end_date_ = last_day_list_ + timedelta(days = 10)
        
    # Case where all timestamps in the list are datetime objects
    elif all_list_elts_datetime:
        # Get start_date and end_date
        start_date = datetime(date_list[0].year, date_list[0].month, date_list[0].day)
        #end_date = datetime(date_list[-1].year, date_list[-1].month, date_list[-1].day) + timedelta(days = 1)
        ## Temporary end_date. The above may not be a business day.
        last_day_list_ = datetime(date_list[-1].year, date_list[-1].month, date_list[-1].day)
        end_date_ = last_day_list_ + timedelta(days = 10)
    
    # Inits
    len_list = len(date_list)
    ### Dictionary of timeframes by no. of intraday timestamps
    timeframe_intraday_dict = {1:("Day", False), 16:("Hour", True), 8:("Hour", False),}
    '''
    # Old version 22/11/24
    timeframe_intraday_dict = {61:("15min", True), 29:("15min", False),
                               31:("30min", True), 15:("30min", False),
                               16:("Hour", True), 8:("Hour", False),
                               1:("Day", False)}
    '''
    ADMISSIBLE_MIN_TIMEFRAMES = ["1min", "2min", "3min", "4min", "5min", "6min",
                                 "10min", "12min", "15min", "20min", "30min"]
    for tf in ADMISSIBLE_MIN_TIMEFRAMES:
        tf_int = int(tf[:-3])
        ## No. of intra-hour timestamps
        n_intrahour_timestamps = int(60/tf_int)
        # No. of intraday timestamps for non-extended/standard trading hours
        n_intraday_timestamps_standard = 7*n_intrahour_timestamps + 1
        timeframe_intraday_dict[n_intraday_timestamps_standard] = (tf, False)
        # No. of intraday timestamps for extended trading hours
        n_intraday_timestamps_extended = 15*n_intrahour_timestamps + 1
        timeframe_intraday_dict[n_intraday_timestamps_extended] = (tf, True)
    
    # Get day list and no. of el'ts therein
    ## Start with day_list_ temporary
    day_list_ = get_market_calendar_day_list(start_date = start_date,
                                            end_date = end_date_,
                                            calendar_name = 'NASDAQ')
    ## Use index of last day in temp list to get correct day_list and end_date
    last_day_list_idx = day_list_.index(last_day_list_)
    end_date = day_list_[last_day_list_idx+1]
    day_list = day_list_[:last_day_list_idx+1]
    n_days = len(day_list)
        
    
    # Get the number of intraday timestamps
    n_intraday_timestamps = int(len_list/n_days)
    
    # Get the timeframe keyword and extended_trading_hours boolean:
    if n_intraday_timestamps not in timeframe_intraday_dict.keys():
        raise ValueError("Unrecognized timeframe for date_list.")
    else:
        timeframe, extended_trading_hours = timeframe_intraday_dict[n_intraday_timestamps]
        
    # Prepare output
    timeframe_info_dict = {}
    timeframe_info_dict["str_date_format"] = str_date_format
    timeframe_info_dict["start_date"] = start_date
    timeframe_info_dict["end_date"] = end_date
    timeframe_info_dict["timeframe"] = timeframe
    timeframe_info_dict["extended_trading_hours"] = extended_trading_hours
    timeframe_info_dict["day_list"] = day_list
    timeframe_info_dict["n_intraday_timestamps"] = n_intraday_timestamps
    
    return timeframe_info_dict


########################################
##### Make intraday timestamp list #####
########################################
## AJ Zerouali, 23/04/20
def make_intraday_timestamp_list(timeframe,
                                 extended_trading_hours,
                                 str_format: bool = True,
                                 feature_eng_list: bool = True,
                                ):
    '''
        Helper function to build list of intra-day timestamps according to the 
        timeframe and whether or not we are considering extended trading hours.
        Note that the resulting list does not contain timestamps at "16:00" and "23:00".
        
        :param timeframe: str, intraday timeframe keyword. Must be an element in:
                    [ "Hour", "1min", "2min", "3min", "4min", "5min",
                      "6min", "10min", "12min", "15min", "20min", "30min"]
        :param extended_trading_hours: bool.        
        :param str_format: bool, True by default. If False, then the entries of the output list are datetime.timedelta
                        objects. If True, the timestamps are converted to str.
        :param feature_eng_list: bool, True by default. If False, the timestamps "23:00" or "16:00" are added.
        
        :return intraday_timestamp_list:
    '''
    ADMISSIBLE_TIMEFRAME_LIST = ["Hour",
                             "1min", "2min", "3min", "4min", "5min", "6min",
                             "10min", "12min", "15min", "20min", "30min"] 
    if timeframe not in ADMISSIBLE_TIMEFRAME_LIST:
        raise ValueError(f"The timeframe parameter must be an element of {ADMISSIBLE_TIMEFRAME_LIST}")
        
    '''
    # Hour integer list 
    if not extended_trading_hours:
        hour_list_int = [x for x in range(9,16)]
    else:
        hour_list_int = [x for x in range(8,23)]
    '''
    # Hour integer list 
    if not extended_trading_hours:
        if feature_eng_list:
            hour_list_int = [x for x in range(9,16)]
        else:
            hour_list_int = [x for x in range(9,17)]
    else:
        if feature_eng_list:
            hour_list_int = [x for x in range(8,23)]
        else:
            hour_list_int = [x for x in range(8,24)]
    
    # Minute integer list 
    if timeframe == "Hour":
        timeframe_int = 60
        min_list_int = [0]
    else:
        timeframe_int = int(timeframe[:-3])
        min_list_int = [x for x in range(0,60,timeframe_int)]
    
    # timestamp list
    intraday_timestamp_list = []
    if feature_eng_list:
        if not str_format:
            for h in hour_list_int:
                for m in min_list_int:
                    intraday_timestamp_list.append(timedelta(hours = h, minutes = m))
        else:
            hour_list_str = [str(x) if x>=10 else "0"+str(x) for x in hour_list_int]
            min_list_str = [str(x) if x>=10 else "0"+str(x) for x in min_list_int]
            for h in hour_list_str:
                for m in min_list_str:
                    intraday_timestamp_list.append(h+":"+m)
    else:
        if not str_format:
            for h in hour_list_int[:-1]:
                for m in min_list_int:
                    intraday_timestamp_list.append(timedelta(hours = h, minutes = m))
            intraday_timestamp_list.append(timedelta(hours = hour_list_int[-1]))
        else:
            hour_list_str = [str(x) if x>=10 else "0"+str(x) for x in hour_list_int]
            min_list_str = [str(x) if x>=10 else "0"+str(x) for x in min_list_int]
            for h in hour_list_str[:-1]:
                for m in min_list_str:
                    intraday_timestamp_list.append(h+":"+m)
            intraday_timestamp_list.append(hour_list_str[-1]+":00")

    return intraday_timestamp_list

###############################
##### Make timestamp list #####
###############################
## AJ Zerouali, 23/04/20
def make_timestamp_list(day_list,
                        timeframe,
                        extended_trading_hours,
                        str_format: bool = True,
                        feature_eng_list: bool = True,
                       ):
    '''
        Function that builds a list of timestamps starting from a list of days.
        Calls the helper function make_intraday_timestamp_list.
        
        
        :param day_list: list of days in datetime.datetime format.
        :param timeframe: str, intraday timeframe keyword. Must be an element in: 
                    [ "Hour", "1min", "2min", "3min", "4min", "5min",
                      "6min", "10min", "12min", "15min", "20min", "30min"]
        :param extended_trading_hours: bool. Whether or not the output timestamps
                        should be in [9:00, 16:00[ (False) or [8:00, 23:00[ (True).
        :param str_format: bool, True by default. If False, then the entries of the output list are datetime.datetime
                        objects. If True, the timestamps are converted to str.
        :param feature_eng_list: bool, True by default. If False, the timestamps "23:00" or "16:00" are added.
                        False corresponds to the lists of the AlpacaDownloader, True corresponds to
                        the lists of the dual timeframe feature engineer object.
        
        :return intraday_timestamp_list: list of timestamps in str or datetime.datetime format.
        
    '''
    
    # Process day list
    all_list_elts_str = all(isinstance(x, str) for x in day_list)
    all_list_elts_datetime = all(isinstance(x, datetime) for x in day_list)
    
    if (not all_list_elts_str) and (not all_list_elts_datetime):
        raise ValueError("All entries of the day_list parameter must str's or datetime objects")
    elif all_list_elts_str:
        # Convert to datetime objects if necessary
        if not str_format:
            day_list_ = [datetime.strptime(x,"%Y-%m-%d") for x in day_list]
        else:
            day_list_ = day_list.copy()
    elif all_list_elts_datetime:
        # Convert to str objects if necessary
        if str_format:
            day_list_ = [x.strftime("%Y-%m-%d") for x in day_list]#[datetime.strptime(x,"%Y-%m-%d") for x in day_list]
        else:
            day_list_ = day_list.copy()
    
    # Get intraday timestamp list
    intraday_timestamp_list = make_intraday_timestamp_list(timeframe = timeframe,
                                                           extended_trading_hours=extended_trading_hours,
                                                           str_format = str_format,
                                                           feature_eng_list = feature_eng_list,
                                                          )
    
    # Make timestamp_list
    timestamp_list = []
    if str_format:
        for d in day_list_:
            for ts in intraday_timestamp_list:
                timestamp_list.append(d+" "+ts)
    else:
        for d in day_list_:
            for ts in intraday_timestamp_list:
                timestamp_list.append(d+ts)
        
    
    return timestamp_list

