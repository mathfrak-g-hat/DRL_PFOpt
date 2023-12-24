'''
    IMPORTS
'''
# Kill warnings
from warnings import filterwarnings
filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

from drl_pfopt import (PortfolioOptEnv, 
                       FeatureEngineer,
                       data_split,
                      )
# Data utils imports
from drl_pfopt.common.data.data_utils import (get_market_calendar_day_list,
                                              get_timeframe_info,
                                              get_str_date_format,
                                              make_intraday_timestamp_list,
                                              make_timestamp_list,
                                             )


'''
    LOAD ALPACA DATA
'''
# Ticker list for this section
TICKER_LIST = ['AXP', 'BA', 'CVX', 'JNJ', 'KO']

# Load DJIA file 
df_data = pd.read_csv(filepath_or_buffer="datasets/5day_min_data_ex.csv",
                      usecols = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic'])

data_timestamp_list_0 = list(df_data.date.unique())
data_timeframe_info_dict_0 = get_timeframe_info(data_timestamp_list_0)



str_date_format = data_timeframe_info_dict_0["str_date_format"]
# All remaining data cases use this day list list
data_day_list = [x.strftime("%Y-%m-%d") for x in data_timeframe_info_dict_0["day_list"]]

example_dict = {}

'''
    ---------------
    ### HOURLY DATA 
    ---------------
    A - EXTENDED HOURS
'''
hour_ext_data_dict = {}
data_timestamp_list = make_timestamp_list(day_list = data_timeframe_info_dict_0["day_list"],
                                          timeframe = "Hour",
                                          extended_trading_hours = True,
                                          feature_eng_list = False,
                                         )
df = df_data[df_data.date.isin(data_timestamp_list)]

# Fill the dictionary for "Hour_ext_data"
hour_ext_data_dict["str_date_format"] = str_date_format
hour_ext_data_dict["extended_trading_hours"] = True
hour_ext_data_dict["data_timestamp_list"] = data_timestamp_list
hour_ext_data_dict["df"] = df

# Add to daily trading dictionary
example_dict["Hour_ext_data"] = hour_ext_data_dict
# Delete temps
del df, data_timestamp_list


'''
    ---------------
    ### HOURLY DATA 
    ---------------
    B - STANDARD HOURS
'''
hour_std_data_dict = {}
data_timestamp_list = make_timestamp_list(day_list = data_timeframe_info_dict_0["day_list"],
                                          timeframe = "Hour",
                                          extended_trading_hours = False,
                                          feature_eng_list = False,)
df = df_data[df_data.date.isin(data_timestamp_list)]

# Fill the dictionary for "Hour_ext_data"
hour_std_data_dict["str_date_format"] = str_date_format
hour_std_data_dict["extended_trading_hours"] = False
hour_std_data_dict["data_timestamp_list"] = data_timestamp_list
hour_std_data_dict["df"] = df

# Add to daily trading dictionary
example_dict["Hour_std_data"] = hour_std_data_dict
# Delete temps
del df, data_timestamp_list

MIN_TIMEFRAME_LIST = ["1min", "2min", "3min", "4min", "5min", "6min",
                             "10min", "12min", "15min", "20min", "30min"]

for tf in MIN_TIMEFRAME_LIST:
    '''
        ---------------
        ### (int)min DATA 
        ---------------
        A - EXTENDED HOURS
    '''
    min_ext_data_dict = {}
    data_timestamp_list = make_timestamp_list(day_list = data_timeframe_info_dict_0["day_list"],
                                              timeframe = tf,
                                              extended_trading_hours = True,
                                              feature_eng_list = False,
                                             )
    df = df_data[df_data.date.isin(data_timestamp_list)]

    # Fill the dictionary for "Hour_ext_data"
    min_ext_data_dict["str_date_format"] = str_date_format
    min_ext_data_dict["extended_trading_hours"] = True
    min_ext_data_dict["data_timestamp_list"] = data_timestamp_list
    min_ext_data_dict["df"] = df

    # Add to daily trading dictionary
    example_dict[tf+"_ext_data"] = min_ext_data_dict
    # Delete temps
    del df, data_timestamp_list


    '''
        ---------------
        ### (int)min DATA
        ---------------
        B - STANDARD HOURS
    '''
    min_std_data_dict = {}
    data_timestamp_list = make_timestamp_list(day_list = data_timeframe_info_dict_0["day_list"],
                                              timeframe = tf,
                                              extended_trading_hours = False,
                                              feature_eng_list = False,)
    df = df_data[df_data.date.isin(data_timestamp_list)]

    # Fill the dictionary for "Hour_ext_data"
    min_std_data_dict["str_date_format"] = str_date_format
    min_std_data_dict["extended_trading_hours"] = False
    min_std_data_dict["data_timestamp_list"] = data_timestamp_list
    min_std_data_dict["df"] = df

    # Add to daily trading dictionary
    example_dict[tf+"_std_data"] = min_std_data_dict
    # Delete temps
    del df, data_timestamp_list
    
    del min_ext_data_dict, min_std_data_dict






