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
                                             )
from trading_timeframe_utils import (get_n_intratrading_timestamps,
                                     get_intraday_data_timestamps,
                                    )

'''
    LOAD ALPACA DATA
'''
# Ticker list for this section
TICKER_LIST = ['AXP', 'BA', 'CVX', 'JNJ', 'KO']

# Load DJIA file 
df_Alpaca_15min = pd.read_csv(filepath_or_buffer="datasets/Alpaca_DJIA_15min_2208-2211.csv",
                           usecols = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic'])

# Reduce to TICKER_LIST data
df_data = df_Alpaca_15min[df_Alpaca_15min['tic'].isin(TICKER_LIST)]
data_timestamp_list_0 = list(df_data.date.unique())
data_timeframe_info_dict_0 = get_timeframe_info(data_timestamp_list_0)

# Delete loaded file
del df_Alpaca_15min

# Dictionary
example_dict = {}


'''
    #####################
    ### DAILY TRADING ###
    #####################
'''
daily_trading_example_dict = {}
'''
    --------------
    ### DAILY DATA 
    --------------
'''
daily_data_dict = {}
str_date_format = "%Y-%m-%d"
data_timestamp_list = [x.strftime(str_date_format) for x in data_timeframe_info_dict_0["day_list"]]
# Take the last data point of the day
data_timestamp_list_ = [x+" 23:00" for x in data_timestamp_list]
# Keep only values for the timestamps "%Y-%m-%d 23:00"
df = df_data[df_data.date.isin(data_timestamp_list_)]
# Format date column
df['date'] = df.date.apply(lambda x: x[:10])

# Fill the dictionary for "Day_data"
daily_data_dict["str_date_format"] = str_date_format
daily_data_dict["data_timestamp_list"] = data_timestamp_list
daily_data_dict["df"] = df

# Add to daily trading dictionary
daily_trading_example_dict["Day_data"] = daily_data_dict

# Delete temps
del df, data_timestamp_list, data_timestamp_list_

########################################################################################################

# All remaining data cases use this string date format
str_date_format = data_timeframe_info_dict_0["str_date_format"]
# All remaining data cases use this day list list
data_day_list = [x.strftime("%Y-%m-%d") for x in data_timeframe_info_dict_0["day_list"]]
# All remaining data cases use the following hour timestamps
hour_ext_list = [str(x) for x in range(8,24)]#[str(x)+":00" for x in range(8,24)]
hour_ext_list[0] = "08"
hour_ext_list[1] = "09"
hour_std_list = [str(x) for x in range(9,17)]#[str(x)+":00" for x in range(9,17)]
hour_std_list[0] = "09"
# All remaining data cases will use the following minute timestamps
minutes_15_list = ["00", "15", "30", "45"]
minutes_30_list = ["00", "30"]

'''
    ---------------
    ### HOURLY DATA 
    ---------------
    A - EXTENDED HOURS
'''
hour_ext_data_dict = {}
data_timestamp_list = []
for d in data_day_list:
    for h in hour_ext_list:
        data_timestamp_list.append(d+" "+h+":00")
df = df_data[df_data.date.isin(data_timestamp_list)]

# Fill the dictionary for "Hour_ext_data"
hour_ext_data_dict["str_date_format"] = str_date_format
hour_ext_data_dict["extended_trading_hours"] = True
hour_ext_data_dict["data_timestamp_list"] = data_timestamp_list
hour_ext_data_dict["df"] = df

# Add to daily trading dictionary
daily_trading_example_dict["Hour_ext_data"] = hour_ext_data_dict
# Delete temps
del df, data_timestamp_list

'''
    ---------------
    ### HOURLY DATA 
    ---------------
    B - STANDARD HOURS
'''
hour_std_data_dict = {}
data_timestamp_list = []
for d in data_day_list:
    for h in hour_std_list:
        data_timestamp_list.append(d+" "+h+":00")
df = df_data[df_data.date.isin(data_timestamp_list)]

# Fill the dictionary for "Hour_ext_data"
hour_std_data_dict["str_date_format"] = str_date_format
hour_std_data_dict["extended_trading_hours"] = False
hour_std_data_dict["data_timestamp_list"] = data_timestamp_list
hour_std_data_dict["df"] = df

# Add to daily trading dictionary
daily_trading_example_dict["Hour_std_data"] = hour_std_data_dict
# Delete temps
del df, data_timestamp_list

'''
    ---------------
    ### 30MIN DATA 
    ---------------
    A - EXTENDED HOURS
'''
min30_ext_data_dict = {}
data_timestamp_list = []
for d in data_day_list:
    for h in hour_ext_list:
        if h != "23":
            for m in minutes_30_list:
                data_timestamp_list.append(d+" "+h+":"+m)
        else:
            data_timestamp_list.append(d+" "+h+":00")
df = df_data[df_data.date.isin(data_timestamp_list)]

# Fill the dictionary for "Hour_ext_data"
min30_ext_data_dict["str_date_format"] = str_date_format
min30_ext_data_dict["extended_trading_hours"] = True
min30_ext_data_dict["data_timestamp_list"] = data_timestamp_list
min30_ext_data_dict["df"] = df

# Add to daily trading dictionary
daily_trading_example_dict["30min_ext_data"] = min30_ext_data_dict
# Delete temps
del df, data_timestamp_list

'''
    ---------------
    ### 30MIN DATA 
    ---------------
    B - STANDARD HOURS
'''
min30_std_data_dict = {}
data_timestamp_list = []
for d in data_day_list:
    for h in hour_std_list:
        if h != "16":
            for m in minutes_30_list:
                data_timestamp_list.append(d+" "+h+":"+m)
        else:
            data_timestamp_list.append(d+" "+h+":00")
df = df_data[df_data.date.isin(data_timestamp_list)]

# Fill the dictionary for "Hour_ext_data"
min30_std_data_dict["str_date_format"] = str_date_format
min30_std_data_dict["extended_trading_hours"] = False
min30_std_data_dict["data_timestamp_list"] = data_timestamp_list
min30_std_data_dict["df"] = df

# Add to daily trading dictionary
daily_trading_example_dict["30min_std_data"] = min30_std_data_dict
# Delete temps
del df, data_timestamp_list

'''
    ---------------
    ### 15MIN DATA 
    ---------------
    A - EXTENDED HOURS
'''
min15_ext_data_dict = {}
data_timestamp_list = data_timestamp_list_0
df = df_data

# Fill the dictionary for "Hour_ext_data"
min15_ext_data_dict["str_date_format"] = str_date_format
min15_ext_data_dict["extended_trading_hours"] = True
min15_ext_data_dict["data_timestamp_list"] = data_timestamp_list
min15_ext_data_dict["df"] = df

# Add to daily trading dictionary
daily_trading_example_dict["15min_ext_data"] = min15_ext_data_dict
# Delete temps
del df, data_timestamp_list


'''
    ---------------
    ### 15MIN DATA 
    ---------------
    B - STANDARD HOURS
'''
min15_std_data_dict = {}
data_timestamp_list = []
for d in data_day_list:
    for h in hour_std_list:
        if h != "16":
            for m in minutes_15_list:
                data_timestamp_list.append(d+" "+h+":"+m)
        else:
            data_timestamp_list.append(d+" "+h+":00")
df = df_data[df_data.date.isin(data_timestamp_list)]

# Fill the dictionary for "Hour_ext_data"
min15_std_data_dict["str_date_format"] = str_date_format
min15_std_data_dict["extended_trading_hours"] = False
min15_std_data_dict["data_timestamp_list"] = data_timestamp_list
min15_std_data_dict["df"] = df

# Add to daily trading dictionary
daily_trading_example_dict["15min_std_data"] = min15_std_data_dict
# Delete temps
del df, data_timestamp_list


example_dict["Day_trading"] = daily_trading_example_dict

'''
    ######################
    ### HOURLY TRADING ###
    ######################
'''
hourly_trading_example_dict = {}
'''
    ---------------
    ### HOURLY DATA 
    ---------------
'''
'''
    ---------------
    ### 30MIN DATA 
    ---------------
'''
'''
    ---------------
    ### 15MIN DATA 
    ---------------
'''