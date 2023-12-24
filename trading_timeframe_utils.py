import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from drl_pfopt.common.data.data_utils import (get_market_calendar_day_list,
                                              get_timeframe_info,
                                              get_str_date_format,
                                             )

def check_timeframes_compatibility(data_timeframe,
                                  data_ext_hours,
                                  trade_timeframe: str = "Data_timeframe",
                                  trade_ext_hours: bool = False,
                                 ):
    '''
        This function checks the compatibility of (data_timeframe, data_ext_hours)
        with (trade_timeframe, trade_ext_hours) and returns a Boolean 
        (timeframes_are_compatible) as output.
        - If trade_timeframe = "Data_timeframe", then the two tuples above
          are set to be equal and timeframes_are_compatible = True.
        - Setting trade_timeframe = "Data_timeframe" aside, there is a total
          of 676 different possible combinations of the input parameters,
          214 of which are compatible:
        - If trade_timeframe = "Day", then regardless of the values of
          data_ext_hours and trade_ext_hours, timeframes_are_compatible is True.
          This gives 52=2x2x13 compatible combinations.
        - If trade_timeframe is "Hour" or "(int)min", then the function
          computes int_trade_timeframe and int_data_timeframe which are integer
          representations in minutes, and checks if int_data_timeframe divides
          int_trade_timeframe. In this case timeframes_are_compatible is True.
          For trade_timeframe ="Hour", there are 3x12=36 compatible cmbinations.
          For trade_timeframe ="(int)min", the data_timeframe must divide the
          trade_timeframe, which leads to 3x42 compatible combinations.
        
        
        :param data_timeframe: str, keyword from ADMISSIBLE_TIMEFRAMES_LIST. The
                                timeframe (i.e. delta t) in the dataset.
        :param data_ext_hours: bool, whether or not the dataset includes entries 
                                from extended trading hours 
                                (9:00-16:00 if False, 8:00-23:00 if True).
        :param trade_timeframe: str, keyword from ADMISSIBLE_TIMEFRAMES_LIST or
                                "Data_timeframe" (default). The timeframe 
                                (i.e. delta t) between two trading timestamps.
        :param trade_ext_hours: bool, False by default. Whether or not trading 
                                is done during extended trading hours.
        
        :return timeframes_are_compatible: bool. True if the tuples (data_timeframe, data_ext_hours) 
                                and with (trade_timeframe, trade_ext_hours) are compatible.
    '''
    
    # Central to this function
    ADMISSIBLE_TIMEFRAME_LIST = ["Day", "Hour",
                             "1min", "2min", "3min", "4min", "5min", "6min",
                             "10min", "12min", "15min", "20min", "30min"]
    admissible_trade_timeframe_list = ADMISSIBLE_TIMEFRAME_LIST.copy()
    admissible_trade_timeframe_list.append("Data_timeframe")
    
    # Init. output
    timeframes_are_compatible = False
    
    # Convert data timeframe to int (unit is minutes)
    if data_timeframe not in ADMISSIBLE_TIMEFRAME_LIST:
        '''
            For good measure only. Will have to be removed when integrated to final function
        '''
        raise ValueError(f"The data_timeframe parameter must be one of the following keywords:\n"\
                             f"{ADMISSIBLE_TIMEFRAME_LIST}")
    else:
        if data_timeframe == "Day":
            '''
                Not necessary but keeping for reference
            '''
            if data_ext_hours:
                int_data_timeframe = 16*60
            else:
                int_data_timeframe = 8*60
        elif data_timeframe == "Hour":
            int_data_timeframe = 60
        else:
            int_data_timeframe = int(data_timeframe[:-3])
            
    # Case of unrecognized trading timeframe
    if trade_timeframe not in admissible_trade_timeframe_list:
        print("ERROR: Data and trading timeframes are incompatible.")
        print(f"==> trading_timeframe must be an element of {admissible_trade_timeframe_list}")
        print(f"data_timeframe = {data_timeframe}, data_ext_hours = {data_ext_hours}")
        print(f"trade_timeframe = {trade_timeframe}, trade_ext_hours = {trade_ext_hours}")
        int_trade_timeframe = -1
    else:
        # Trivial case: same timeframe
        if trade_timeframe == "Data_timeframe":
            trade_timeframe = data_timeframe
            trade_ext_hours = data_ext_hours
            print("The trading and data timeframes are the same.")
            timeframes_are_compatible = True
        
        # Case of daily trading
        ### Extended trading hour booleans are disregarded
        elif trade_timeframe == "Day":
            timeframes_are_compatible = True
        else:
            # Check the extended trading hours
            if trade_ext_hours and (not data_ext_hours):
                print("ERROR: Data and trading timeframes are incompatible.")
                print("==> Trading cannot be over extended trading hours if data is not.")
                print(f"data_timeframe = {data_timeframe}, data_ext_hours = {data_ext_hours}")
                print(f"trade_timeframe = {trade_timeframe}, trade_ext_hours = {trade_ext_hours}")
                timeframes_are_compatible = False
            else:
                if trade_timeframe == "Hour":
                    int_trade_timeframe = 60
                else:
                    int_trade_timeframe = int(trade_timeframe[:-3])
                '''
                    Crux of function: Check divisibility.
                    The quantity int_trade_timeframe/int_data_timeframe
                    is the intra-trading number of timestamps. 
                    Relevant elsewhere.
                '''
                timeframes_are_compatible = ((int_trade_timeframe % int_data_timeframe) == 0)
    # output
    return timeframes_are_compatible

def get_intraday_data_timestamps(data_timeframe,
                                 data_ext_hours):
    '''
        :return n_intraday_timestamps:
    '''
    # Central to this function
    ADMISSIBLE_TIMEFRAME_LIST = ["Day", "Hour",
                             "1min", "2min", "3min", "4min", "5min", "6min",
                             "10min", "12min", "15min", "20min", "30min"]
    if data_timeframe not in ADMISSIBLE_TIMEFRAME_LIST:
        '''
            For good measure only. Will have to be removed when integrated to final function
        '''
        raise ValueError(f"The data_timeframe parameter must be one of the following keywords:\n"\
                             f"{ADMISSIBLE_TIMEFRAME_LIST}")
    else:
        # Day
        if data_timeframe == "Day":
            n_intraday_data_timestamps = 1
            
        # Hour
        elif data_timeframe == "Hour":
            if data_ext_hours:
                n_intraday_data_timestamps = 16
            else:
                n_intraday_data_timestamps = 8
                
        # (int)min
        else:
            int_data_timeframe = int(data_timeframe[:-3])
            int_intra_hour_timestamps = int(60/int_data_timeframe)
            
            if data_ext_hours:
                n_intraday_data_timestamps = 15*int_intra_hour_timestamps+1
            else:
                n_intraday_data_timestamps = 7*int_intra_hour_timestamps+1
    # Output
    return n_intraday_data_timestamps


def get_n_intratrading_timestamps(data_timeframe,
                                  n_data_intraday_timestamps,
                                  data_ext_hours,
                                  trade_timeframe: str = "Data_timeframe",
                                  trade_ext_hours: bool = False,
                                 ):
    '''
        This function should be merged with check_timeframe_divisibility().
        
        :param data_timeframe: str, keyword from ADMISSIBLE_TIMEFRAMES_LIST. The
                                timeframe (i.e. delta t) in the dataset.
        :param n_data_intraday_timestamps: from get_timeframe_info()
        :param data_ext_hours: bool, whether or not the dataset includes entries 
                                from extended trading hours 
                                (9:00-16:00 if False, 8:00-23:00 if True).
        :param trade_timeframe: str, keyword from ADMISSIBLE_TIMEFRAMES_LIST or
                                "Data_timeframe" (default). The timeframe 
                                (i.e. delta t) between two trading timestamps.
        :param trade_ext_hours: bool, False by default. Whether or not trading 
                                is done during extended trading hours.
        
        :return timeframes_are_compatible: bool. True if the tuples (data_timeframe, data_ext_hours) 
                                and with (trade_timeframe, trade_ext_hours) are compatible.
        :return n_intratrading_timestamps: int. No. of data timestamps in every trading period.
    '''
    
    # Central to this function
    ADMISSIBLE_TIMEFRAME_LIST = ["Day", "Hour",
                             "1min", "2min", "3min", "4min", "5min", "6min",
                             "10min", "12min", "15min", "20min", "30min"]
    admissible_trade_timeframe_list = ADMISSIBLE_TIMEFRAME_LIST.copy()
    admissible_trade_timeframe_list.append("Data_timeframe")
    
    # Init. output
    timeframes_are_compatible = False
    
    # Convert data timeframe to int (unit is minutes)
    if data_timeframe not in ADMISSIBLE_TIMEFRAME_LIST:
        '''
            For good measure only. Will have to be removed when integrated to final function
        '''
        raise ValueError(f"The data_timeframe parameter must be one of the following keywords:\n"\
                             f"{ADMISSIBLE_TIMEFRAME_LIST}")
            
    # Case of unrecognized trading timeframe
    if trade_timeframe not in admissible_trade_timeframe_list:
        print("ERROR: Data and trading timeframes are incompatible.")
        print(f"==> trading_timeframe must be an element of {admissible_trade_timeframe_list}")
        print(f"data_timeframe = {data_timeframe}, data_ext_hours = {data_ext_hours}")
        print(f"trade_timeframe = {trade_timeframe}, trade_ext_hours = {trade_ext_hours}")
        n_data_intratrading_timestamps = 0
    else:
        # Trivial case: same timeframe
        if trade_timeframe == "Data_timeframe":
            trade_timeframe = data_timeframe
            trade_ext_hours = data_ext_hours
            print("The trading and data timeframes are the same.")
            timeframes_are_compatible = True
            n_data_intratrading_timestamps = 1
        
        # Case of daily trading
        ### Extended trading hour booleans are disregarded
        elif trade_timeframe == "Day":
            timeframes_are_compatible = True
            n_data_intratrading_timestamps = n_data_intraday_timestamps
        else:
            # Check the extended trading hours
            if trade_ext_hours and (not data_ext_hours):
                print("ERROR: Data and trading timeframes are incompatible.")
                print("==> Trading cannot be over extended trading hours if data is not.")
                print(f"data_timeframe = {data_timeframe}, data_ext_hours = {data_ext_hours}")
                print(f"trade_timeframe = {trade_timeframe}, trade_ext_hours = {trade_ext_hours}")
                timeframes_are_compatible = False
                n_data_intratrading_timestamps = 0
            else:
                # Convert data timeframe to integer
                ## Day (incompatible)
                if data_timeframe == "Day":
                    int_data_timeframe = 7*60
                    '''
                    print("ERROR: Data and trading timeframes are incompatible.")
                    print("==> Trading cannot be over extended trading hours if data is not.")
                    print(f"data_timeframe = {data_timeframe}, data_ext_hours = {data_ext_hours}")
                    print(f"trade_timeframe = {trade_timeframe}, trade_ext_hours = {trade_ext_hours}")
                    n_data_intratrading_timestamps = 0
                    '''
                ## Hour
                elif data_timeframe == "Hour":
                    int_data_timeframe = 60
                ## (int)min
                else:
                    int_data_timeframe = int(data_timeframe[:-3])
                
                # Convert data timeframe to integer
                ## Day (incompatible)
                if trade_timeframe == "Hour":
                    int_trade_timeframe = 60
                else:
                    int_trade_timeframe = int(trade_timeframe[:-3])
                '''
                    Crux of function: Check divisibility.
                    The quantity int_trade_timeframe/int_data_timeframe
                    is the intra-trading number of timestamps. 
                '''
                if ((int_trade_timeframe % int_data_timeframe) > 0):
                    print("ERROR: Data and trading timeframes are incompatible.")
                    print("==> Trading cannot be over extended trading hours if data is not.")
                    print(f"data_timeframe = {data_timeframe}, data_ext_hours = {data_ext_hours}")
                    print(f"trade_timeframe = {trade_timeframe}, trade_ext_hours = {trade_ext_hours}")
                    n_data_intratrading_timestamps = 0
                elif ((int_trade_timeframe % int_data_timeframe) == 0):
                    timeframes_are_compatible = True
                    n_data_intratrading_timestamps = int(int_trade_timeframe/int_data_timeframe)
                    
                
    # output
    return timeframes_are_compatible, n_data_intratrading_timestamps


def drop_last_timestamp(df_data,
                        extended_trading_hours: bool = False
                       ):
    '''
        Drops the rows of input dataframe that correspond to the last
        timestamp of the day ("16:00" or "23:00")
        :param df_data: pd.DataFrame of data obtained from downloader (e.g. Alpaca)
        :param extended_trading_hours: bool, False by default. Determines the last
            timestamp to drop.
            
        :return df: pd.DataFrame. Truncated timeframe.
    '''
    
    timestamp_list_ = list(df_data.date.unique())
    
    if extended_trading_hours:
        last_timestamp = "23:00"
    else:
        last_timestamp = "16:00"
    
    timestamp_list = [x for x in timestamp_list_ if (x[-5:]!=last_timestamp)]
    df = df_data[df_data['date'].isin(timestamp_list)]
    
    return df

def make_intraday_timestamp_list(timeframe,
                                 extended_trading_hours,
                                 str_format: bool = True,
                                 feature_eng_list: bool = True,
                                ):
    '''
        Helper function to build list of intra-day timestamps according to the 
        timeframe and whether or not we are considering extended trading hours.
        Note that the resulting list does not contain timestamps at "16:00" and "23:00".
        
        :param timeframe: str, intraday timeframe.
        :param extended_trading_hours: bool.        
        :param str_format: bool, True by default. If False, then the entries of the output list are datetime.timedelta
                        objects. If True, the timestamps are converted to str.
        :param feature_eng_list: bool, True by default. If False, the timestamps "23:00" or "16:00" are added.
        
        :return intraday_timestamp_list:
    '''
    ADMISSIBLE_TIMEFRAME_LIST = ["Hour",
                             "1min", "2min", "3min", "4min", "5min", "6min",
                             "10min", "12min", "15min", "20min", "30min"] # "Day" not admissible here
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
                    [ "Day", "Hour", "1min", "2min", "3min", "4min", "5min",
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

'''
################################################
#### GET TRADING TIMEFRAME INFO - VERSION 3 ####
################################################
Quasi-final version. Improves version 1 and
takes into account the no. of lookback trading 
periods.
'''
def get_trading_timeframe_info(df_X: pd.DataFrame,
                               trade_timeframe: str = "Data_timeframe",
                               trade_ext_hours: bool = False,
                               N_lookback_trade_prds: int = 1,
                              ):
    '''
        Helper function for FeatureEngDualTF.process_dual_timeframes().
        This function completes 4 tasks:
        (1) Constructs the data timeframe info dictionary.
        (2) Formats the input dataframe by removing the last daily timestamp.
        (3) Computes the number of intra-trading data timestamps.
        (4) Constructs the trading timeframe info dictionary.
        
        Important notes:
        ================
        - The input dataframe and trading timeframe parameters must have  
          the same format as that the AlpacaDownloader. To avoid issues,
          ensure that df_X was produced by the AlpacaDownloader.
          
        - The format of the output dataframe is different from that
          of the AlpacaDownloader. This function removes the last daily
          timestamp, namely "16:00" for standard trading hours and
          "23:00" for extended trading hours. This format is also
          used for the (gym) portfolio optimization environment.
          
        - For task (1), the data timeframe info dictionary is obtained
          using get_timeframe_info() (drl_pfopt.common.data.data_utils).
          The output of this function is then modified to satisfy the
          new dataframe format. The attributes recorded in (1) are:
          
          data_timeframe_info_dict["str_date_format"]: "%Y-%m-%d" or "%Y-%m-%d %H:%M"
          data_timeframe_info_dict["start_date"]: datetime obj for first data day
          data_timeframe_info_dict["end_date"]: datetime obj for day after last data day
          data_timeframe_info_dict["timeframe"]: str keyword for delta t between two datapoints
          data_timeframe_info_dict["extended_trading_hours"]: bool for extended hours
          data_timeframe_info_dict["day_list"]: list of datetime objs in the dataset
          data_timeframe_info_dict["n_intraday_timestamps"]: int no. of data timestamps in one day
          
        - Task (3) above calls the helper function get_n_intratrading_timestamps(),
          which checks the compatibility of the data timeframe info extracted from
          df_X and the input trading timeframe parameters. In total, considering
          standard and extended trading hours, the 13 possible timeframe keywords
          give 676 combinations, with only 214 of which are compatible. For compatible
          timeframes, the function returns n_intratrading_timestamps, which is
          the number of data timestamps in each trading period.
          
        - Task (4) calls two helper functions: make_timestamp_list() and 
          get_market_calendar_day_list(). The output of this part is the
          trading timeframe info dictionary, containing attributes that
          are similar to data_timeframe_info_dict:
                    
          trade_timeframe_info_dict["str_date_format"]
          trade_timeframe_info_dict["start_date"]
          trade_timeframe_info_dict["end_date"]
          trade_timeframe_info_dict["timeframe"]
          trade_timeframe_info_dict["extended_trading_hours"]
          trade_timeframe_info_dict["day_list"]
          trade_timeframe_info_dict["n_intraday_timestamps"]
          
          as well as:
          
          trade_timeframe_info_dict["trade_timestamp_list"]: List of trading timestamps.
          trade_timeframe_info_dict["n_intratrading_timestamps"]: No. of intratrading timestamps
          trade_timeframe_info_dict["N_lookback_trading_prds"]: Last input parameter of this f'n.
          trade_timeframe_info_dict["trading_data_schedule"]: Dictionary of data timestamps for each trading timestamp.
          
        - The trade timestamp list is not as symmetric as the data timestamp list.
          We are assuming that the trading timestamps correspond to the beginning of the
          trading period (when the portfolio is rebalanced). As such, there is no data
          for the first trading timestamp of the first day of the dataset, and similarly
          the last trading timestamp is the first trading period of the day after
          last in the original dataset.
          
        - The keys trading_data_schedule dictionary are the elements of trade_timestamp_list,
          and the corresponding values are lists of data timestamps of length:
            N_lookback_trade_prds*n_intratrading_timestamps.
          This dictionary is called "the schedule" because the new weights of the portfolio
          at a given trading timestamp are computed using the data at the timestamps
          in the corresponding list.
          
        :param df_X: pd.DataFrame of stock data.
        :param trade_timeframe: str,  A keyword in the following list:
                    [ "Data_timeframe" (default), "Day", "Hour",
                     "1min", "2min", "3min", "4min", "5min", "6min",
                     "10min", "12min", "15min", "20min", "30min"]
        :param trade_ext_hours: bool, False by default. If True, trading timestamps
                    are in [9:00, 16:00[, and in [8:00, 23:00[ otherwise.
        
        :return timeframe_dict: dict of output. Its keys are:
                ["df"]: Formatted pd.DataFrame. See (2) above.
                ["data_timeframe_info_dict"]: Info dictionary for data timeframe. See (1).
                ["trade_timeframe_info_dict"]: Info dictionary for data timeframe. See (3) and (4).
    '''
        
    # Initializations
    ADMISSIBLE_TIMEFRAME_LIST = [ "Day", "Hour", "1min", "2min", "3min", "4min", "5min", "6min",
                     "10min", "12min", "15min", "20min", "30min"]
    df = df_X.copy()
    
    # Raise value error if N_lookback_trade_prds is nonpositive
    if N_lookback_trade_prds<=0:
        raise ValueError(f"Number of lookback trading periods must be non-positive.\n"\
                         f"Given value: N_lookback_trade_prds = {N_lookback_trade_prds}."
                        )
    
    ##################################
    ### 1) Get data timeframe info ###
    ##################################
    data_timestamp_list_ = list(df_X.date.unique())
    data_timeframe_info_dict = get_timeframe_info(date_list = data_timestamp_list_)
    # Data timeframe attributes
    data_timeframe = data_timeframe_info_dict["timeframe"]
    data_ext_hours = data_timeframe_info_dict["extended_trading_hours"]
    data_day_list = data_timeframe_info_dict["day_list"]
    
    
    ##########################################################
    ### 2) Remove last daily timestamp (for intraday data) ###
    ##########################################################
    if data_timeframe != "Day":
        if data_timeframe_info_dict["extended_trading_hours"]:
            last_timestamp = "23:00"
        else:
            last_timestamp = "16:00"

        data_timestamp_list = [x for x in data_timestamp_list_ if (x[-5:]!=last_timestamp)]
        df = df[df['date'].isin(data_timestamp_list)]
        # Correct the data_n_intraday_timestamps integer
        data_timeframe_info_dict["n_intraday_timestamps"] = data_timeframe_info_dict["n_intraday_timestamps"]-1
    else:
        data_timestamp_list = data_timestamp_list_
    
    # No. of intraday data timestamps
    data_n_intraday_timestamps = data_timeframe_info_dict["n_intraday_timestamps"]
    
    ####################################################################################
    ### 3) Check timeframe compatibility and get no. of intratrading data timestamps ###
    ####################################################################################
    # Trivial case of trade timeframe
    if trade_timeframe == "Data_timeframe":
        trade_timeframe = data_timeframe
        trade_ext_hours = data_ext_hours
        N_lookback_trade_prds = 1
    
    # Get compatibility of the timeframes and no. of intra-trading data timestamps (OUTPUT)
    timeframes_are_compatible, n_intratrading_timestamps = \
        get_n_intratrading_timestamps(data_timeframe = data_timeframe,
                                      n_data_intraday_timestamps=data_n_intraday_timestamps,
                                      data_ext_hours = data_ext_hours,
                                      trade_timeframe = trade_timeframe,
                                      trade_ext_hours = trade_ext_hours,
                                     )
    
    # Raise ValueError if timeframes are not compatible
    if not timeframes_are_compatible:
        raise ValueError(f"Trading and data timeframes are incompatible. Given parameters:\n"\
                         f"data_timeframe = {data_timeframe}, data_ext_hours = {data_ext_hours}\n"\
                         f"trade_timeframe = {trade_timeframe}, trade_ext_hours = {trade_ext_hours}."\
                        )
    # Raise value error if N_lookback_trade_prds is too high
    if N_lookback_trade_prds*n_intratrading_timestamps >= len(data_timestamp_list):
        raise ValueError(f"Number of lookback trading periods is too high:\n"\
                         f"N_lookback_trade_prds*n_intratrading_timestamps = "\
                         f"{N_lookback_trade_prds*n_intratrading_timestamps} \n"\
                         f"must be lower than len(data_timestamp_list) = {len(data_timestamp_list)}."\
                        )
    ####################################################
    ### 4) Build the trade timeframe info dictionary ###
    ####################################################
    else:
        
        if trade_timeframe == "Day":
            '''
            ##########################
            ### DAILY TRADING CASE ###
            ##########################
            '''
            # String date format and no. of intraday timestamps (OUTPUT)
            trade_str_date_format = "%Y-%m-%d"
            trade_n_intraday_timestamps = 1
            
            # Get temporary trade_day_list
            '''
                The last trading day is one business (NASDAQ) day after the last day in data_day_list.
                We use a temporary day list to get the correct list and start/end dates. In particular,
                we add 10 days to the end date of the data by precaution.
            '''
            trade_day_list_ = get_market_calendar_day_list(start_date = data_day_list[0],
                                                           end_date = data_day_list[-1]+timedelta(days=10)
                                                          )
            # Get trade_day_list and trade_timestamp_list (OUTPUT)
            trade_day_list = trade_day_list_[N_lookback_trade_prds:len(data_day_list)+1] 
            trade_timestamp_list = [x.strftime("%Y-%m-%d") for x in trade_day_list]
            
            # Get trade_start_date and trade_end_date (OUTPUT)
            trade_start_date = trade_day_list[0]
            trade_end_date = trade_day_list_[len(data_day_list)+1]
            del trade_day_list_
            
            # Make trade_data_schedule dictionary (OUTPUT)
            ## Init. schedule dict.
            trade_data_schedule_dict = {}
            ## Fill schedule dict (i_tts = index_trading_timestamp)
            for i_tts in range(len(trade_timestamp_list)):
                trade_data_schedule_dict[trade_timestamp_list[i_tts]]=\
                    data_timestamp_list[i_tts*n_intratrading_timestamps\
                                        :(i_tts+N_lookback_trade_prds)*n_intratrading_timestamps] 

        else:
            '''
            #######################################
            ### HOUR and (int)min TRADING CASES ###
            #######################################
            '''
            # Trading timeframe integer in minutes
            if trade_timeframe == "Hour":
                int_trade_timeframe = 60
            else:
                int_trade_timeframe = int(trade_timeframe[:-3])
            
            # String date format (OUTPUT)
            trade_str_date_format = "%Y-%m-%d %H:%M"
            
            # Get temporary trade_day_list_ and trade_timestamp_list_
            '''
                The last trading day is one business (NASDAQ) day after the last day in data_day_list.
                We use a temporary day list to get the correct list and start/end dates. In particular,
                we add 10 days to the end date of the data by precaution.
            '''
            trade_day_list_ = get_market_calendar_day_list(start_date = data_day_list[0],
                                                           end_date = data_day_list[-1]+timedelta(days=10)
                                                          )
            trade_timestamp_list_ = make_timestamp_list(day_list = trade_day_list_[:len(data_day_list)],
                                                       timeframe = trade_timeframe,
                                                       extended_trading_hours=trade_ext_hours,
                                                       str_format = True,)
            
            # Get trade_timestamp_list and trade_n_intraday_timestamps (OUTPUT)
            ### Omit the lookback trading periods from trade_timestamp_list_
            trade_timestamp_list = trade_timestamp_list_[N_lookback_trade_prds:]
            '''
                ### Compute no. of trading intraday timestamps and add last entry of trade_timestamp_list
                The last trading timestamp is the first hour of trade_day_list_[len(data_day_list)],
                which is the day after data_day_list[-1].
            '''
            if trade_ext_hours:
                trade_n_intraday_timestamps = 15*int(60/int_trade_timeframe)
                trade_timestamp_list.append((trade_day_list_[len(data_day_list)]+timedelta(hours=8))\
                                            .strftime(trade_str_date_format))
            else:
                trade_n_intraday_timestamps = 7*int(60/int_trade_timeframe)
                trade_timestamp_list.append((trade_day_list_[len(data_day_list)]+timedelta(hours=9))\
                                            .strftime(trade_str_date_format))
            
            
            # Get trade_day_list, trade_start_date, and trade_end_date (OUTPUT)
            ## Date of first trading day in the timestamp list
            trade_start_date = datetime.strptime(trade_timestamp_list[0][:10], "%Y-%m-%d")
            ## Date of the last trading day
            trade_end_date = trade_day_list_[len(data_day_list)+1]
            trade_day_list = trade_day_list_[trade_day_list_.index(trade_start_date)\
                                            :trade_day_list_.index(trade_end_date)]
            ## Delete temp lists
            del trade_day_list_, trade_timestamp_list_
            
            # Build trading/data schedule dictionary (OUTPUT)
            '''
                Using list.index() is not optimal, but this function should be
                called only once.
            '''
            ## Init. schedule dict.
            trade_data_schedule_dict = {}
            ## Assign data timestamp lists to next trading timestamps,
            ## except last one (i_tts = index_trading_timestamp, beware of its values)
            for i_tts in range(len(trade_timestamp_list)-1):
                trade_data_schedule_dict[trade_timestamp_list[i_tts]]=\
                    data_timestamp_list[data_timestamp_list.index(trade_timestamp_list[i_tts])\
                                                -n_intratrading_timestamps*N_lookback_trade_prds
                                            :data_timestamp_list.index(trade_timestamp_list[i_tts])]
            # Assign data timestamp list to last trading timestamp
            trade_data_schedule_dict[trade_timestamp_list[-1]] = \
                data_timestamp_list[-n_intratrading_timestamps*N_lookback_trade_prds:]
        
        # Fill trading timeframe info dictionary
        trade_timeframe_info_dict = {}
        trade_timeframe_info_dict["str_date_format"] = trade_str_date_format
        trade_timeframe_info_dict["start_date"] = trade_start_date
        trade_timeframe_info_dict["end_date"] = trade_end_date
        trade_timeframe_info_dict["timeframe"] = trade_timeframe
        trade_timeframe_info_dict["extended_trading_hours"] = trade_ext_hours
        trade_timeframe_info_dict["day_list"] = trade_day_list
        trade_timeframe_info_dict["n_intraday_timestamps"] = trade_n_intraday_timestamps
        trade_timeframe_info_dict["trade_timestamp_list"] = trade_timestamp_list
        trade_timeframe_info_dict["trading_data_schedule"]= trade_data_schedule_dict
        trade_timeframe_info_dict["n_intratrading_timestamps"] = n_intratrading_timestamps
        trade_timeframe_info_dict["N_lookback_trade_prds"] = N_lookback_trade_prds
        
        # Prepare output
        timeframe_dict = {}
        timeframe_dict["df"] = df
        timeframe_dict["data_timeframe_info_dict"] = data_timeframe_info_dict
        timeframe_dict["trade_timeframe_info_dict"] = trade_timeframe_info_dict
    
    return timeframe_dict

'''
    ############################################################
    #### Get trading timeframe info - Preliminary version 0 ####
    ############################################################
    Backup of preliminary version written in Dual_Timeframe_Env_XP 
    notebook, added the v0 suffix for clarity.
    This version is quick and dirty, and doesn't use the
    N_lookback_trade_prds parameter.
'''

def get_trading_timeframe_info_v0(df_X: pd.DataFrame,
                                trade_timeframe: str = "Data_timeframe",
                                trade_ext_hours: bool = False,
                                #N_lookback_trade_prds: int = 1,
                              ):
    '''
        Draft version of the function that processes the timeframes 
        and that is called by the FeatureEngDualTF.preprocess_data().
        This preliminary version uses N_lookback_trade_prds =1.
        The outputs of this function are:
        =================================
        
        a) Usual timeframe info attributes:
        trade_timeframe_info_dict["str_date_format"]
        trade_timeframe_info_dict["start_date"]
        trade_timeframe_info_dict["end_date"]
        trade_timeframe_info_dict["timeframe"]
        trade_timeframe_info_dict["extended_trading_hours"]
        trade_timeframe_info_dict["day_list"]
        trade_timeframe_info_dict["n_intraday_timestamps"]
        
        b) New timeframe info attributes:
        trade_timeframe_info_dict["trade_timestamp_list"]
        trade_timeframe_info_dict["n_intratrading_timestamps"]
        trade_timeframe_info_dict["N_lookback_trading_prds"]
        trade_timeframe_info_dict["trading_data_schedule"]
        
        BEWARE: The N_lookback_days param is redundant...
    '''
        
    # Initializations
    ADMISSIBLE_TIMEFRAME_LIST = ["Day", "Hour",
                             "1min", "2min", "3min", "4min", "5min", "6min",
                             "10min", "12min", "15min", "20min", "30min"]
    df = df_X.copy()
    
    # DEBUG/PRELIM
    N_lookback_trade_prds = 1 
    
    ##################################
    ### 1) Get data timeframe info ###
    ##################################
    data_timestamp_list_ = list(df_X.date.unique())
    data_timeframe_info_dict = get_timeframe_info(date_list = data_timestamp_list_)
    # Data timeframe attributes
    data_timeframe = data_timeframe_info_dict["timeframe"]
    data_ext_hours = data_timeframe_info_dict["extended_trading_hours"]
    data_day_list = data_timeframe_info_dict["day_list"]
    
    
    ##########################################################
    ### 2) Remove last daily timestamp (for intraday data) ###
    ##########################################################
    if data_timeframe != "Day":
        if data_timeframe_info_dict["extended_trading_hours"]:
            last_timestamp = "23:00"
        else:
            last_timestamp = "16:00"

        data_timestamp_list = [x for x in data_timestamp_list_ if (x[-5:]!=last_timestamp)]
        df = df[df['date'].isin(data_timestamp_list)]
        data_timeframe_info_dict["n_intraday_timestamps"] = data_timeframe_info_dict["n_intraday_timestamps"]-1
    else:
        data_timestamp_list = data_timestamp_list_
    
    # No. of intraday data timestamps
    data_n_intraday_timestamps = data_timeframe_info_dict["n_intraday_timestamps"]
    
    ####################################################################################
    ### 3) Check timeframe compatibility and get no. of intratrading data timestamps ###
    ####################################################################################
    # Trivial case of trade timeframe
    if trade_timeframe == "Data_timeframe":
        trade_timeframe = data_timeframe
        trade_ext_hours = data_ext_hours
        N_lookback_trade_prds = 1
    
    # Get compatibility of the timeframes and no. of intra-trading data timestamps
    timeframes_are_compatible, n_intratrading_timestamps = \
        get_n_intratrading_timestamps(data_timeframe = data_timeframe,
                                      n_data_intraday_timestamps=data_n_intraday_timestamps,
                                      data_ext_hours = data_ext_hours,
                                      trade_timeframe = trade_timeframe,
                                      trade_ext_hours = trade_ext_hours,
                                     )
    
    # Raise ValueError if timeframes are not compatible
    if not timeframes_are_compatible:
        raise ValueError(f"Trading and data timeframes are incompatible. Given parameters:\n"\
                         f"data_timeframe = {data_timeframe}, data_ext_hours = {data_ext_hours}\n"\
                         f"trade_timeframe = {trade_timeframe}, trade_ext_hours = {trade_ext_hours}\n."\
                        )
    
    ####################################################
    ### 4) Build the trade timeframe info dictionary ###
    ####################################################
    else:
        
        if trade_timeframe == "Day":
            '''
            ##########################
            ### DAILY TRADING CASE ###
            ##########################
        
                ####################################################
                ### Part A: Obvious trading timeframe attributes ###
                ####################################################
            '''
            # String date format and no. of intraday timestamps
            trade_str_date_format = "%Y-%m-%d"
            trade_n_intraday_timestamps = 1
            '''
                # To be modified when N_lookback_trade_prds is generalized
            
            # Compute N_lookback_trade_days = int(N_lookback_trade_prds/trade_n_intraday_timestamps)
            N_lookback_trade_days = N_lookback_trade_prds
            
            # Check N_lookback_trade_prds is admissible
            if N_lookback_trade_days>= len(data_day_list)+1:
                raise ValueError(f"N_lookback_trade_prds = {N_lookback_trade_prds} is too high."\
                                 f"The number of days in the data set is {len(data_day_list)}")
            '''
            
            
            '''
                ###########################################################
                ### Part B: Get trading day and trading timestamp lists ###
                ###########################################################
                This is the easiest case. The trading timestamp
                list should be a list of str's.
            '''
            # Get trade_day_list
            '''
                The last trading day is one business (NASDAQ) day after the last day in data_day_list.
                We use a temporary day list to get the correct list and start/end dates. In particular,
                we add 10 days to the end date of the data by precaution.
            '''
            trade_day_list_ = get_market_calendar_day_list(start_date = data_day_list[0],
                                                           end_date = data_day_list[-1]+timedelta(days=10)
                                                          )
            '''
                NOTE: To be modified for N_lookback_trade_prds>1
            '''
            trade_day_list = trade_day_list_[:len(data_day_list)+1] 
            trade_start_date = trade_day_list[0]
            trade_end_date = trade_day_list_[len(data_day_list)+1]
            del trade_day_list_
            
            trade_timestamp_list = [x.strftime("%Y-%m-%d") for x in trade_day_list]
            
                        
            '''
                ####################################################
                ### Part C: Get trading/data schedule dictionary ###
                ####################################################
                The keys for this dictionary are the entries of trade_timestamp_list.
            '''
            # Init. schedule dict.
            trade_data_schedule_dict = {}
            
            # Fill schedule dict (i_tts = index_trading_timestamp)
            '''
                NOTE: To be modified for N_lookback_trade_prds>1
                        Had to modify this. Beware of i_tts
            '''
            trade_data_schedule_dict[trade_timestamp_list[0]]= []
            for i_tts in range(1,len(trade_timestamp_list)):
                trade_data_schedule_dict[trade_timestamp_list[i_tts]]=\
                    data_timestamp_list[(i_tts-1)*n_intratrading_timestamps\
                                        :(i_tts)*n_intratrading_timestamps] 
            #pass
        
        elif trade_timeframe == "Hour":
            '''
            ###########################
            ### HOURLY TRADING CASE ###
            ###########################
            
                ####################################################
                ### Part A: Obvious trading timeframe attributes ###
                ####################################################
            '''
            # String date format and no. of intraday timestamps
            trade_str_date_format = "%Y-%m-%d %H:%M"
            if trade_ext_hours:
                trade_n_intraday_timestamps = 15
            else:
                trade_n_intraday_timestamps = 7
            '''
                # To be modified when N_lookback_trade_prds is generalized
            
            # Compute N_lookback_trade_days = int(N_lookback_trade_prds/trade_n_intraday_timestamps)
            N_lookback_trade_days = N_lookback_trade_prds
            
            # Check N_lookback_trade_prds is admissible
            if N_lookback_trade_days>= len(data_day_list)+1:
                raise ValueError(f"N_lookback_trade_prds = {N_lookback_trade_prds} is too high."\
                                 f"The number of days in the data set is {len(data_day_list)}")
            '''
            
            
            '''
                ###########################################################
                ### Part B: Get trading day and trading timestamp lists ###
                ###########################################################
                This is the easiest case. The trading timestamp
                list should be a list of str's.
            '''
            # Get trade_day_list
            '''
                The last trading day is one business (NASDAQ) day after the last day in data_day_list.
                We use a temporary day list to get the correct list and start/end dates. In particular,
                we add 10 days to the end date of the data by precaution.
            '''
            trade_day_list_ = get_market_calendar_day_list(start_date = data_day_list[0],
                                                           end_date = data_day_list[-1]+timedelta(days=10)
                                                          )
            '''
                NOTE: To be modified for N_lookback_trade_prds>1
            '''
            trade_day_list = trade_day_list_[:len(data_day_list)+1] 
            trade_start_date = trade_day_list[0]
            trade_end_date = trade_day_list_[len(data_day_list)+1]
            del trade_day_list_
            
            
            # Make trading timestamp list
            ### Get intraday timestamp list for all trading days except the last one
            trade_timestamp_list = make_timestamp_list(day_list = trade_day_list[:-1],
                                                       timeframe = trade_timeframe,
                                                       extended_trading_hours=trade_ext_hours,
                                                       str_format = True,)
            ### Only first hour of last entry of trade_day_list has enough data
            if trade_ext_hours:
                trade_timestamp_list.append((trade_day_list[-1]+timedelta(hours=8))\
                                            .strftime(trade_str_date_format))
            else:
                trade_timestamp_list.append((trade_day_list[-1]+timedelta(hours=9))\
                                            .strftime(trade_str_date_format))
            
                        
            '''
                ####################################################
                ### Part C: Get trading/data schedule dictionary ###
                ####################################################
                The keys for this dictionary are the entries of trade_timestamp_list.
            '''
            # Init. schedule dict.
            trade_data_schedule_dict = {}
            
            # Fill schedule dict (i_tts = index_trading_timestamp)
            '''
                NOTE: To be modified for N_lookback_trade_prds>1
                        Had to modify this. Beware of i_tts
            '''
            # First trade timestamp has no data if trade_ext_hours = data_ext_hours
            if trade_ext_hours == data_ext_hours:
                trade_data_schedule_dict[trade_timestamp_list[0]]= []
            else: # Case of trade_ext_hours = False and data_ext_hours=True
                trade_data_schedule_dict[trade_timestamp_list[0]]= data_timestamp_list[:n_intratrading_timestamps]            
            # Assign data timestamp lists to next trading timestamps,
            # except last one
            for i_tts in range(1,len(trade_timestamp_list)-1):                
                '''
                    Using list.index() is not optimal but works here.
                '''
                trade_data_schedule_dict[trade_timestamp_list[i_tts]]=\
                    data_timestamp_list[data_timestamp_list.index(trade_timestamp_list[i_tts])\
                                                -n_intratrading_timestamps
                                            :data_timestamp_list.index(trade_timestamp_list[i_tts])]
            # Assign data timestamp list to last trading timestamp
            trade_data_schedule_dict[trade_timestamp_list[-1]] = \
                data_timestamp_list[-n_intratrading_timestamps:]
        
        else:
            '''
            #############################
            ### (int)min TRADING CASE ###
            #############################
            
                ####################################################
                ### Part A: Obvious trading timeframe attributes ###
                ####################################################
            '''
            # Trading timeframe integer
            int_trade_timeframe = int(trade_timeframe[:-3])
            
            # String date format and no. of intraday timestamps
            trade_str_date_format = "%Y-%m-%d %H:%M"
            if trade_ext_hours:
                trade_n_intraday_timestamps = 15*int(60/int_trade_timeframe)
            else:
                trade_n_intraday_timestamps = 7*int(60/int_trade_timeframe)
            '''
                # To be modified when N_lookback_trade_prds is generalized
            
            # Compute N_lookback_trade_days = int(N_lookback_trade_prds/trade_n_intraday_timestamps)
            N_lookback_trade_days = N_lookback_trade_prds
            
            # Check N_lookback_trade_prds is admissible
            if N_lookback_trade_days>= len(data_day_list)+1:
                raise ValueError(f"N_lookback_trade_prds = {N_lookback_trade_prds} is too high."\
                                 f"The number of days in the data set is {len(data_day_list)}")
            '''
            
            
            '''
                ###########################################################
                ### Part B: Get trading day and trading timestamp lists ###
                ###########################################################
                This is the easiest case. The trading timestamp
                list should be a list of str's.
            '''
            # Get trade_day_list
            '''
                The last trading day is one business (NASDAQ) day after the last day in data_day_list.
                We use a temporary day list to get the correct list and start/end dates. In particular,
                we add 10 days to the end date of the data by precaution.
            '''
            trade_day_list_ = get_market_calendar_day_list(start_date = data_day_list[0],
                                                           end_date = data_day_list[-1]+timedelta(days=10)
                                                          )
            '''
                NOTE: To be modified for N_lookback_trade_prds>1
            '''
            trade_day_list = trade_day_list_[:len(data_day_list)+1] 
            trade_start_date = trade_day_list[0]
            trade_end_date = trade_day_list_[len(data_day_list)+1]
            del trade_day_list_
            
            
            # Make trading timestamp list
            ### Get intraday timestamp list for all trading days except the last one
            trade_timestamp_list = make_timestamp_list(day_list = trade_day_list[:-1],
                                                       timeframe = trade_timeframe,
                                                       extended_trading_hours=trade_ext_hours,
                                                       str_format = True,)
            ### Only first hour of last entry of trade_day_list has enough data
            if trade_ext_hours:
                trade_timestamp_list.append((trade_day_list[-1]+timedelta(hours=8))\
                                            .strftime(trade_str_date_format))
            else:
                trade_timestamp_list.append((trade_day_list[-1]+timedelta(hours=9))\
                                            .strftime(trade_str_date_format))
            
                        
            '''
                ####################################################
                ### Part C: Get trading/data schedule dictionary ###
                ####################################################
                The keys for this dictionary are the entries of trade_timestamp_list.
            '''
            # Init. schedule dict.
            trade_data_schedule_dict = {}
            
            # Fill schedule dict (i_tts = index_trading_timestamp)
            '''
                NOTE: To be modified for N_lookback_trade_prds>1
                        Had to modify this. Beware of i_tts
            '''
            # First trade timestamp has no data if trade_ext_hours = data_ext_hours
            if trade_ext_hours == data_ext_hours:
                trade_data_schedule_dict[trade_timestamp_list[0]]= []
            else: # Case of trade_ext_hours = False and data_ext_hours=True
                #trade_data_schedule_dict[trade_timestamp_list[0]]= data_timestamp_list[:n_intratrading_timestamps]            
                trade_data_schedule_dict[trade_timestamp_list[0]]= \
                    data_timestamp_list[data_timestamp_list.index(trade_timestamp_list[0])-n_intratrading_timestamps\
                                       :data_timestamp_list.index(trade_timestamp_list[0])]
            # Assign data timestamp lists to next trading timestamps,
            # except last one
            for i_tts in range(1,len(trade_timestamp_list)-1):                
                '''
                    Using list.index() is not optimal but works here.
                '''
                trade_data_schedule_dict[trade_timestamp_list[i_tts]]=\
                    data_timestamp_list[data_timestamp_list.index(trade_timestamp_list[i_tts])\
                                                -n_intratrading_timestamps
                                            :data_timestamp_list.index(trade_timestamp_list[i_tts])]
            # Assign data timestamp list to last trading timestamp
            trade_data_schedule_dict[trade_timestamp_list[-1]] = \
                data_timestamp_list[-n_intratrading_timestamps:]
        
        # Fill trading timeframe info dictionary
        trade_timeframe_info_dict = {}
        trade_timeframe_info_dict["str_date_format"] = trade_str_date_format
        trade_timeframe_info_dict["start_date"] = trade_start_date
        trade_timeframe_info_dict["end_date"] = trade_end_date
        trade_timeframe_info_dict["timeframe"] = trade_timeframe
        trade_timeframe_info_dict["extended_trading_hours"] = trade_ext_hours
        trade_timeframe_info_dict["day_list"] = trade_day_list
        trade_timeframe_info_dict["n_intraday_timestamps"] = trade_n_intraday_timestamps
        trade_timeframe_info_dict["trade_timestamp_list"] = trade_timestamp_list
        trade_timeframe_info_dict["trading_data_schedule"]= trade_data_schedule_dict
        trade_timeframe_info_dict["n_intratrading_timestamps"] = n_intratrading_timestamps
        trade_timeframe_info_dict["N_lookback_trade_prds"] = N_lookback_trade_prds
        
        # Prepare output
        timeframe_dict = {}
        timeframe_dict["df"] = df
        timeframe_dict["data_timeframe_info_dict"] = data_timeframe_info_dict
        timeframe_dict["trade_timeframe_info_dict"] = trade_timeframe_info_dict
    
    return timeframe_dict


'''
################################################
#### GET TRADING TIMEFRAME INFO - VERSION 1 ####
################################################
Shortened version of version 0 

'''
def get_trading_timeframe_info_(df_X: pd.DataFrame,
                                trade_timeframe: str = "Data_timeframe",
                                trade_ext_hours: bool = False,
                                #N_lookback_trade_prds: int = 1,
                              ):
    '''
        Draft version of the function that processes the timeframes 
        and that is called by the FeatureEngDualTF.preprocess_data().
        This preliminary version uses N_lookback_trade_prds =1.
        The outputs of this function are:
        =================================
        
        a) Usual timeframe info attributes:
        trade_timeframe_info_dict["str_date_format"]
        trade_timeframe_info_dict["start_date"]
        trade_timeframe_info_dict["end_date"]
        trade_timeframe_info_dict["timeframe"]
        trade_timeframe_info_dict["extended_trading_hours"]
        trade_timeframe_info_dict["day_list"]
        trade_timeframe_info_dict["n_intraday_timestamps"]
        
        b) New timeframe info attributes:
        trade_timeframe_info_dict["trade_timestamp_list"]
        trade_timeframe_info_dict["n_intratrading_timestamps"]
        trade_timeframe_info_dict["N_lookback_trading_prds"]
        trade_timeframe_info_dict["trading_data_schedule"]
        
        BEWARE: The N_lookback_days param is redundant...
    '''
        
    # Initializations
    ADMISSIBLE_TIMEFRAME_LIST = ["Day", "Hour",
                             "1min", "2min", "3min", "4min", "5min", "6min",
                             "10min", "12min", "15min", "20min", "30min"]
    df = df_X.copy()
    
    # DEBUG/PRELIM
    N_lookback_trade_prds = 1 
    
    ##################################
    ### 1) Get data timeframe info ###
    ##################################
    data_timestamp_list_ = list(df_X.date.unique())
    data_timeframe_info_dict = get_timeframe_info(date_list = data_timestamp_list_)
    # Data timeframe attributes
    data_timeframe = data_timeframe_info_dict["timeframe"]
    data_ext_hours = data_timeframe_info_dict["extended_trading_hours"]
    data_day_list = data_timeframe_info_dict["day_list"]
    
    
    ##########################################################
    ### 2) Remove last daily timestamp (for intraday data) ###
    ##########################################################
    if data_timeframe != "Day":
        if data_timeframe_info_dict["extended_trading_hours"]:
            last_timestamp = "23:00"
        else:
            last_timestamp = "16:00"

        data_timestamp_list = [x for x in data_timestamp_list_ if (x[-5:]!=last_timestamp)]
        df = df[df['date'].isin(data_timestamp_list)]
        data_timeframe_info_dict["n_intraday_timestamps"] = data_timeframe_info_dict["n_intraday_timestamps"]-1
    else:
        data_timestamp_list = data_timestamp_list_
    
    # No. of intraday data timestamps
    data_n_intraday_timestamps = data_timeframe_info_dict["n_intraday_timestamps"]
    
    ####################################################################################
    ### 3) Check timeframe compatibility and get no. of intratrading data timestamps ###
    ####################################################################################
    # Trivial case of trade timeframe
    if trade_timeframe == "Data_timeframe":
        trade_timeframe = data_timeframe
        trade_ext_hours = data_ext_hours
        N_lookback_trade_prds = 1
    
    # Get compatibility of the timeframes and no. of intra-trading data timestamps
    timeframes_are_compatible, n_intratrading_timestamps = \
        get_n_intratrading_timestamps(data_timeframe = data_timeframe,
                                      n_data_intraday_timestamps=data_n_intraday_timestamps,
                                      data_ext_hours = data_ext_hours,
                                      trade_timeframe = trade_timeframe,
                                      trade_ext_hours = trade_ext_hours,
                                     )
    
    # Raise ValueError if timeframes are not compatible
    if not timeframes_are_compatible:
        raise ValueError(f"Trading and data timeframes are incompatible. Given parameters:\n"\
                         f"data_timeframe = {data_timeframe}, data_ext_hours = {data_ext_hours}\n"\
                         f"trade_timeframe = {trade_timeframe}, trade_ext_hours = {trade_ext_hours}\n."\
                        )
    
    ####################################################
    ### 4) Build the trade timeframe info dictionary ###
    ####################################################
    else:
        
        if trade_timeframe == "Day":
            '''
            ##########################
            ### DAILY TRADING CASE ###
            ##########################
        
                ####################################################
                ### Part A: Obvious trading timeframe attributes ###
                ####################################################
            '''
            # String date format and no. of intraday timestamps
            trade_str_date_format = "%Y-%m-%d"
            trade_n_intraday_timestamps = 1
            '''
                # To be modified when N_lookback_trade_prds is generalized
            
            # Compute N_lookback_trade_days = int(N_lookback_trade_prds/trade_n_intraday_timestamps)
            N_lookback_trade_days = N_lookback_trade_prds
            
            # Check N_lookback_trade_prds is admissible
            if N_lookback_trade_days>= len(data_day_list)+1:
                raise ValueError(f"N_lookback_trade_prds = {N_lookback_trade_prds} is too high."\
                                 f"The number of days in the data set is {len(data_day_list)}")
            '''
            
            
            '''
                ###########################################################
                ### Part B: Get trading day and trading timestamp lists ###
                ###########################################################
                This is the easiest case. The trading timestamp
                list should be a list of str's.
            '''
            # Get trade_day_list
            '''
                The last trading day is one business (NASDAQ) day after the last day in data_day_list.
                We use a temporary day list to get the correct list and start/end dates. In particular,
                we add 10 days to the end date of the data by precaution.
            '''
            trade_day_list_ = get_market_calendar_day_list(start_date = data_day_list[0],
                                                           end_date = data_day_list[-1]+timedelta(days=10)
                                                          )
            '''
                NOTE: To be modified for N_lookback_trade_prds>1
            '''
            trade_day_list = trade_day_list_[:len(data_day_list)+1] 
            trade_start_date = trade_day_list[0]
            trade_end_date = trade_day_list_[len(data_day_list)+1]
            del trade_day_list_
            
            trade_timestamp_list = [x.strftime("%Y-%m-%d") for x in trade_day_list]
            
                        
            '''
                ####################################################
                ### Part C: Get trading/data schedule dictionary ###
                ####################################################
                The keys for this dictionary are the entries of trade_timestamp_list.
            '''
            # Init. schedule dict.
            trade_data_schedule_dict = {}
            
            # Fill schedule dict (i_tts = index_trading_timestamp)
            '''
                NOTE: To be modified for N_lookback_trade_prds>1
                        Had to modify this. Beware of i_tts
            '''
            trade_data_schedule_dict[trade_timestamp_list[0]]= []
            for i_tts in range(1,len(trade_timestamp_list)):
                trade_data_schedule_dict[trade_timestamp_list[i_tts]]=\
                    data_timestamp_list[(i_tts-1)*n_intratrading_timestamps\
                                        :(i_tts)*n_intratrading_timestamps] 

        else:
            '''
            #######################################
            ### HOUR and (int)min TRADING CASES ###
            #######################################
            
            '''
            # Trading timeframe integer in minutes
            if trade_timeframe == "Hour":
                int_trade_timeframe = 60
            else:
                int_trade_timeframe = int(trade_timeframe[:-3])
            
            # String date format (OUTPUT)
            trade_str_date_format = "%Y-%m-%d %H:%M"
            
            # Get trade_day_list, trade_start_date, trade_end_date (OUTPUT)
            '''
                The last trading day is one business (NASDAQ) day after the last day in data_day_list.
                We use a temporary day list to get the correct list and start/end dates. In particular,
                we add 10 days to the end date of the data by precaution.
            '''
            trade_day_list_ = get_market_calendar_day_list(start_date = data_day_list[0],
                                                           end_date = data_day_list[-1]+timedelta(days=10)
                                                          )
            trade_day_list = trade_day_list_[:len(data_day_list)+1] 
            trade_start_date = trade_day_list[0]
            trade_end_date = trade_day_list_[len(data_day_list)+1]
            del trade_day_list_
            
            # Get intraday timestamp list for all trading days except the last one (finished atnext step)
            trade_timestamp_list = make_timestamp_list(day_list = trade_day_list[:-1],
                                                       timeframe = trade_timeframe,
                                                       extended_trading_hours=trade_ext_hours,
                                                       str_format = True,)
            
            # No. of trading intraday timestamps and finish trading timestamp lists (OUTPUT)
            if trade_ext_hours:
                trade_n_intraday_timestamps = 15*int(60/int_trade_timeframe)
                trade_timestamp_list.append((trade_day_list[-1]+timedelta(hours=8))\
                                            .strftime(trade_str_date_format))
            else:
                trade_n_intraday_timestamps = 7*int(60/int_trade_timeframe)
                trade_timestamp_list.append((trade_day_list[-1]+timedelta(hours=9))\
                                            .strftime(trade_str_date_format))
           
                        
            # Build trading/data schedule dictionary (OUTPUT)
            '''
                Using list.index() is not optimal, but this function should be
                called only once.
            '''
            ## Init. schedule dict.
            trade_data_schedule_dict = {}
            ## First trade timestamp has no data if trade_ext_hours = data_ext_hours
            if trade_ext_hours == data_ext_hours:
                trade_data_schedule_dict[trade_timestamp_list[0]]= []
            else: # Case of trade_ext_hours = False and data_ext_hours=True           
                trade_data_schedule_dict[trade_timestamp_list[0]]= \
                    data_timestamp_list[data_timestamp_list.index(trade_timestamp_list[0])-n_intratrading_timestamps\
                                       :data_timestamp_list.index(trade_timestamp_list[0])]
            ## Assign data timestamp lists to next trading timestamps,
            ## except last one (i_tts = index_trading_timestamp, beware of its values)
            for i_tts in range(1,len(trade_timestamp_list)-1):
                trade_data_schedule_dict[trade_timestamp_list[i_tts]]=\
                    data_timestamp_list[data_timestamp_list.index(trade_timestamp_list[i_tts])\
                                                -n_intratrading_timestamps
                                            :data_timestamp_list.index(trade_timestamp_list[i_tts])]
            # Assign data timestamp list to last trading timestamp
            trade_data_schedule_dict[trade_timestamp_list[-1]] = \
                data_timestamp_list[-n_intratrading_timestamps:]
        
        # Fill trading timeframe info dictionary
        trade_timeframe_info_dict = {}
        trade_timeframe_info_dict["str_date_format"] = trade_str_date_format
        trade_timeframe_info_dict["start_date"] = trade_start_date
        trade_timeframe_info_dict["end_date"] = trade_end_date
        trade_timeframe_info_dict["timeframe"] = trade_timeframe
        trade_timeframe_info_dict["extended_trading_hours"] = trade_ext_hours
        trade_timeframe_info_dict["day_list"] = trade_day_list
        trade_timeframe_info_dict["n_intraday_timestamps"] = trade_n_intraday_timestamps
        trade_timeframe_info_dict["trade_timestamp_list"] = trade_timestamp_list
        trade_timeframe_info_dict["trading_data_schedule"]= trade_data_schedule_dict
        trade_timeframe_info_dict["n_intratrading_timestamps"] = n_intratrading_timestamps
        trade_timeframe_info_dict["N_lookback_trade_prds"] = N_lookback_trade_prds
        
        # Prepare output
        timeframe_dict = {}
        timeframe_dict["df"] = df
        timeframe_dict["data_timeframe_info_dict"] = data_timeframe_info_dict
        timeframe_dict["trade_timeframe_info_dict"] = trade_timeframe_info_dict
    
    return timeframe_dict