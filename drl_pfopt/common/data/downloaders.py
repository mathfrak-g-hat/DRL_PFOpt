######################################################################
### Deep RL portfolio optimization - Alpaca and Yahoo Downloaders  ###
######################################################################
### 2023/04/10, A.J. Zerouali
'''
    Notes: 
    1) Visit: https://github.com/alpacahq/alpaca-trade-api-python
    alpaca-trade-api will be deprecated in 2023.
    As such it is better to use their current SDK alpaca-py:
    https://github.com/alpacahq/alpaca-py
    2) 
'''

from __future__ import print_function, division
from builtins import range

### Basics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_market_calendars as pd_mkt_cals


### Alpaca
from datetime import datetime, timedelta, date
import alpaca
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

### Yahoo
import yfinance as yf

from drl_pfopt.common.data.data_utils import get_market_calendar_day_list

### Timeframe list
#### The timeframe is either "Day", "Hour", or "_div_min",
#### with _div_ a divisor of 60 strictly lower than 60.
ADMISSIBLE_TIMEFRAME_LIST = ["Day", "Hour",
                             "1min", "2min", "3min", "4min", "5min", "6min",
                             "10min", "12min", "15min", "20min", "30min"]

'''
    ##################################
    #####   ALPACA DOWNLOADER    #####
    ##################################
'''
###################################
### Downloader Class for Alpaca ###
###################################
### 23/04/10, AJ Zerouali
### Updated: Modified to support larger number of intra-hour timeframes.
class AlpacaDownloader:
    '''
        Helper class for downloading stock historical data 
        from Alpaca. See docstring of fetch_data() for
        detailed description.
        
        ATTRIBUTES:
        ===========
        
        :param api_key: Alpaca API key.
        :param api_secret: Alpaca API secret code.
        
        METHODS:
        ========
        
        fetch_data(): Main function for downloading and returning
                    formatted dataframe.
        check_timeframe(): Helper function for checking admissibility
                    of the timeframe keyword.
        download_Alpaca_df(): Method to download raw data using the
                    Alpaca API (alpaca-py package).
        format_download_columns(): Method to create the "date" and "tic"
                    columns from the raw data, and for re-ordering the
                    input columns.
        make_timestamp_list(): Helper method to create the timestamp
                    list of the output dataset.
        fill_missing_timestamps(): Method to back-fill or forward-fill
                    the timestamps missing from the output of
                    make_timestamp_list().
        truncate_df(): Method to reduce the dataset to the timestamps
                    included in the output of make_timestamp_list().
        format_date_col(): Converts the entries of the "date" column
                    of the output dataframe to str.
    '''
    ###################
    ### Constructor ###
    ###################
    def __init__(self, 
                 api_key: str, 
                 api_secret: str,
                 ):
        self.api_key = api_key
        self.api_secret = api_secret

    ##################
    ### Fetch Data ###
    ##################
    def fetch_data(self,
                   start_date: str, 
                   end_date: str, 
                   ticker_list: list,
                   timeframe: str = "Hour",
                   extended_trading_hours: bool = False,
                   OHLCV_format: bool = True,
                   convert_dates_to_str: bool = True,
                  )->pd.DataFrame:
        '''
            Main function of the class. Downloads the stock historical
            data from Alpaca, fills the missing timestamps from the
            desired timeframe, and returns a dataset with the format
            used for PortfolioOptEnv.
            
            :param start_date: str. Start date of dataset.
            :param end_date: str. End date of dataset (not included).
            :param ticker_list: list. Ticker symbols to download.
            :param timeframe: str in ADMISSIBLE_TIMEFRAME_LIST.
            :param extended_trading_hours: bool. If False (default), each day in the output
                    dataframe contains data on 8 hours (9:00 to 16:00).
                    If True, the output contains data on 16 hours (8:00 to 23:00).
            :param OHLCV_format: bool. If False, keeps the "trade_count" and "vwap"
                    columns downloaded by default. True by default.
            :param convert_dates_to_str: bool. Whether to convert entries of the "date" column
                    of output dataframe to string objects. True by default, and if False
                    the "date" entries are pd.Timestamp objects.
            
            :return df_out: pd.DataFrame of formatted dataset.
        '''
        # Convert dates to datetime objects
        start_date_ = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_ = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Raise ValueError if dates are incoherent
        if end_date_<= start_date_:
            raise ValueError("Given end_date precedes start_date.") 
        
        # Download data
        df_data = self.download_Alpaca_df(start_date = start_date_,
                                          end_date=end_date_, 
                                          ticker_list = ticker_list,
                                          timeframe = timeframe,)
        
        # Format columns of downloaded dataframe
        df_data = self.format_download_columns(df_data = df_data,
                                               OHLCV_format = OHLCV_format,)
            
        # Print ignored tickers if any:
        missing_tickers_list = list(set(ticker_list) - set(df_data.tic.unique()))
        if len(missing_tickers_list)>0:
            print("Due to unavailable data, the following tickers have been omitted:")
            print(missing_tickers_list)
                
        # Make target timestamp list
        target_timestamp_list = self.make_timestamp_list(start_date = start_date_,
                                                         end_date = end_date_,
                                                         timeframe= timeframe,
                                                         extended_trading_hours = extended_trading_hours,
                                                         pandas_timestamps = True,)
        
        # Fill missing timestamps from downloaded data
        df_data = self.fill_missing_timestamps(df_data = df_data,
                                               target_timestamp_list = target_timestamp_list,)
            
        # Drop rows with timestamps not included in target_timestamp_list
        df_out = self.truncate_df(df_data = df_data,
                                  target_timestamp_list = target_timestamp_list,)
        
        # Format date column of output dataset
        if convert_dates_to_str:
            df_out = self.format_date_col(df_data = df_out,
                                          timeframe = timeframe)
        
        return df_out
    
    #############################
    ### Check timeframe param ###
    #############################
    def check_timeframe(self,
                        timeframe: str):
        '''
            Method to check if the timeframe keyword is included in ADMISSIBLE_TIMEFRAME_LIST,
            i.e. one of the keywords:
                           ["Day", "Hour",
                             "1min", "2min", "3min", "4min", "5min", "6min",
                             "10min", "12min", "15min", "20min", "30min"]. 
            Returns True if it is the case, and raises a ValueError exception otherwise.
            
            :param timeframe: str. Keyword controlling the size and contents
                        of the output dataset.
        '''
        if timeframe not in ADMISSIBLE_TIMEFRAME_LIST:
            raise ValueError(f"The timeframe parameter must be one of the following keywords:\n"\
                             f"{ADMISSIBLE_TIMEFRAME_LIST}")
        else:
            return True
        
    
    #####################
    ### Download Data ###
    #####################
    def download_Alpaca_df(self, 
                           start_date: datetime, 
                           end_date: datetime, 
                           ticker_list: list,
                           timeframe: str = "Hour",
                          )->pd.DataFrame:
        
        '''
            Method to download raw dataframe using Alpaca's API.

            :param start_date: datetime object.
            :param end_date: datetime object.
            :param ticker_list: list. Ticker symbols to include in dataset.
            :param timframe: str for trading period length. Should be
                                "Day", "Hour", "30min", or "15min".

            :return df_data: pd.DataFrame. Raw dataset obtained from Alpaca's
                            StockHistoricalDataClient object.

        '''
        if self.check_timeframe(timeframe):
            
            # Init. REST object
            stock_REST = StockHistoricalDataClient(api_key = self.api_key,
                                                   secret_key= self.api_secret)
            # Get appropriate timeframe object
            # See: https://github.com/alpacahq/alpaca-py/blob/master/alpaca/data/timeframe.py
            if timeframe == "Day":
                timeframe_obj = TimeFrame.Day
            elif timeframe == "Hour":
                timeframe_obj = TimeFrame.Hour
            # All remaining values in ADMISSIBLE_TIMEFRAME_LIST 
            else:
                timeframe_obj = TimeFrame.Minute

            # Init. request params
            request_params = StockBarsRequest(symbol_or_symbols = ticker_list,
                                              start = start_date,
                                              end = end_date,
                                              timeframe = timeframe_obj,)

            # Download data
            df_data = stock_REST.get_stock_bars(request_params).df

            return df_data
    
    ###############################
    ### Format Download Columns ###
    ###############################
    def format_download_columns(self, 
                                df_data: pd.DataFrame,
                                OHLCV_format: bool = True,
                               )->pd.DataFrame:
        '''
            Helper function to format the columns of dataframe downloaded from Alpaca.
            The output columns are in the following order:
                ["date", "open", "high", "low", "close", "volume", "tic"]
            where "tic" stands for the ticker symbol.


            :param df_data: Dataframe obtained from Alpaca.
            :param OHLCV_format: bool. Whether or not to drop the "trade_count" and "vwap" columns
                        downloaded by Alpaca.

            :return df_X: Formatted pd.DataFrame object.

        '''
        # Init.
        if OHLCV_format:
            drop_col_list =  ["trade_count", "vwap"]
            output_columns = ["date", "open", "high", "low", "close", "volume", "tic"]
            df_X = df_data.copy().drop(columns = drop_col_list)
        else:
            output_columns = ["date", "open", "high", "low", "close", "volume", "trade_count", "vwap", "tic"]
            df_X = df_data.copy()

        # Sort by date and ticker
        df_X = df_X.sort_values(by = ["timestamp", "symbol"])

        # Reset index and rename "timestamp" and "symbol" columns
        df_X = df_X.reset_index()
        df_X = df_X.rename(columns={"timestamp":"date", "symbol":"tic"})

        # Reorder output columns
        df_X = df_X[output_columns]

        return df_X
    
    ###########################
    ### Make Timestamp List ###
    ###########################
    def make_timestamp_list(self, 
                            start_date: datetime, 
                            end_date: datetime,
                            timeframe: str="Hour",
                            extended_trading_hours: bool =False,
                            pandas_timestamps: bool = False,
                           )->list:
        '''
            Method to build the list of timestamps to include
            in the output dataset.
            
            :param start_date: datetime object.
            :param end_date: datetime object.
            :param timeframe: str keyword from ["Day", "Hour", "1min", "2min", "3min", "4min",
                                "5min", "6min","10min", "12min", "15min", "20min", "30min"]
            :param extended_trading_hours: If False (default), each day in the output
                    dataframe contains data on 8 hours (9:00 to 16:00).
                    If True, the output contains data on 16 hours (8:00 to 23:00).
            :param pandas_timestamps: bool. If False (default), the output list consists of 
                    datetime instances. If True, these are converted to pd.Timestamp objects.
                            
            :return timestamp_list: List of timestamps to include in the output 
                    dataframe of fetch_data().
        '''
        # Raise ValueError if dates are incoherent
        if end_date<= start_date:
            raise ValueError("Given end_date precedes start_date.") 
            
        if self.check_timeframe(timeframe):
            # Get day list
            day_list = get_market_calendar_day_list(start_date=start_date,
                                                   end_date = end_date,
                                                   calendar_name = "NASDAQ")
            
            # timeframe = "Day" case
            if timeframe == "Day":
                timestamp_list = day_list
                #return day_list
            else:
                # Initializations
                timestamp_list = []
                ## 8-hour day format: 09:00 to 16:00
                if not extended_trading_hours:
                    start_hour = timedelta(hours=9)
                    n_hours = 8
                ## 16-hour day format: 08:00 to 23:00
                else:
                    start_hour = timedelta(hours=8)
                    n_hours = 16

                # timeframe="Hour" case
                if timeframe=="Hour":
                    for day in day_list:
                        for i in range(n_hours):
                            timestamp_list.append(day+start_hour+timedelta(hours=i))
                            
                # Case of timeframe in minutes
                else:
                    # Convert timeframe to int
                    timeframe_int_min = int(timeframe[:-3])
                    # Set timedelta in minutes
                    delta_minutes = timedelta(minutes = timeframe_int_min)
                    # Get no. of intra-hour timestamps
                    n_intra_hour = int(60/timeframe_int_min)
                    '''
                    # timeframe=="30min" case
                    if timeframe=="30min":
                        delta_minutes = timedelta(minutes=30)
                        n_intra_hour = 2
                        pass
                    # timeframe=="15min" case
                    elif timeframe=="15min":
                        delta_minutes = timedelta(minutes=15)
                        n_intra_hour = 4
                    
                    '''
                    # Timestamp loop
                    for day in day_list:
                        for i in range(n_hours-1):
                            for j in range(n_intra_hour):
                                timestamp_list.append(day+start_hour+timedelta(hours=i)+j*delta_minutes)
                        timestamp_list.append(day+start_hour+timedelta(hours=(n_hours-1)))

            # Return output 
            if not pandas_timestamps:
                return timestamp_list
            else:
                timestamp_list_ = [pd.to_datetime(x, utc=True) for x in timestamp_list]
                return timestamp_list_
            

    ###############################
    ### Fill Missing Timestamps ###
    ###############################
    def fill_missing_timestamps(self, 
                                df_data: pd.DataFrame,
                                target_timestamp_list: list,
                               )->pd.DataFrame:
    
        '''
            Method to add the dates in target_timestamp_list to df_data in
            case they are missing. Uses Pandas pd.DataFrame.fillna() with
            both "bfill" and "ffill".
            Typically, df_data is the output of format_download_columns()
            and target_timestamp_list is the output of make_timestamp_list().
            
            :param df_data: pd.DataFrame of downloaded data with formatted
                    columns.
            :param target_timestamp_list: List of timestamps that must be
                    included in the output dataframe.
            
            :return df_out: pd.DataFrame with timestamps originally downloaded
                    plus those in target_timestamp_list.

        '''
        if "date" not in list(df_data.columns):
            raise ValueError("No \"date\" column found in parameter df_data.")
            
        if not all(isinstance(x,pd.Timestamp) for x in target_timestamp_list):
            raise ValueError("All entries of target_timestamp_list must be instances "\
                             "of pd.Timestamp.")
        
        # Initializations
        ticker_list = list(df_data.tic.unique())
        df_out = pd.DataFrame()

        for tic in ticker_list:
            # Fix ticker
            df_temp = df_data[df_data.tic==tic]
            # Get missing timestamps
            missing_timestamps_list = list(set(target_timestamp_list)-set(df_temp.date.unique()))
            # Add missing timestamp rows
            df_temp = pd.concat([df_temp, pd.DataFrame({"date":missing_timestamps_list})])
            # Set "tic" column to ticker symbol
            df_temp["tic"] = tic
            # Sort by date
            df_temp = df_temp.sort_values(by="date")
            # Apply forward fill
            df_temp = df_temp.fillna(method = "ffill")
            # Apply backward fill (for missing values at the beginning)
            df_temp = df_temp.fillna(method = "bfill")
            # Append to output
            df_out = pd.concat([df_out, df_temp])

        # Sort and reset index
        df_out = df_out.sort_values(by=["date","tic"])
        df_out = df_out.reset_index(drop = True)

        return df_out
    
    ##########################
    ### Truncate DataFrame ###
    ##########################
    def truncate_df(self, 
                    df_data: pd.DataFrame,
                    target_timestamp_list: list,
                   )->pd.DataFrame:
        '''
            Method to reduce the downloaded data to the timestamps specified
            in target_timestamp_list.
            Typically, df_data is the output of fill_missing_timestamps()
            and target_timestamp_list is the output of make_timestamp_list().
            
            :param df_data: pd.DataFrame of downloaded data with formatted
                    columns and no missing timestamps.
            :param target_timestamp_list: List of timestamps that must be
                    included in the output dataframe.
            
            :return df_out: pd.DataFrame with whose date column entries are exactly
                    those in target_timestamp_list.
        '''
        if "date" not in list(df_data.columns):
            raise ValueError("No \"date\" column found in parameter df_data.")
            
        if not all(isinstance(x,pd.Timestamp) for x in target_timestamp_list):
            raise ValueError("All entries of target_timestamp_list must be instances "\
                             "of pd.Timestamp.")
            
        # Truncate and reset index
        df_out = pd.DataFrame()
        df_out = df_data[df_data["date"].isin(target_timestamp_list)]
        df_out = df_out.reset_index(drop=True)

        return df_out
    
    ##########################
    ### Format Date Column ###
    ##########################
    def format_date_col(self,
                        df_data: pd.DataFrame,
                        timeframe: str,
                       )->pd.DataFrame:
        '''
            Method to convert the entries of the "date" column
            in df_data to strings with the "%Y-%m-%d" or
            the "%Y-%m-%d %H:%M" format.
            Typically, df_data is the output of truncate_df().
            
            :param df_data: pd.DataFrame of processed dataset.
            :param timeframe: str. Keyword in ADMISSIBLE_TIMEFRAME_LIST.
            
            :return df_data:
        '''
        if "date" not in list(df_data.columns):
            raise ValueError("No \"date\" column found in parameter df_data.")
        
        if not all(isinstance(x,datetime) for x in list(df_data.date.unique())):
            raise ValueError("All entries of the \"date\" column of df_data must be instances "\
                             "of datetime.datetime or pd.Timestamp.")
        
        if self.check_timeframe(timeframe):
            if timeframe == "Day":
                date_str_format = "%Y-%m-%d"
            else:
                date_str_format = "%Y-%m-%d %H:%M"

            df_data["date"] = df_data.date.apply(lambda x: x.strftime(date_str_format))

            return df_data





'''
    ##################################
    #####    YAHOO DOWNLOADER    #####
    ##################################
'''

###################################
##### YahooDownloader #####
###################################
## AJ Zerouali, 22/11/21
## Modified for uniformity with AlpacaDownloader
## Approach borrowed from finrl.meta.preprocessor.yahoodownloader
class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from neofinrl_config.py)
        end_date : str
            end date of the data (modified from neofinrl_config.py)
        ticker_list : list
            a list of stock tickers (modified from neofinrl_config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """
        
    def fetch_data(self, 
                   start_date: str, 
                   end_date: str, 
                   ticker_list: list,
                   include_day_of_week: bool = False,
                   proxy=None) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        for tic in ticker_list:
            temp_df = yf.download(tic, 
                                  start = start_date, 
                                  end = end_date, 
                                  proxy=proxy,)
            temp_df["tic"] = tic
            data_df = pd.concat([data_df, temp_df]) # Updated 22/11/21: The index is the date
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            column_list = ["date", "open", "high", "low",
                           "close", "adjcp", "volume", "tic",]
            # convert the column names to standardized names
            data_df.columns = column_list
            # use adjusted close price instead of close price
            data_df["close"] = data_df["adjcp"]
            # drop the adjusted close price column
            data_df = data_df.drop(labels="adjcp", axis=1)
            
        except NotImplementedError:
            print("the features are not supported currently")
        
        # create day of the week column if required (monday = 0)
        if include_day_of_week:
            data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        return data_df

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df