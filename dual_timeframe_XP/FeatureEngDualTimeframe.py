########################################################
### Deep RL portfolio optimization - Data processing ###
########################################################
### 2023/04/25, A.J. Zerouali


from __future__ import annotations, print_function, division

import time
from datetime import datetime, timedelta
import pandas_market_calendars as pd_mkt_cals
from builtins import range

# np, pd plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# stockstats for FeatureEngineer
from stockstats import StockDataFrame as Sdf

# Downloaders and timeframe list
from drl_pfopt.common.data.downloaders import (AlpacaDownloader,
                                               YahooDownloader,
                                               ADMISSIBLE_TIMEFRAME_LIST,
                                              )
# Data helper functions
from drl_pfopt.common.data.data_utils import (get_timeframe_info,
                                              get_market_calendar_day_list,
                                              make_timestamp_list,
                                              make_intraday_timestamp_list,
                                             )


########################
### Global variables ###
########################
TECHNICAL_INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", 
                        "cci_30", "dx_30", "close_30_sma", "close_60_sma",
                        "turbulence", "vix"]

'''
###########################
##### FeatureEngineer #####
###########################
'''
### C Bill Cheng, 22/10/28
### AJ Zerouali, 23/04/23
### Helper class for cleaning downloaded financial data
### and adding technical indicators. Modified version of:
### finrl.meta.preprocessor.preprocessors.FeatureEngineer
class FeatureEngDualTF:
    """
    ########################
    ### TO DO (23/04/23) ###
    ########################
    - Test new implementation of the preprocess_data() method.
    - Test new implementation of close returns and cov
    - Test new implementation of turbulence idx computation
    - Test new implementation of the VIX (volatility idx) addition 
    
    
    Feature Engineer object for dual timeframes.
    Provides methods 

    Attributes
    ----------
        tech_ind_list : list
            List of technical indicator keys. If the list is empty then no technical 
            indicators are computed. Most technical indicators are computed using
            the stockstats package, and if an indicator isn't included in this packages
            then it is computed by a separate method.
        use_return_covs: bool
            Whether or not to compute close returns and covariances of close returns over
            a lookback period.
        n_lookback_prds: int
            Number of lookback periods used for the computation of close return time series
            and close returns covariance matrix.

    Methods
    -------
    preprocess_data()
        Main method to perform feature engineering. Cleans data and computes technical indicators.
        
    get_returns_covariances()
        Computes close returns and their covariances based on given data and no. of lookback periods.
    
    clean_data()
        Cleans dataset. Mainly removes NaN values and drops assets with stock split/inconsistent
        data from dataset.
        
    add_technical_indicators()
        Method to compute the technical indicators supported by the stockstats package. 
        GitHub repo: https://github.com/jealous/stockstats.
    
    add_vix()
        Adds the CBOE volatility index to the dataset by calling the YahooDownloader.
    
    add_turbulence()
        Adds the Kritzman-Li turbulence index for each of the assets in the dataset.
    
    calculate_turbulence()
        Computes the Kritzman-Li turbulence index for each of the assets in the dataset.

    """
    
    
    #################
    ## Constructor ##
    #################
    
    ### 2022/10/28, Cheng C.
    ### 2023/05/17, AJ Zerouali.
    def __init__(self, 
                 tech_indicator_list: list = TECHNICAL_INDICATORS, # 
                 use_return_covs: bool = False, # Use return covariances
                 data_source: str = "Alpaca", # Source of the dataframes provided eventually.
                ):
        '''
            See also preprocess_data() docstring for future modifications.
        '''
        ##### NEW (22/11/20) #####
        # Check data_source param
        if data_source not in ["Alpaca","Yahoo"]:
            raise ValueError("The data_source parameter must be one of the following keywords:\n"\
                             "            \"Alpaca\", or \"Yahoo\".")
        else:
            self.data_source = data_source
            if data_source == "Alpaca":
                self.api_key = None
                self.api_secret = None
                print("Chosen data source is Alapaca API. Use set_Alpaca_parameters()\n"\
                      "to provide the following parameters:\n"\
                      "api_key (str), api_secret (str).")
            
        
        # Attributes for technical indicators computations
        self.tech_indicator_list = tech_indicator_list
        self.use_tech_ind = (len(tech_indicator_list)>0) # Use technical indicators Boolean. (22/10/30, AJ Zerouali)
        self.use_vix = ("vix" in tech_indicator_list) # Use volatility index Boolean. 
        self.use_turbulence = ("turbulence" in tech_indicator_list) # Use "turbulence" Boolean.
        
        # Attributes for close returns and return covariances computations.
        self.use_return_covs = use_return_covs
        #self.user_defined_feature = user_defined_feature # For future version
        
        
    ##################
    ## Set API Keys ##
    ##################
    ## 23/04/10, AJ Zerouali 
    def set_Alpaca_parameters(self,
                              api_key: str, 
                              api_secret: str,
                             ):
        '''
            Method to assign the API key and secret code
            for Alpaca account. This information is relevant 
            only if preprocess_data() has to download
            some indicator using Alpaca (e.g. to download
            the VXX indicator when use_vix = True)
        '''
        
        self.api_key = api_key
        self.api_secret = api_secret
        

    ########################
    ## Data preprocessing ##
    ########################
    ## 22/10/23, AJ Zerouali
    
    def preprocess_data(self, 
                        df_X: pd.DataFrame,
                        trade_timeframe: str = "Data_timeframe",
                        trade_ext_hours: bool = False,
                        N_lookback_trade_prds: int = 1,
                       ):
        """
            Main feature engineering function of the class. Follows these steps:
            - Clean data from NaNs and incomplete ticker data.
            - Get the data timeframe info dictionary.
            - Drop the last daily timestamp from the input dataframe ("16:00" or "23:00").
            - Check the compatibility of the trading and data timeframes.
            - Build the trading timeframe info dictionary.
            - If use_return_covs = True, computes close prices returns (time series) over 
              previous N_lookback (= n_intratrading_timestamps*N_lookback_trade_prds) data timestamps, 
              as well as corresponding covariance matrix at each trading timestamp.
            - If use_tech_ind = True, adds specified list of technical indicators to output df.
            - If use_vix = True, adds volatility index to output df.
            - If use_turbulence = True, adds Kritzman-Li turbulence index to output df.
            
            Comments:
            - The present implementation is taylored to dataframes produced by the Alpaca downloader
              in drl_pfopt.common.data.downloaders. It is therefore highly recommended that the df_X 
              parameter be obtained using the downloaders in the drl_pfopt.common.data.downloaders
              submodule, especially for intraday price data. This class calls the get_timeframe_info()
              function in drl_pfopt.common.data.data_utils, which is very particular regarding the
              intraday timestamps and the date format.
            - The output of this method removes the last daily timestamp ("16:00" or "23:00") from
              df_X. This is to ensure that the number of intra-trading data timestamps remains constant
              for every trading timestamp (after the very first one). The rationale behind this 
              omission is that we rebalance the portfolio weights at the BEGINNING of each trading period,
              and for the very first trading period, we use equal weights.
              
            
            
            ## Note (22/11/23): If any custom technical indicator **custom_tech_ind** is added to TECHNICAL_INDICATORS, 
            ##     and if it is not supported by stockstats, then the correct way of modifying this class is as follows: 
            ##     1) Add a Boolean use_**custom_tech_ind** = ("**custom_tech_ind**" in tech_indicator_list) in
            ##        the constructor of FeatureEngineer.
            ##     2) Implement calculate_**custom_tech_ind**() and add_**custom_tech_ind**() to compute **custom_tech_ind**
            ##        and add it to the output pd.DataFrame. Use NumPy as much as possible for optimal execution times.
            ##     3) Modify preprocess_data() by calling add_**custom_tech_ind**() if use_**custom_tech_ind** = True.
            ##     4) If the code is modified to output additional dataframes or arrays, add the correspoding key(s) 
            ##        and container(s) to the processed_data_dict output.
        
            :param df_X: pd.DataFrame of input data
            
            :return processed_data_dict: Dictionary of processed data. The keys access the following:
                    - "df": pd.DatFrame of preprocessed data plus chosen technical indicators
                    - "np_close_returns": np.ndarray of returns over lookback period
                    - "np_returns_covs": np.ndarray of return covariances over lookback period
                    - "data_timeframe_info": dict of attributes associated to data timestamps
                    - "trade_timeframe_info": dict of attributes of trading timestamps
        """
        
        
        # Initializations
        processed_data_dict = {}
        feature_shape_dict = {} # To keep track of feature shapes at each trading timestamp
        df = df_X.copy()
        
        # Clean data
        '''
            NOTE (23/04/18 AJ Zerouali)
            This next instruction is mostly relevant if df_X
            was obtained using something other than the 
            Alpaca downloader.
        '''
        df = self.clean_data(df)
        
        # Process dual timeframes and get formatted dataframe
        df, data_timeframe_info_dict, trade_timeframe_info_dict = \
            self.process_dual_timeframes(df_X = df,
                                         trade_timeframe = trade_timeframe,
                                         trade_ext_hours = trade_ext_hours,
                                         N_lookback_trade_prds = N_lookback_trade_prds,
                                        )
        '''
            NOTE (23/04/22)
            
            When adding a riskless asset, the instruction should be here.
        '''
        # Add feature shapes to feature_dictionary
        n_assets = len(df.tic.unique())
        n_intratrading_timestamps = trade_timeframe_info_dict["n_intratrading_timestamps"]
        N_lookback = n_intratrading_timestamps*trade_timeframe_info_dict["N_lookback_trade_prds"]
        feature_list = list(df.columns).copy()
        feature_list.remove("date")
        feature_list.remove("tic")
        for feature in feature_list:
            feature_shape_dict[feature] = (n_intratrading_timestamps, n_assets)
        # Get no. of trading periods
        '''
            NOTE (23/04/18, AJ Zerouali)
            Needs modification:
            Now there should be 2 different numbers of periods:
            - len(trade_timestamp_list), number of trading periods
            - len(data_timestamp_list), number of data timestamps
        '''
        #N_periods = len(df.date.unique()) # Do we need this enywhere
        
        # Get returns and their covariances array
        '''
            NOTE (23/04/18, AJ Zerouali)
            When adding the riskless asset, the covariances with the bond/cash acct. should be 0.
            Will be easier with an input parameter for the riskless asset.
        '''
        if self.use_return_covs:
            # Check there are enough datapoints to compute statistics
            if N_lookback < 50:
                raise ValueError(f"ERROR: Not enough data to compute returns."\
                                 f" Total number of lookback timestamps must be larger than 50.\n"\
                                 f" Here: N_lookback_trade_prds = {N_lookback_trade_prds}, "\
                                 f" n_intratrading_timestamps = {n_intratrading_timestamps}, and:\n"\
                                 f" N_lookback = {N_lookback}")
            else:
                np_close_returns, np_returns_covs = compute_close_returns_covariances(df,trade_timeframe_info_dict)
                feature_shape_dict["close_returns"] = (N_lookback, n_assets)
                feature_shape_dict["returns_cov"] = (n_assets, n_assets)
                print("Successfully computed asset returns and their covariances")
        # Add empty arrays for gym environment
        else:
            np_close_returns = np.empty(shape=0)
            np_returns_covs = np.empty(shape=0)
            feature_shape_dict["close_returns"] = 0
            feature_shape_dict["returns_cov"] = 0

        # Add technical indicators using stockstats
        if self.use_tech_ind:
            df, sdf_technical_indicator_list = self.add_technical_indicators(df)
            for feature in sdf_technical_indicator_list:
                feature_shape_dict[feature] = (N_lookback, n_assets)
            print("Successfully added technical indicators")

        # Add vix
        if self.use_vix:
            df = self.add_vix(df, data_timeframe_info_dict)
            feature_shape_dict["vix"] = (n_intratrading_timestamps, n_assets)
            print("Successfully added volatility index (vix)")

        # Add turbulence index
        if self.use_turbulence:
            df = self.add_turbulence(df, trade_timeframe_info_dict)
            feature_shape_dict["turbulence"] = (n_intratrading_timestamps, n_assets)
            print("Successfully added turbulence index")
        
        # Fill the missing values at the beginning and the end
        df = df.fillna(method="ffill").fillna(method="bfill")
        
        # Re-order columns
        column_list = list(df.columns).copy()
        column_list.remove("tic")
        column_list.append("tic")
        df = df[column_list]
        
        # Combine output dictionary (this is the 1st input param. of the gym environment)
        df = df.reset_index(drop = True)
        processed_data_dict["df"] = df
        processed_data_dict["np_close_returns"] = np_close_returns
        processed_data_dict["np_returns_covs"] = np_returns_covs 
        processed_data_dict["data_timeframe_info"] = data_timeframe_info_dict
        processed_data_dict["trade_timeframe_info"] = trade_timeframe_info_dict
        processed_data_dict["feature_shape_dict"] = feature_shape_dict
        
        return processed_data_dict
    
    '''
    #############################
    ## Process dual timeframes ##
    #############################
    ## 23/04/20 version
    '''
    def process_dual_timeframes(self,
                                df_X: pd.DataFrame,
                                trade_timeframe: str = "Data_timeframe",
                                trade_ext_hours: bool = False,
                                N_lookback_trade_prds: int = 1,
                               ):
        '''
            Wrapper for get_trading_timeframe_info().
            See docstring of this function for details.
            
            :param df_X:
            :param trade_timeframe:
            :param trade_ext_hours:
            :param N_lookback_trade_prds:
                        
            :return df:
            :return data_timeframe_info_dict:
            :return trade_timeframe_info_dict:
        '''
        timeframe_dict = get_trading_timeframe_info(df_X = df_X,
                                                    trade_timeframe = trade_timeframe,
                                                    trade_ext_hours = trade_ext_hours,
                                                    N_lookback_trade_prds= N_lookback_trade_prds,
                                                   )
        return timeframe_dict['df'], timeframe_dict['data_timeframe_info_dict'], timeframe_dict['trade_timeframe_info_dict']
        
    '''
    #################################
    ## Get returns and covariances ##
    #################################
    ## 22/10/23 version
    '''
    def get_returns_covariances(self, df: pd.DataFrame):
        '''
            ###################################
            ### DEPRECATED AS OF 2023/04/22 ###
            ###################################
            Compute asset returns on close prices over specified no. of lookback periods,
            as well as return covariances over same time frame.
            
            :param df: pd.DataFrame of cleaned financial data.
            
            :return np_close_returns: np.ndarray of shape ((N_periods-n_lookback), n_lookback, n_assets), where:
                        * N_periods is the no. of (unique) trading dates in df,
                        * n_lookback is the no. of lookback periods used for computation,
                        * n_assets is the no. of (unique) tickers in df,
                        * np_close_returns[i,:,:] is the returns matrix on day i, i in [n_lookback, N_periods]
                        
            :return np_returns_covs: np.ndarray of shape ((N_periods-n_lookback), n_assets, n_assets), where
                        np_returns_covs[i,:,:] is the covariance matrix of np_close_returns[i,:,:].
        
        
        # Initializations
        N_periods = len(df.date.unique())
        n_assets = len(df.tic.unique())
        n_lookback = self.n_lookback_prds
        
        np_close_returns = np.empty(shape = ((N_periods-n_lookback), n_lookback, n_assets), 
                                   dtype = np.float64)
        np_returns_covs = np.empty(shape = ((N_periods-n_lookback), n_assets, n_assets), 
                                   dtype = np.float64)
        df_close = df.pivot_table(index = 'date', columns = 'tic', values = 'close')
        
        # Main loop - i is the dataframe index
        for i in range(n_lookback, N_periods):
            
            # Index for np arrays
            j = i-n_lookback
            
            # Temp array for computations
            df_temp = df_close.iloc[i-n_lookback:i+1]
            X = df_temp.to_numpy()
            
            # Returns
            np_close_returns[j,:,:] = (X[1:,:]-X[0:-1,:])/X[0:-1,:]
            
            # Covariance. IMPORTANT: rowvar = False because columns of X are the variables (tickers)
            np_returns_covs[j,:,:] = np.cov(m=np_close_returns[j,:,:], rowvar = False)
        
        # Output
        return np_close_returns, np_returns_covs
        '''
        pass


    ##################
    ## Data cleaner ##
    ##################
    def clean_data(self, data):
        """
            Clean the raw dataframe and process missing values.
            Stocks could be delisted or not incorporated at the time step.
            Used in first step of preprocess_data() after copying dataframe.
            
            :param data: (df) pandas dataframe of raw data.
            
            :return: (df) pandas dataframe of cleaned data
        """
        df = data.copy()
        df = df.sort_values(["date", "tic"], ignore_index=True)
        df.index = df.date.factorize()[0]
        merged_closes = df.pivot_table(index="date", columns="tic", values="close")
        merged_closes = merged_closes.dropna(axis=1)
        tics = merged_closes.columns
        df = df[df.tic.isin(tics)]
        return df

    ##########################
    ## Technical indicators ##
    ##########################
    ### 2022/10/30, AJ Zerouali: Updated technical indicator list for main loop (sdf_technical_indicator_list).
    def add_technical_indicators(self, data):
        """
            Computes technical indicators supported by stockstats, then adds
            results to the dataframe. Called by preprocess_data().
            
            :param data: (df) pandas dataframe
            
            :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()
        
        # List of stockstats technical indicators (22/10/30, AJ Zerouali)
        sdf_technical_indicator_list = self.tech_indicator_list.copy()
        # Remove all technical indicators not computed by stockstats from list
        ### NOTE (22/10/30): If custom technical indicators are added 
        if self.use_vix:
            sdf_technical_indicator_list.remove("vix")
        if self.use_turbulence:
            sdf_technical_indicator_list.remove("turbulence")

        for indicator in sdf_technical_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["date"] = df[df.tic == unique_ticker[i]][
                        "date"
                    ].to_list()
                    indicator_df = pd.concat([indicator_df, temp_indicator], ignore_index = True) #22/11/20
                except Exception as e:
                    print(e)
            df = df.merge(indicator_df[["tic", "date", indicator]], 
                          on=["tic", "date"], 
                          how="left")
        df = df.sort_values(by=["date", "tic"])
        return df, sdf_technical_indicator_list

    '''
    ### Comment (22/10/22, AJ Zerouali): Implemented in some future version.
    def add_user_defined_feature(self, data):
        """
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df["daily_return"] = df.close.pct_change(1)
        return df
    '''
    
    ##########################
    ## Add Volatility Index ##
    ##########################
    ## AJ Zerouali, 23/04/22
    # Updated to download VXX by Alpaca if that's the chosen data source
    # For the choice of VXX instead of VIX here, see discussion at:
    # https://forum.alpaca.markets/t/how-to-get-volatility-data-vix-vxtlt-etc/2907
    def add_vix(self, 
                df_X: pd.DataFrame, 
                data_timeframe_info:dict):
        """
            Add the CBOE volatility index using a Yahoo object,
            or the VXX index using an Alpaca downloader object.
            Called by preprocess_data().
            
            NOTE (AJ Zerouali, 23/04/26):
            - I am not satisfied with how this indicator is added to
              the data. 
            - This indicator should perhaps be added as a "virtual"
              asset to the portfolio, and instead of storing a matrix 
              identical columns at a given timestamp, one should 
              with perhaps compute the covariance between this indicator
              and the portfolio assets.
            
            :param data: (df) pandas dataframe
            :return: (df) pandas dataframe
        """
        # Initializations
        df = df_X.copy()
        data_timeframe = data_timeframe_info["timeframe"]
        data_ext_hours = data_timeframe_info["extended_trading_hours"]
        start_date = data_timeframe_info["start_date"].strftime("%Y-%m-%d")
        end_date = data_timeframe_info["end_date"].strftime("%Y-%m-%d")
        
        # Download VIX using YahooDownloader
        if self.data_source == "Yahoo":
            if data_timeframe!="Day":
                # Using data_timeframe_info["timeframe"] in the f-string returns an error from the interpreter
                raise ValueError(f"Data timeframe is incompatible with Yahoo downloader.\n"\
                                 f"Here: data_timeframe = {data_timeframe},"\
                                 f"instead of \"Day\"."
                                )
            else:
                downloader = YahooDownloader()
                df_vix = downloader.fetch_data(start_date=start_date,
                                             end_date=end_date,
                                             ticker_list=["^VIX"],)
            
        # Downlaod VXX using AlpacaDownloader
        elif self.data_source == "Alpaca":
            print("Downloading the VXX proxy for ^VIX instead of the latter.\n"\
                  "(Since ^VIX is not a US equity, its data cannot be obtained from Alpaca.)")
            
            downloader = AlpacaDownloader(self.api_key, self.api_secret)
            
            df_vix_ = downloader.fetch_data(start_date=start_date,
                                           end_date=end_date,
                                           ticker_list = ["VXX"],
                                           timeframe = data_timeframe,
                                           extended_trading_hours = data_ext_hours,
                                           OHLCV_format = True,
                                           convert_dates_to_str = True,)
            
            # Remove last daily timestamps from df_vix_ (16:00 or 23:00) if intraday data
            if data_timeframe =="Day":
                df_vix = df_vix_
            else:
                data_timestamp_list = list(df_X.date.unique())
                df_vix = df_vix_[df_vix_.date.isin(data_timestamp_list)]
        
        # Process output dataframe
        tic = "vix"
        vix = df_vix[["date", "close"]]
        vix.columns = ["date", tic]

        df = df.merge(vix, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        
        return df

    ##################################
    ## Add (Kritzman-Li) Turbulence ##
    ##################################
    ## AJ Zerouali, 23/04/22
    def add_turbulence(self, 
                       df_X: pd.DataFrame,
                       trade_timeframe_info: dict,
                      ):
        """
            Add the Kritman-Li turbulence index from the output of calculate_turbulence().
            Called by preprocess_data().
            
            :param df_X: pd.DataFrame of clean data.
            :param trade_timeframe_info: dict of 
            
            :return df: Input dataframe augmented with turbulence values for each asset.
        
        """
        
        # Initialization
        df = df_X.copy()
        # Call helper function
        df_turbulence_idx = compute_turbulence_idx(df, trade_timeframe_info)
        df = df.merge(df_turbulence_idx, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    ######################################
    ## Compute (Kritzman-Li) Turbulence ##
    ######################################
    ## AJ Zerouali, 23/04/22: See compute_turbulence_idx() helper function.
    ## 22/10/28, Cheng C: numpy version of Turbulence calculation.
    ## Tested this function in notebook test_np_turbulence_function.ipynb 
    ## in paperspace, get same result as FinRL's original computations.
    ## Original FinRL submodule: finrl.meta.preprocessors.preprocessors

    def calculate_turbulence(self, data):
        
        '''
            ###################################
            ### DEPRECATED AS OF 2023/04/22 ###
            ###################################
            calculate user defined features
        
            :param data: Cleaned pandas dataframe
        
            :return df: pd.DataFrame with date and turb, len is number of unique days in df.

            Note (22/10/28, C. 'Bill' Cheng): Input is already free of missing so no need to do 
            .dropna(axis=1) as original turbulence function. Definition can be found at:
            https://www.top1000funds.com/wp-content/uploads/2010/11/FAJskulls.pdf
            https://www.tandfonline.com/doi/abs/10.2469/faj.v66.n5.3
            
        
        
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()
        N_periods = len(df.date.unique())
        n_lookback = self.n_lookback_prds #### Question: can this n_lookback for calculating turbulence the same as self.n_lookback_prds (60) ??
        
        turbulence_index = np.zeros(N_periods)
        
        count = 0
        returns = df_price_pivot.to_numpy()
        returns[0] = returns[1] ## do a backward fill for first row nan values
        for i in range(n_lookback,N_periods):
            current_returns = returns[i]
            hist_return = returns[i-n_lookback:i]  
            cov_temp = np.cov(m=hist_return, rowvar = False) ## compute sigma_t
            current_temp = current_returns - np.mean(hist_return,axis=0) ## compute yt - mu_t
            temp = np.dot(current_temp,np.dot(np.linalg.pinv(cov_temp),current_temp.T)) ## compute (yt - mu_t) sigma_t^-1 (yt - mu_t)T

            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_index[i] = temp
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_index[i] = 0
            else:
                turbulence_index[i] = 0
           
        try:
            turbulence_index = pd.DataFrame(
                {"date": df_price_pivot.index, "turbulence": turbulence_index}
            )
        except ValueError:
            raise Exception("Turbulence information could not be added.")
        return turbulence_index
        '''
        pass
    
'''
    ##########################
    #### HELPER FUNCTIONS ####
    ##########################
'''
'''
################################################
#### GET TRADING TIMEFRAME INFO (VERSION 3) ####
################################################
AJ Zerouali, 23/04/21
Quasi-final version. Seems to do what
it's supposed to.
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
###############################################
#### GET NUMBER OF INTRATRADING TIMESTAMPS ####
###############################################
AJ Zerouali, 23/04/21
'''
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

'''
###########################################################
#### COMPUTE CLOSE PRICE RETURNS AND THEIR COVARIANCES ####
###########################################################
AJ Zerouali, 23/04/23
'''
def compute_close_returns_covariances(df: pd.DataFrame,
                                      trade_timeframe_info: dict,
                                      percentage_returns: bool = False
                                     ):
    '''
        Compute asset returns on close prices and covariance matrices thereof.
        
        
        :param df: pd.DataFrame of cleaned financial data.
        :param trade_timeframe_info:
        :param percentage_returns:
        
        :return np_close_returns: np.ndarray of shape ((N_periods-n_lookback), n_lookback, n_assets), where:
                * N_periods is the no. of (unique) trading dates in df,
                * n_lookback is the no. of lookback periods used for computation,
                * n_assets is the no. of (unique) tickers in df,
                * np_close_returns[i,:,:] is the returns matrix on day i, i in [n_lookback, N_periods]
                
        :return np_returns_covs: np.ndarray of shape ((N_periods-n_lookback), n_assets, n_assets), where
                np_returns_covs[i,:,:] is the covariance matrix of np_close_returns[i,:,:].
    '''
    
    # Initializations
    ## Trade timestamp list
    trade_timestamp_list = trade_timeframe_info["trade_timestamp_list"]
    ## No. of trading periods
    N_trade_prds = len(trade_timestamp_list)
    ## No. of assets in dataframe
    n_assets = len(df.tic.unique())
    ## Length of the close return series
    ## Equals no. of lookback trading periods times no. of intra-trading timestamps
    n_clret_ts = trade_timeframe_info["N_lookback_trade_prds"]\
                *trade_timeframe_info["n_intratrading_timestamps"]
    
    ## Trading/data schedule
    trading_data_schedule = trade_timeframe_info["trading_data_schedule"]
    ## Percentage returns
    if percentage_returns:
        mult_factor = 100.0
    else:
        mult_factor = 1.0
    
    
    # Initialize output arrays
    np_close_returns = np.empty(shape = (N_trade_prds, n_clret_ts, n_assets), 
                                dtype = np.float64)
    np_returns_covs = np.empty(shape = (N_trade_prds, n_assets, n_assets), 
                               dtype = np.float64)
    
    # Close prices dataframe and array
    df_close = df.pivot_table(index = 'date', columns = 'tic', values = 'close')
    np_close = df_close.to_numpy()
    # Close returns dataframe
    df_clret = pd.DataFrame(index = list(df_close.index), columns = list(df_close.columns))
    # Return at timestamp t is r_t = (c_t-c_(t-1))/c_(t-1), omit first timestamp
    df_clret.iloc[1:]=((np_close[1:,:]-np_close[:-1,:])/np_close[:-1,:])*mult_factor
    # Backfill for first timestamp
    df_clret.iloc[0] = df_clret.iloc[1]
    
    
    # Main loop - i is the index of the trading timestamp in trade_timestamp_list
    for i in range(N_trade_prds):
        
        # Trading timestamp
        trade_timestamp_temp = trade_timestamp_list[i]
        
        # Data timestamp list
        data_timestamp_list_temp = trading_data_schedule[trade_timestamp_temp]
        
        # Store close returns time series for trading_timestamp_temp
        np_close_returns[i,:,:] = df_clret[df_clret.index.isin(data_timestamp_list_temp)].to_numpy()
        
        # Covariance. IMPORTANT: rowvar = False because columns of X are the variables (tickers)
        np_returns_covs[i,:,:] = np.cov(m=np_close_returns[i,:,:], rowvar = False)
        
    # Output
    return np_close_returns, np_returns_covs

'''
####################################################
#### COMPUTE THE (KRITZMAN-LI) TURBULENCE INDEX ####
####################################################
AJ Zerouali, 23/04/23
'''
def compute_turbulence_idx(df_X: pd.DataFrame,
                           trade_timeframe_info: dict,
                          ):
    
    '''
        Helper function to compute the Kritzman-Li turbulence index.
        TO DO:
        - Add comment on the 2 computations
        References:
        https://www.top1000funds.com/wp-content/uploads/2010/11/FAJskulls.pdf
        https://www.tandfonline.com/doi/abs/10.2469/faj.v66.n5.3
        
        :param df: pd.DataFrame
        
        
        :return df: pd.DataFrame with date and turb, len is number of unique days in df.
        
    '''
    # Initializations
    N_data_timestamps = len(df_X.date.unique())
    n_assets = len(df_X.tic.unique())
    n_intratrading_timestamps = trade_timeframe_info["n_intratrading_timestamps"]
    N_lookback_trade_prds = trade_timeframe_info["N_lookback_trade_prds"]
    N_lookback = n_intratrading_timestamps*N_lookback_trade_prds
    np_turbulence_idx = np.zeros(N_data_timestamps)
       
    # Close prices dataframe and array
    df_close = df_X.pivot_table(index = 'date', columns = 'tic', values = 'close')
    np_close = df_close.to_numpy()
        
    # Close returns array
    np_clret = np.zeros(shape=(N_data_timestamps, n_assets))
    np_clret[1:,:] = ((np_close[1:,:]-np_close[:-1,:])/np_close[:-1,:])*100.0
    np_clret[0,:] = np_clret[1,:]
    
    # Sample mean and sample covariance over entire dataset
    mu_S = np.mean(np_clret,axis=0)
    Sigma_S = np.cov(m = np_clret, rowvar = False)
    # Pseudo-inverse of sample covariance
    inv_Sigma_S = np.linalg.pinv(Sigma_S)
    
    # Fill turbulence index array
    if (N_lookback < 50):
        '''
            If the total no. lookback data timestamps is lower than 50,
            compute the turbulence index using the sample average/cov
            of the entire dataset
        '''
        
        for t in range(N_data_timestamps):
            turb_temp_t = np.dot((np_clret[t,:]- mu_S),\
                                 np.dot(inv_Sigma_S, (np_clret[t,:]- mu_S))
                                )
            # The turbulence index should be non-negative
            if turb_temp_t>0:
                np_turbulence_idx[t] = turb_temp_t
    else:
        '''
            If the total no. lookback data timestamps is greater than 50,
            compute the turbulence index using the sample average/cov
            over last N_lookback = n_intratrading_timestamps*N_lookback_trade_prds
            periods. Fill first N_lookback values with previous computation.
        '''
        for t in range(N_lookback):
            # Temp value
            turb_temp_t = np.dot((np_clret[t,:]- mu_S),\
                                 np.dot(inv_Sigma_S, (np_clret[t,:]- mu_S))
                                )
            # The turbulence index should be non-negative
            if turb_temp_t>0:
                np_turbulence_idx[t] = turb_temp_t
        
        for t in range(N_lookback,N_data_timestamps):
            # Compute sample average and covariance over last N_lookback returns
            mu_ = np.mean(np_clret[t - N_lookback:t, :], axis = 0)
            Sigma_ = np.cov(m = np_clret[t - N_lookback:t, :], rowvar= False)
            inv_Sigma_ = np.linalg.pinv(Sigma_)
            # Temp value
            turb_temp_t = np.dot((np_clret[t, :] - mu_),\
                                 np.dot(inv_Sigma_, (np_clret[t, :] - mu_))
                                )
            # The turbulence index should be non-negative
            if turb_temp_t>0:
                np_turbulence_idx[t] = turb_temp_t
    
    # Check output
    try:
        df_turbulence_idx = pd.DataFrame({"date": df_close.index, "turbulence": np_turbulence_idx})
    except ValueError:
        raise Exception("Turbulence information could not be added.")
    
    
    return df_turbulence_idx
