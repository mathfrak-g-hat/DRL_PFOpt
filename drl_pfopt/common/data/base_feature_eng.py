########################################################
### Deep RL portfolio optimization - Data processing ###
########################################################
### 2022/11/24, A.J. Zerouali


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

# Downloaders
from drl_pfopt.common.data.downloaders import AlpacaDownloader, YahooDownloader

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
### AJ Zerouali, 23/04/10
### Helper class for cleaning downloaded financial data
### and adding technical indicators. Modified version of:
### finrl.meta.preprocessor.preprocessors.FeatureEngineer
class FeatureEngineer:
    """Provides methods for preprocessing the stock price data

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
    
    ### 2022/10/28, Cheng C.: TECHNICAL_INDICATORS already has turbulence, why do we still need use_turbulence?
    ### 2022/10/30, AJ Zerouali: Updated Boolean attributes for technical indicators. See also add_technical_indicators().
    def __init__(self, 
                 tech_indicator_list: list = TECHNICAL_INDICATORS, # 
                 use_return_covs: bool = False, # Use return covariances
                 n_lookback_prds: int = 60, # Number of lookback periods for stat computations
                 data_source: str = "Alpaca", # Source of the dataframes provided eventually.
                ):
        '''
            ### To do (22/10/22):
            - Add a condition checking if a lookback period 
              is provided when use_return_covs = True.
            
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
                self.timeframe = None
                self.extended_trading_hours = None
                print("Chosen data source is Alapaca API. Use set_Alpaca_parameters()\n"\
                      "to provide the following parameters:\n"\
                      "api_key (str), api_secret (str), timeframe (str), and extended_trading_hours (bool)")
            
        
        # Attributes for technical indicators computations
        self.tech_indicator_list = tech_indicator_list
        self.use_tech_ind = (len(tech_indicator_list)>0) # Use technical indicators Boolean. (22/10/30, AJ Zerouali)
        self.use_vix = ("vix" in tech_indicator_list) # Use volatility index Boolean. 
        self.use_turbulence = ("turbulence" in tech_indicator_list) # Use "turbulence" Boolean.
        
        # Attributes for close returns and return covariances computations.
        self.use_return_covs = use_return_covs
        self.n_lookback_prds = n_lookback_prds
        #self.user_defined_feature = user_defined_feature # For future version
        
        
    ##################
    ## Set API Keys ##
    ##################
    ## 23/04/10, AJ Zerouali 
    def set_Alpaca_parameters(self,
                              api_key: str, 
                              api_secret: str,
                              timeframe: str,
                              extended_trading_hours: bool,
                             ):
        # Check timeframe param
        if timeframe not in ["Day", "Hour",
                             "1min", "2min", "3min", "4min", "5min", "6min",
                             "10min", "12min", "15min", "20min", "30min"]:
            
            raise ValueError("The timeframe parameter must be one of the following keywords:\n"\
                             "            \"Day\", \"Hour\", \"1min\", \"2min\", \"3min\", \"4min\", \"5min\", \n"\
                             "            \"6min\", \"10min\", \"12min\", \"15min\", \"20min\", \"30min\"")
        else:
            self.timeframe = timeframe
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.extended_trading_hours = extended_trading_hours
        

    ########################
    ## Data preprocessing ##
    ########################
    ## 22/10/23, AJ Zerouali
    
    def preprocess_data(self, df_X: pd.DataFrame):
        """
            Main feature engineering function of the class. Follows these steps:
            - Clean data from NaNs and incomplete ticker data.
            - If use_return_covs = True, computes returns over close prices for 
              previous n_lookback_prds, as well as corresponding daily covariance
              matrix. In this case, the method drops data of the first n_lookback_prds dates 
              from the df output dataframe.
            - If use_tech_ind = True, adds specified list of technical indicators to output df.
            - If use_vix = True, adds volatility index to output df.
            - If use_turbulence = True, adds Kritzman-Li turbulence index to output df.
            
            
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
        """
        
        
        # Initializations
        processed_data_dict = {}
        df = df_X.copy()
        
        # Clean data
        df = self.clean_data(df)
        
        # Get no. of trading periods
        N_periods = len(df.date.unique())
        
        # Get returns and their covariances array
        if self.use_return_covs:
            # Coherence check
            if self.n_lookback_prds >= N_periods:
                raise ValueError(f"ERROR: Not enough data to compute returns."\
                                 f"  Number of trading dates in input data must be larger than"\
                                 f"  number of lookback periods. Here:"\
                                 f"   n_lookback_prds = {self.n_lookback_prds} >= N_periods = {N_periods})")
            else: 
                np_close_returns, np_returns_covs = self.get_returns_covariances(df)
                print("Successfully computed asset returns and their covariances")

        # add technical indicators using stockstats
        if self.use_tech_ind:
            df = self.add_technical_indicators(df)
            print("Successfully added technical indicators")

        # add vix for multiple stock
        if self.use_vix:
            df = self.add_vix(df)
            print("Successfully added volatility index (vix)")

        # add turbulence index for multiple stock
        if self.use_turbulence:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")
        
        # fill the missing values at the beginning and the end
        df = df.fillna(method="ffill").fillna(method="bfill")
        
        # Re-order columns
        column_list = list(df.columns).copy()
        column_list.remove("tic")
        column_list.append("tic")
        df = df[column_list]
        
        # Return output
        ## Case of dataframe only
        if not self.use_return_covs:
            # Prepare output
            df = df.reset_index(drop = True)
            processed_data_dict["df"] = df
            
        ## Dataframe, close returns and return covariances
        else:            
            # Number of indexes to drop from df
            ## = n_lookback_prds*n_assets
            m = self.n_lookback_prds*len(df.tic.unique())
            
            # Drop first n_lookback_prds
            df = df.sort_values(by = ['date','tic']).reset_index(drop=True)
            df_ = df.iloc[m:]
            df_ = df_.reset_index(drop = True)
            
            # Prepare output
            processed_data_dict["df"] = df_
            processed_data_dict["np_close_returns"] = np_close_returns
            processed_data_dict["np_returns_covs"] = np_returns_covs
        
        return processed_data_dict
        
    '''
    #################################
    ## Get returns and covariances ##
    #################################
    ## 22/10/23 version
    '''
    def get_returns_covariances(self, df: pd.DataFrame):
        '''
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
        '''
        
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
        return df

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
    ## AJ Zerouali, 23/04/10
    # Updated to download VXX by Alpaca if that's the chosen data source
    # For the choice of VXX instead of VIX here, see discussion at:
    # https://forum.alpaca.markets/t/how-to-get-volatility-data-vix-vxtlt-etc/2907
    def add_vix(self, data):
        """
            Add the CBOE volatility index using a YahooDownloader object.
            Called by preprocess_data().
            
            :param data: (df) pandas dataframe
            :return: (df) pandas dataframe
        """
        # Initializations
        df = data.copy()
        first_date = df.date.min()
        last_date = df.date.max()
        
        # Check type and length of entries of "date" col
        if all(isinstance(x,str) for x in df.date.unique()):
            all_dates_Ymd = all(len(x)==10 for x in list(df.date.unique()))
            all_dates_YmdHM = all(len(x)==16 for x in list(df.date.unique()))
            
            if (not all_dates_Ymd) and (not all_dates_YmdHM):
                raise ValueError("All entries of the \"date\" column of the "\
                                 "data parameter must have the same format.\n"\
                                 "(\"%Y-%m-%d\" or \"%Y-%m-%d %H:%M\".)")
        else:
            raise ValueError("All entries of the \"date\" column of the "\
                             "data parameter must be str instances")
        
        # Get start and end dates of the dataset (should be str)
        ## Date format is "%Y-%m-%d"
        if all_dates_Ymd:
            start_date = first_date
            end_date = (datetime.strptime(last_date,"%Y-%m-%d")+timedelta(days=1)).strftime("%Y-%m-%d")
        ## Date format is "%Y-%m-%d %H:%M"
        elif all_dates_YmdHM:
            start_date = first_date[:10]
            end_date = (datetime.strptime(last_date[:10],"%Y-%m-%d")+timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Download VIX using YahooDownloader
        if self.data_source == "Yahoo":
            tic = "vix"
            downloader = YahooDownloader()
            df_vix = downloader.fetch_data(start_date=start_date,
                                         end_date=end_date,
                                         ticker_list=["^VIX"],)
            
        # Downlaod VXX using AlpacaDownloader
        elif self.data_source == "Alpaca":
            print("Downloading the VXX proxy for ^VIX instead of the latter.\n"\
                  "(Since ^VIX is not a US equity, its data cannot be obtained from Alpaca.)")
            tic = "vxx"
            # 23/03/26 update: Uses AlpacaDownloaderAug instead of AlpacaDownloader
            ### WARNING: BEWARE OF IMPORTED DOWNLOADER CLASS WHEN MODIFYING THE FEATURE ENGINEER MODULE
            downloader = AlpacaDownloader(self.api_key, self.api_secret)
            df_vix = downloader.fetch_data(start_date=start_date,
                                           end_date=end_date,
                                           ticker_list = ["VXX"],
                                           timeframe = self.timeframe,
                                           extended_trading_hours = self.extended_trading_hours,
                                           OHLCV_format = True,
                                           convert_dates_to_str = True,)

        vix = df_vix[["date", "close"]]
        vix.columns = ["date", tic]

        df = df.merge(vix, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    ##################################
    ## Add (Kritzman-Li) Turbulence ##
    ##################################
    def add_turbulence(self, data):
        """
            Add the Kritman-Li turbulence index from the output of calculate_turbulence().
            Called by preprocess_data().
            
            :param data: pd.DataFrame of clean data.
            
            :return df: Input dataframe augmented with turbulence values for each asset.
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    ######################################
    ## Compute (Kritzman-Li) Turbulence ##
    ######################################
    ## 22/10/28, Cheng C: numpy version of Turbulence calculation.
    ## Tested this function in notebook test_np_turbulence_function.ipynb 
    ## in paperspace, get same result as FinRL's original computations.
    ## Original FinRL submodule: finrl.meta.preprocessors.preprocessors

    def calculate_turbulence(self, data):
        '''
            calculate user defined features
        
            :param data: Cleaned pandas dataframe
        
            :return df: pd.DataFrame with date and turb, len is number of unique days in df.

            Note (22/10/28, C. 'Bill' Cheng): Input is already free of missing so no need to do 
            .dropna(axis=1) as original turbulence function. Definition can be found at:
            https://www.top1000funds.com/wp-content/uploads/2010/11/FAJskulls.pdf
            https://www.tandfonline.com/doi/abs/10.2469/faj.v66.n5.3
            
        '''
        
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
