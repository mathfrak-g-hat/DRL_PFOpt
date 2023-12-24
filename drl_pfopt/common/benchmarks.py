## AJ Zerouali, 22/11/24
## Functions to obtain benchmarks for backtests
from __future__ import annotations, print_function, division

import time
from datetime import datetime, timedelta
import pandas_market_calendars as pd_mkt_cals
from builtins import range

# np, pd plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyFolio imports
import pyfolio
from pyfolio import create_full_tear_sheet
from pyfolio.timeseries import perf_stats

from drl_pfopt.common.data.downloaders import AlpacaDownloader, YahooDownloader
from drl_pfopt.common.data.data_utils import get_str_date_format, get_daily_return
from drl_pfopt.common.envs import PortfolioOptEnv

#######################################################
#### Download benchmark prices and compute returns ####
#######################################################
### 22/11/23, AJ Zerouali
def get_benchmark_prices_and_returns(ticker: str, # "^DJIA"
                                     start_date: str, 
                                     end_date: str,
                                     data_source: str = "Yahoo",
                                     api_key: str = "",
                                     api_secret: str = "",
                                     timeframe: str = "Day",
                                     extended_trading_hours: bool = False,
                                     ):
    '''
        Helper function for downloading data of a benchmark ticker,
        and computes the benchmark daily returns over the close prices 
        in the specified time period. Uses FinRL's dataframe formatting.
        
        ### Note (22/10/20): Temporary implementation borrowed from FinRL, 
        ###             to be modified. Helper function for benchmark returns
        ###             required by PFOpt_DRL_Agent.plot_backtest_results().
        
        :param ticker: Benchmark ticker.
        :param start_date: Start date of data.
        :param end_date: End date of data.
        
        :return df_benchmark: pd.DataFrame of benchmark ticker,
                                index = Period int index in data,
                                columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic', 'day'].
        :return df_benchmark_returns: pd.DataFrame of daily returns (over 'close'),
                                index = Period int index in data,
                                columns = ['date', 'daily_return'].
                              

    '''
    # Check data_source param
    if data_source not in ["Alpaca","Yahoo"]:
        raise ValueError("The data_source parameter must be one of the following keywords:\n"\
                             "            \"Alpaca\", or \"Yahoo\".")
        
    # Download VIX using YahooDownloader
    if data_source == "Yahoo":
        downloader = YahooDownloader()
        df_benchmark = downloader.fetch_data(start_date=start_date,
                                             end_date=end_date,
                                             ticker_list=[ticker],)
    
    # Downlaod VXX using AlpacaDownloader
    elif data_source == "Alpaca":
        downloader = AlpacaDownloader(api_key,api_secret)
        df_benchmark = downloader.fetch_data(start_date=start_date,
                                             end_date=end_date,
                                             ticker_list = [ticker],
                                             timeframe = timeframe,
                                             extended_trading_hours = extended_trading_hours,
                                             OHLCV_format = True,
                                             convert_dates_to_str = True,)
    
    # Compute returns on "close" of df_benchmark, get np.ndarray of returns
    benchmark_returns_array = get_daily_return(df_benchmark, value_col_name="close").to_numpy()
    
    # Make returns dataframe
    df_benchmark_returns = pd.DataFrame({"date":list(df_benchmark.date), 
                                         "daily_return" : benchmark_returns_array})
    df_benchmark_returns = df_benchmark_returns.fillna(method="ffill").fillna(method="bfill") 
        
    return df_benchmark, df_benchmark_returns

##########################################################
#### Execute Equal Weights Strategy in an Environment ####
##########################################################
### 22/11/15, AJ Zerouali
def get_eq_wts_benchmark(env: PortfolioOptEnv):
    '''
        Function to compute the portfolio value, returns, and Sharpe ratio
        of an equal weights strategy, given the data of a portfolio 
        optimization environment.
        
        :param env: DRL_PFOpt_gymEnv.PortfolioOptEnv object. 
        
        :return ew_benchmark_dict: dict of equal weights strategy results
                    over data in env. Keys are as follows:
                    * "value_hist": History of portfolio values.
                    * "return_hist": History of portfolio returns.
                    * "Sharpe_ratio": Sharpe ratio of the strategy over the period.
                    * "performance_stats": Performance stats computed by
                                        pyfolio.timeseries.perf_stats
    '''
    # Initializations
    env.reset()
    actions_eq_wts = np.ones(shape = (env.n_assets,))/env.n_assets
    str_date_format = get_str_date_format(env.date_hist)
    
    # Main loop
    for i in range(env.N_periods):
        env.step(actions_eq_wts)
        if i == env.N_periods - 2:
            df_ew_value_hist = env.get_pf_value_hist()
            df_ew_return_hist = env.get_pf_return_hist()
            ew_Sharpe_ratio = (np.sqrt(252)*df_ew_return_hist["daily_return"].to_numpy().mean())\
                            /(df_ew_return_hist["daily_return"].to_numpy().std())
    
    env.reset()
    
    # Make performance stats dataframe from pyfolio's perf_stats
    df_returns_ = df_ew_return_hist.copy()
    df_returns_["date"] = pd.to_datetime(df_returns_["date"], 
                                         format=str_date_format)
    df_returns_ = df_returns_.set_index('date')
    series_performance_stats = perf_stats(returns=df_returns_["daily_return"])
    del df_returns_
    df_ew_perf_stats = pd.DataFrame(series_performance_stats)
    df_ew_perf_stats = df_ew_perf_stats.rename(columns = {0:"Value"})
    
    # Prepare output
    ew_benchmark_dict = {}
    ew_benchmark_dict["value_hist"] = df_ew_value_hist
    ew_benchmark_dict["return_hist"] = df_ew_return_hist
    ew_benchmark_dict["Sharpe_ratio"] = ew_Sharpe_ratio
    ew_benchmark_dict["performance_stats"] = df_ew_perf_stats
    
    return ew_benchmark_dict