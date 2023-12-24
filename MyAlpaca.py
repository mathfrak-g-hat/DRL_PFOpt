# ALpaca API
API_KEY = "ALPACA_API_KEY"
API_SECRET = "ALPACA_API_SECRET"
API_BASE_URL = 'https://paper-api.alpaca.markets'

# S&P500 top 50 list
## Note: Should be updated (Meta is still there)
SP50_TICKER_LIST = ["AAPL", "MSFT", "AMZN", "TSLA", "BRK-B",
        "UNH", "GOOGL", "XOM", "JNJ", "GOOG",
        "JPM", "NVDA", "CVX", "V", "PG",
        "HD", "LLY", "MA", "PFE", "ABBV",
        "BAC", "MRK", "PEP", "KO", "COST",
        "META", "MCD", "WMT", "TMO", "CSCO", 
        "DIS", "AVGO", "WFC", "COP", "ABT",
        "BMY", "ACN", "DHR", "VZ", "NEE",
        "LIN", "CRM", "TXN", "AMGN", "RTX",
        "HON", "PM", "ADBE", "CMCSA", "T",]

# S&P500 top 101 list
SP100_TICKER_LIST = [ "AAPL", "ABBV", "ABT", "ACN", "ADBE",
                     "AIG", "AMD", "AMGN", "AMT", "AMZN",
                     "AVGO", "AXP", "BA", "BAC", "BK",
                     "BKNG", "BLK", "BMY", "BRK-B", "C",
                     "CAT", "CHTR", "CL", "CMCSA", "COF",
                     "COP", "COST", "CRM", "CSCO", "CVS",
                     "CVX", "DHR", "DIS", "DOW", "DUK",
                     "EMR", "EXC", "F", "FDX", "GD",
                     "GE", "GILD", "GM", "GOOG", "GOOGL",
                     "GS", "HD", "HON", "IBM", "INTC",
                     "JNJ", "JPM", "KHC", "KO", "LIN",
                     "LLY", "LMT", "LOW", "MA", "MCD",
                     "MDLZ", "MDT", "MET", "META", "MMM",
                     "MO", "MRK", "MS", "MSFT", "NEE",
                     "NFLX", "NKE", "NVDA", "ORCL", "PEP",
                     "PFE", "PG", "PM", "PYPL", "QCOM",
                     "RTX", "SBUX", "SCHW", "SO", "SPG",
                     "T", "TGT", "TMO", "TMUS", "TSLA",
                     "TXN", "UNH", "UNP", "UPS", "USB",
                     "V", "VZ", "WBA", "WFC", "WMT",
                     "XOM",]

# NASDAQ100
NSDQ100_TICKER_LIST = ["ATVI", "ADBE", "ADP", "ABNB", "ALGN",
                       "GOOGL", "GOOG", "AMZN", "AMD", "AEP",
                       "AMGN", "ADI", "ANSS", "AAPL", "AMAT",
                       "ASML", "AZN", "TEAM", "ADSK", "BIDU",
                       "BIIB", "BKNG", "AVGO", "CDNS", "CHTR",
                       "CTAS", "CSCO", "CTSH", "CMCSA", "CEG",
                       "CPRT", "COST", "CRWD", "CSX", "DDOG",
                       "DXCM", "DOCU", "DLTR", "EBAY", "EA",
                       "ENPH", "EXC", "FAST", "FISV", "FTNT",
                       "GILD", "HON", "IDXX", "ILMN", "INTC",
                       "INTU", "ISRG", "JD", "KDP", "KLAC",
                       "KHC", "LRCX", "LCID", "LULU", "MAR",
                       "MRVL", "MTCH", "MELI", "META", "MCHP",
                       "MU", "MSFT", "MRNA", "MDLZ", "MNST",
                       "NTES", "NFLX", "NVDA", "NXPI", "ORLY",
                       "ODFL", "PCAR", "PANW", "PAYX", "PYPL",
                       "PEP", "PDD", "QCOM", "REGN", "ROST",
                       "SGEN", "SIRI", "SWKS", "SPLK", "SBUX",
                       "SNPS", "TMUS", "TSLA", "TXN", "VRSN",
                       "VRSK", "VRTX", "WBA", "WDAY", "XEL",
                       "ZM", "ZS",]



# Dow Jones Industrial Avg list
## Note: Should also be updated from time to time
DJIA_TICKER_LIST = ["AAPL", "AMGN", "AXP", "BA", "CAT",
                    "CRM", "CSCO", "CVX", "DIS", "DOW",
                    "GS", "HD", "HON", "IBM", "INTC",
                    "JNJ", "JPM", "KO", "MCD", "MMM",
                    "MRK", "MSFT", "NKE", "PG", "TRV",
                    "UNH", "V", "VZ", "WBA", "WMT",]

# Aboussalah's list
EX_1_LIST = ["AMT", "AXP", "BA", "CVX", "JNJ", 
             "KO", "MCD", "MSFT", "T", "WMT",]

# My list (22/12/02)
## This is a union of all lists abbove
## EX_1_LIST and SP50 are contained in SP100
## The ticker 'TRV' in DJIA is missing from NASDAQ100 and SP100
DATABASE_TICKER_LIST = set(NSDQ100_TICKER_LIST).union(set(SP100_TICKER_LIST))
DATABASE_TICKER_LIST = list(DATABASE_TICKER_LIST.union(set(DJIA_TICKER_LIST)))
DATABASE_TICKER_LIST.sort()