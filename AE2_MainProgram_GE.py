import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
# List of 30 cryptocurrencies
crypto_list = [
    'BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'BCH-USD', 'ADA-USD', 'DOT-USD',
    'BNB-USD', 'LINK-USD', 'XLM-USD', 'DOGE-USD', 'UNI-USD', 'AAVE-USD', 'ATOM-USD',
    'AVAX-USD', 'PEPE24478-USD', 'SOL-USD', 'CHR-USD', 'ALGO-USD', 'FTT-USD', 'VET-USD',
    'USDT-USD', 'TRX-USD', 'ETC-USD', 'XMR-USD', 'EOS-USD', 'THETA-USD', 'NEO-USD',
    'DASH-USD', 'WIF-USD'
]
def download_data(cryptos, duration):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(duration-1))

    #initialize empty data frame
    data=pd.DataFrame()

    data = yf.download(cryptos, start=start_date, end=end_date)
    return data

df = download_data(crypto_list, 365) 

#Convert downloaded data to csv format
df.to_csv(r'All_Data.csv',index=True, header = True)