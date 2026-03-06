import yfinance as yf
import pandas as pd
import pytz

try:
    df = yf.Ticker('BTC-USD').history(period='1d', interval='1h')
    print("Crypto (BTC-USD) Index:", df.index.tz)
    print("Sample:", df.index[0])
    
    df_th = yf.Ticker('PTT.BK').history(period='1d', interval='1h')
    print("Thai Stock (PTT.BK) Index:", df_th.index.tz)
    print("Sample:", df_th.index[0])
    
    print("pytz available:", True)
except Exception as e:
    print("Error:", e)
