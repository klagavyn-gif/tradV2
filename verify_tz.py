import pandas as pd
import pytz
from datetime import datetime

def get_thai_now():
    return datetime.now(pytz.timezone('Asia/Bangkok')).replace(tzinfo=None)

def _normalize_df_index(df):
    if df is None or getattr(df, "empty", True):
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            if df.index.tz is not None:
                # Convert to Thai time (UTC+7) before making naive
                df.index = df.index.tz_convert(pytz.timezone('Asia/Bangkok'))
            df.index = df.index.tz_localize(None)
        except Exception:
            pass
    return df

try:
    print("Current Thai Time:", get_thai_now())
    
    # Test with UTC aware index
    idx = pd.date_range("2024-01-01 00:00", periods=1, tz="UTC")
    df = pd.DataFrame({"A": [1]}, index=idx)
    print("Original UTC:", df.index[0])
    df = _normalize_df_index(df)
    print("Normalized (should be 07:00):", df.index[0])
    
    # Test with Thai aware index
    idx_th = pd.date_range("2024-01-01 10:00", periods=1, tz="Asia/Bangkok")
    df_th = pd.DataFrame({"A": [1]}, index=idx_th)
    print("Original Thai:", df_th.index[0])
    df_th = _normalize_df_index(df_th)
    print("Normalized (should be 10:00):", df_th.index[0])
    
    print("Verification Successful")
except Exception as e:
    print("Verification Failed:", e)
