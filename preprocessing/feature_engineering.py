import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def extract_features(df):
    """
    Applies hour extraction, age calculation, and haversine distance.
    Drops original columns exactly as per the notebook.
    """
    df_out = df.copy()

    # isolate time
    df_out["trans_date_trans_time"] = pd.to_datetime(df_out["trans_date_trans_time"])
    df_out["hour"] = df_out["trans_date_trans_time"].dt.hour
    df_out = df_out.drop(columns='trans_date_trans_time')

    # calculate age
    df_out["dob"] = pd.to_datetime(df_out["dob"])
    df_out["age"] = (pd.Timestamp.now() - df_out["dob"]).dt.days // 365
    df_out = df_out.drop(columns='dob')

    # calculate distances
    df_out["distance"] = haversine(
        df_out["lat"],
        df_out["long"],
        df_out["merch_lat"],
        df_out["merch_long"]
    )
    cols_to_drop = ["lat", "long", "merch_lat", "merch_long"]
    df_out = df_out.drop(columns=cols_to_drop)

    return df_out
