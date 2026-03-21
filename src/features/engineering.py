import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Tuple


def create_time_features(df):
    df_new = df.copy()

    df_new['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

    df_new['hour'] = df_new['trans_date_trans_time'].dt.hour
    df_new['day_of_week'] = df_new['trans_date_trans_time'].dt.weekday
    df_new['month'] = df_new['trans_date_trans_time'].dt.month
    df_new['is_weekend'] = df_new['day_of_week'].isin([5, 6]).astype(int)
    df_new['is_night'] = df_new['hour'].isin([22, 23, 0, 1, 2, 3]).astype(int)

    return df_new

def create_amount_features(df):
    df_new = df.copy()

    df_new['amt_log'] = np.log1p(df['amt'])
    df_new['amt_zscore'] = (df['amt'] - df['amt'].mean()) / df['amt'].std()
    
    return df_new

def create_age_feature(df):
    df_new = df.copy()

    df_new['dob'] = pd.to_datetime(df_new['dob'])
    df_new['age'] = df_new['trans_date_trans_time'].dt.year - df_new['dob'].dt.year
    # Drop dob after creating age
    df_new.drop('dob', axis=1, inplace=True) 
    
    return df_new

def compute_distance_km(df):
    """
    Compute Haversine distance between cardholder and merchant
    """
    earth_radius_km = 6371.0

    lat1 = np.radians(df['lat'])
    lon1 = np.radians(df['long'])
    lat2 = np.radians(df['merch_lat'])
    lon2 = np.radians(df['merch_long'])

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = np.sin(delta_lat / 2) ** 2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2

    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius_km * c

def create_geo_features(df):

    df_new = df.copy()

    df_new['distance_km'] = compute_distance_km(df_new)

    return df_new

def create_aggregation_features(df):
    df_new = df.copy()

    df_new['merchant_freq'] = df['merchant'].map(df['merchant'].value_counts())
    df_new['category_freq'] = df['category'].map(df['category'].value_counts())
    df_new['city_freq'] = df['city'].map(df['city'].value_counts())

    return df_new

def target_encode(train, test, col, target='is_fraud'):
    # ONLY APPLY ON TRAIN → then map to test
    means = train.groupby(col)[target].mean()
    train[col + '_fraud_rate'] = train[col].map(means)
    test[col + '_fraud_rate'] = test[col].map(means)
    
    return train, test