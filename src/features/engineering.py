import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

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

def create_aggregation_features(train, test):
    train_new = train.copy()
    test_new = test.copy()

    merchant_counts = train_new['merchant'].value_counts()

    train_new['merchant_freq'] = train_new['merchant'].map(merchant_counts)
    test_new['merchant_freq'] = test_new['merchant'].map(merchant_counts).fillna(0)

    category_counts = train_new['category'].value_counts()
    train_new['category_freq'] = train_new['category'].map(category_counts)
    test_new['category_freq'] = test_new['category'].map(category_counts).fillna(0)

    city_counts = train_new['city'].value_counts()
    train_new['city_freq'] = train_new['city'].map(city_counts)
    test_new['city_freq'] = test_new['city'].map(city_counts).fillna(0)

    return train_new, test_new

def target_encode(train, test, col, target='is_fraud', n_splits=5, alpha=10):
    train_encoded = np.zeros(len(train))
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(train, train[target]):
        X_tr, X_val = train.iloc[train_idx], train.iloc[val_idx]

        global_mean = X_tr[target].mean()
        stats = X_tr.groupby(col)[target].agg(['mean', 'count'])
        smooth = (stats['count'] * stats['mean'] + alpha * global_mean) / (stats['count'] + alpha)

        train_encoded[val_idx] = X_val[col].map(smooth)

    global_mean = train[target].mean()
    train[col + '_fraud_rate'] = np.where(np.isnan(train_encoded), global_mean, train_encoded)

    stats = train.groupby(col)[target].agg(['mean', 'count'])
    smooth = (stats['count'] * stats['mean'] + alpha * global_mean) / (stats['count'] + alpha)

    test[col + '_fraud_rate'] = test[col].map(smooth).fillna(global_mean)

    return train, test