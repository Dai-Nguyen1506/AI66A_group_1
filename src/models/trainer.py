import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def split_data(df: pd.DataFrame, target_col: str = 'is_fraud'):
    y = df[target_col]

    X = df.drop(columns=[target_col])
    
    return X, y

def clean_feature_names(df):
    df.columns = [
        re.sub(r'[^A-Za-z0-9_]+', '_', col)
        for col in df.columns
    ]
    df = df.loc[:, ~df.columns.duplicated()]
    
    return df

def encoding (X_train, X_test):
    '''
        One-hot encoding is unsuitable because of high cardinality in some categorical features, 
        so we will use label encoding instead.

    '''
    # Label encoding
    for col in X_train.select_dtypes(include='object').columns:
        
        le = LabelEncoder()
        
        # Fit on combined data
        combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)

        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
    
    print("Feature shape:", X_train.shape)

    return X_train, X_test

def class_weights(y_train):
    fraud = y_train.sum()
    non_fraud = len(y_train) - fraud

    scale_pos_weight = (non_fraud / fraud).round(2)
    print("scale_pos_weight:", scale_pos_weight)

    return scale_pos_weight

def smote_oversampling(X_train, y_train):
    smote = SMOTE(sampling_strategy=0.2,random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print("After SMOTE:")
    print(pd.Series(y_res).value_counts())
    
    return X_res, y_res

def undersampling(X_train, y_train, strategy=0.5):
    rus = RandomUnderSampler(sampling_strategy=strategy, random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)

    print("After undersampling:")
    print(pd.Series(y_res).value_counts())
    
    return X_res, y_res