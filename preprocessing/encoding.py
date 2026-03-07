import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def fit_encoders(df):
    """
    Fits OneHotEncoder and OrdinalEncoder based on training data.
    Returns the fitted encoders.
    """
    oo = OneHotEncoder(sparse_output=False)
    oo.fit(df[['gender', 'category', 'state']])

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(df[['merchant', 'city', 'job']])

    return oo, enc

def apply_encoders(df, oo, enc):
    """
    Applies fitted encoders to a dataframe.
    """
    df_out = df.copy()
    
    # OneHotEncoding
    encoded = oo.transform(df_out[['gender', 'category', 'state']])
    encoded_df = pd.DataFrame(
        encoded,
        columns=oo.get_feature_names_out(['gender', 'category', 'state']),
        index=df_out.index
    )
    df_out = pd.concat([df_out.drop(['gender', 'category', 'state'], axis=1), encoded_df], axis=1)
    
    onehot_cols = df_out.filter(regex='^(gender_|category_|state_)').columns
    df_out[onehot_cols] = df_out[onehot_cols].astype('bool')

    # OrdinalEncoding
    df_out[['merchant', 'city', 'job']] = enc.transform(df_out[['merchant', 'city', 'job']])
    
    return df_out
