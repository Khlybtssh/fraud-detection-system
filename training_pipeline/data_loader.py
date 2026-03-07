import pandas as pd

def load_data(train_path, test_path):
    """
    Loads raw datasets and drops original unwated columns exactly as per the notebook.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Drop irrelevant columns as per notebook
    cols_to_drop = ['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'trans_num', 'zip']
    train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
    test_df = test_df.drop(columns=cols_to_drop, errors='ignore')
    
    return train_df, test_df
