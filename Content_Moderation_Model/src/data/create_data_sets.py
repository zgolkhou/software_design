import pandas as pd
import numpy as np
from features.process_all_data import process_data
from sklearn.model_selection import train_test_split

def create_data():
    """create balanced, unbalanced, train, validation and test sets
    Input: Path to source csv files for data set
    Output: Saves several csv data sets to /data/processed folder
    """

    # Import raw data set
    d0 = pd.read_csv("../data/raw/000000000000.csv")
    d1 = pd.read_csv("../data/raw/000000000001.csv")
    d2 = pd.read_csv("../data/raw/000000000002.csv")
    d3 = pd.read_csv("../data/raw/000000000003.csv")
    d4 = pd.read_csv("../data/raw/000000000004.csv")
    df = pd.concat([d0, d1, d2, d3, d4])

    # Constrain dates of data to 2019.01-2020.09
    df = df[(df['PostedDate']>='2019-01-01') & (df['PostedDate']<='2020-09-30')]
    print('Data set reduced to 2019.01-2020.09 time frame.')

    # Run data cleaning function. This might take quite a while to run, from 20mins to over 1hr for 1.7M rows.
    df = process_data(df) 
    print('Data cleaning complete.')

    # Create out-of-time data set for 2020.08-2020.09
    df_oot = df[(df['PostedDate']>='2020-08-01') & (df['PostedDate']<='2020-09-30')]
    df_oot.to_csv('../data/processed/oot_validation_202008_202009.csv')
    print('Out-of-time data set created and exported.')

    # Create balanced dataset with 100k in each category
    df = df[(df['PostedDate']>='2019-01-01') & (df['PostedDate']<='2020-07-31')]
    c0 = df[df['Violation']!=0].sample(n=100000, random_state=2)
    c1 = df[df['Violation']==0].sample(n=100000, random_state=2)
    s0 = pd.concat([c0, c1])
    s0.to_csv('../data/processed/balanced_test.csv')
    print('Balanced dataset created and exported.')

    # Create unbalanced dataset
    df_train, df_valid = train_test_split(df, train_size = 0.5, random_state=2)
    df_train.to_csv('../data/processed/unbalanced_train.csv')
    df_valid.to_csv('../data/processed/unbalanced_valid.csv')
    print('Unbalanced data set completed and exported.')


