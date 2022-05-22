import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    load messages and categories datasets and merge into one dataframe

    Args:
    messages_filepath: string. File path for messages source dataset.
    categories_filepath: string. File path for messages source dataset.

    Returns:
    df: dataframe. Dataframes that combined messages and categories datasets.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    #Merge the messages and categories datasets using the common id
    #Assign this combined dataset to df, which will be cleaned in the following steps
    # merge datasets
    df = messages.merge(categories, left_on='id', right_on='id', how='left')
    return df

def clean_data(df):
    """
    data cleanning and data wrangling of the dataframe.

    Args:
    df: dataframe. Dataframes that combined messages and categories datasets.

    Returns:
    df: dataframe. Dataframes after data cleaning and pre-processing.
    """
    #Split categories into separate category columns
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [x[0:(len(x)-2)] for x in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    #Iterate through the category columns in df to keep only the last character of each string (the 1 or 0).
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
    # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    
    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Remove rows with a related value of 2
    df = df[df['related'] != 2]

    # drop duplicates
    df = df.drop_duplicates()
    # check number of duplicates
    df.duplicated().value_counts()
    return df

def save_data(df, database_filename):
    """
    save the dataframe into a sql database.

    Args:
    df: dataframe. Dataframes after data cleaning and pre-processing.
    database_filename: string. Filename for saved sql database.

    Returns:
    None
    """
    engine = create_engine('sqlite:///' + database_filename) 
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()