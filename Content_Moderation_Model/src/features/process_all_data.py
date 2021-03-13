import pickle
import re
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from bs4 import BeautifulSoup

import nltk
from nltk.stem import WordNetLemmatizer



def process_text_series(series):
    """Pre-processes text by lower-casing, replacing newlines & punctuation, and lemmatizing"""
    series = series.str.lower()
    series = series.str.replace("\n", " ")
    series = series.str.replace("[.!?\\-]", " X ") # replace all punctuation with an upper-case X

    return series

def create_regex_matcher(series):
    if type(series) != list:
        series = series.to_list()
    matcher = [r'\b'+ s +r'\b' for s in series]
    matcher = r'|'.join(matcher)
    
    return matcher

def extract_string_features(df):
    """Function to extract string length, word counts, and average word length for Title and Text columns of
    a csv file, and create new columns with those values 
    
    Parameters
    ----------
    df: Pandas DataFrame
    
    Returns
    ----------
    df: Pandas DataFrame
    """
    def avg_word_length(string):
        """Function to determine the average length of words in a string"""
        word_list = string.split()
        if len(word_list) == 0:
            return 0
        else:
            total = 0
            for word in word_list:
                total += len(word)
            return round(total / len(word_list), 2)
        
    # Replace NaNs with empty strings ''
    df.fillna(value={'Title': '', 'Text': ''}, inplace=True)

    # Create columns of string lengths for Title and Text columns
    df['title_len'] = df.Title.apply(len)
    df['text_len'] = df.Text.apply(len)
    df['titletext_len'] = df.TitleText.apply(len)
    

    # Create columns of word counts for Title and Text columns
    df['title_word_no'] = df.Title.str.split().apply(len)
    df['text_word_no'] = df.Text.str.split().apply(len)
    df['titletext_word_no'] = df.TitleText.str.split().apply(len)

    # Create columns of average word length for Title and Text columns
    df['title_avg_word_length'] = df.Title.apply(avg_word_length)
    df['text_avg_word_length'] = df.Text.apply(avg_word_length)
    df['titletext_avg_word_length'] = df.TitleText.apply(avg_word_length)

    return df

def extract_date_features(df):
    """Extract year, month, day and year/month features from datetime info"""

    df['year'] = pd.DatetimeIndex(df['PostedDate']).year
    df['month'] = pd.DatetimeIndex(df['PostedDate']).month
    df['hour'] = pd.DatetimeIndex(df['PostedDate']).hour
    df['YYMM'] = 100*(df['year']%2000) + df['month']

    return df


def process_data(df):
    """Cleans data from iHerb and re-assigns ViolationBool & Violation
    Inputs: Pandas DataFrame 
    Returns: Pandas Dataframe"""

    # Download relevant NLTK corpora
    nltk.download('punkt')
    print("------------------------------------------------------------")
    nltk.download('wordnet')

    # Saving the lemmatizer into an object
    wordnet_lemmatizer = WordNetLemmatizer()

    # Data cleaning and homogenizing
    start_time = time.time()

    df['TitleText'] = (df['Title'] + ' ' + df['Text']).fillna(' ')
    
    df['TitleText'] = process_text_series(df['TitleText'])

    df['TitleText'] = list(map(wordnet_lemmatizer.lemmatize, df['TitleText']))
    print('Data cleaning complete.')

    # Add columns, if not present
    if 'Violation' not in df.columns:
        df['Violation'] = df['Violation']
        print('Violation column added.')

    if 'ViolationBool' not in df.columns:
        df['ViolationBool'] = np.where(df['Violation'] > 0, 1, 0)
        print('ViolationBool column added.')

    # Fill NaNs with 0 in Violation column   
    df['Violation'] = df['Violation'].fillna(0)
    print('Violation NaNs filled.')

    # Remove observations from original violations data set
    orig = pd.read_csv("../data/raw/reviews_violations_english_v5.csv")
    df = df[~df['Id'].isin(orig['ReviewID'])]
    print('Original data set observations removed.')

    # This section does pattern matching for the officially banned words, phrases, & names
    ugc_names = pd.read_csv('../data/raw/ugc_names.csv')

    # Replicate data cleaning & homogenization, add word boundaries to the regex, & create regex matcher
    ugc_names['Banned words'] = process_text_series(ugc_names['Banned words'])  
    ugc_names = create_regex_matcher(ugc_names['Banned words'])
    print('UGC names cleaning complete.')

    # Create list of provided medical words
    medical_words = ['cure', 'cured', 'curing', 'prevent', 'preventing', 'treat', 'treating']

    # Replicate data cleaning & homogenization, add word boundaries to the regex, & create regex matcher
    medical_words = create_regex_matcher(medical_words)
    print('Medical words cleaning complete.')

    # Import provided English profanity & rename column from -1 to Profanity
    profanity = pd.read_excel('../data/raw/UCM bad words.xlsx', engine='openpyxl', sheet_name='English (en-US)')
    profanity.columns=['Profanity']

    # Replicate data cleaning & homogenization
    profanity['Profanity'] = process_text_series(profanity['Profanity'])   
    profanity = create_regex_matcher(profanity['Profanity'])
    print('Profanity words cleaning complete.')

    # Import Emoji list from Unicode & create regex matcher

    # Start timing emoji_list execution
    start_emoji = time.time()
    with open('../data/raw/view-source_www.unicode.org_emoji_charts_full-emoji-list.html', 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read())
        emoji_list = soup.prettify()
        
        # Separates 4-digit & 5-digit unicodes to facilitate regex matching
        emoji1 = re.findall(r'[\>\s]U\+[\dABCDEF]{4}[\<\s]', emoji_list)
        emoji2 = re.findall(r'[\>\s]U\+[\dABCDEF]{5}[\<\s]', emoji_list)

    emoji1 = [e.replace('U+', '\\U0000') for e in emoji1]
    emoji2 = [e.replace('U+', '\\U000') for e in emoji2]
    emoji = emoji1 + emoji2

    emoji = list(set(emoji)) # Remove duplicated values
    emoji = [e.replace('<', '').replace('>', '').strip() for e in emoji]

    emoji = sorted(emoji)
    emoji = emoji[12:] # This removes common punctuation marks & numbers from the list
    emoji = r'|'.join(emoji)
    print('Emoji cleaning execution time: {}'.format(time.time() - start_emoji))
    print('Emoji cleaning complete.')

    # Auto match certain reviews
    DEPRECATED_VIOLATIONS = df['Violation'].isin([6, 8, 12])
    print('DEPRECATED_VIOLATIONS created')
    HAS_LOW_QUALITY = df['TitleText'].str.len() < 20
    print('HAS_LOW_QUALITY created')

    has_banned_words_start = time.time()
    HAS_BANNED_WORDS = df['TitleText'].str.contains(ugc_names) 
    print('HAS_BANNED_WORDS execution time: {}'.format(time.time() - has_banned_words_start))
    print('HAS_BANNED_WORDS created')

    HAS_MEDICAL_ADVICE = df['TitleText'].str.contains(medical_words)
    print('HAS_MEDICAL_ADVICE created')

    has_profanity_start = time.time()
    HAS_PROFANITY = df['TitleText'].str.contains(profanity)
    print('HAS_PROFANITY execution time: {}'.format(time.time() - has_profanity_start))
    print('HAS_PROFANITY created')

    # Time HAS_EMOJI_VIOLATION execution
    start_emoji_violation = time.time()
    HAS_EMOJI_VIOLATION = df['TitleText'].str.count(emoji) > 3
    print('HAS_EMOJI_VIOLATION execution time: {}'.format(time.time() - start_emoji_violation))
    print('HAS_EMOJI_VIOLATION created')
   

    # Create dummy columns to speed text matches
    df['Low quality'] = np.where(HAS_LOW_QUALITY, 1, 0)
    df['Banned words'] = np.where(HAS_BANNED_WORDS, 1, 0)
    df['Medical'] = np.where(HAS_MEDICAL_ADVICE, 1, 0)
    df['Profanity'] = np.where(HAS_PROFANITY, 1, 0)
    # Time creation of Emoji column
    start_emoji_column = time.time()
    df['Emoji'] = np.where(HAS_EMOJI_VIOLATION, 1, 0) 
    print('Emoji column creation time: {}'.format(time.time() - start_emoji_column))   
    print('Dummy columns created.')

    # Change Violation Boolean for all auto matches & for non-emoji violations with 50+ characters marked as emoji violations
    df['ViolationBool'] = np.where(DEPRECATED_VIOLATIONS, 0, df['ViolationBool'])
    df['ViolationBool'] = np.where((df['Medical']==1)|(df['Profanity']==1)|(df['Emoji']==1)|(df['Banned words']==1), 1, df['ViolationBool'])
    df['ViolationBool'] = np.where((df['Emoji']==0)&(df['Violation']==7)&(df['TitleText'].str.len()>=50), 0, df['ViolationBool'])
    df['ViolationBool'] = np.where((df['Low quality']==1)&(df['Violation']==0), 1, df['ViolationBool'])
    print('Violation boolean changed for all auto matches & for non-emoji violations with 50+ characters marked as emoji violations.')

    # Relabel deprecated or unpredictable violations
    df['Violation'] = np.where(DEPRECATED_VIOLATIONS, 0, df['Violation'])

    # Create column to sum number of violations caught by simulated UGC system
    df['Auto_Violations'] = df['Banned words'] + df['Medical'] + df['Profanity'] + df['Emoji']

    # Set classes for auto-violations
    df['Violation'] = np.where((df['Low quality']==1)&(df['Violation']==0), 3, df['Violation'])
    df['Violation'] = np.where(df['Banned words']==1, 2, df['Violation'])
    df['Violation'] = np.where(df['Medical']==1, 9, df['Violation']) # This is matched between Banned words & Profanity since r"cure(d|ing)?" are in the Banned words list
    df['Violation'] = np.where(df['Profanity']==1, 2, df['Violation'])
    df['Violation'] = np.where(df['Emoji']==1, 7, df['Violation'])
    print('Classes set for auto-violations.')

    # Re-label mislabelled emoji violations
    df['Violation'] = np.where((df['Emoji']==0)&(df['Violation']==7)&(df.TitleText.str.len()<50), 3, df['Violation'])
    df['Violation'] = np.where((df['Emoji']==0)&(df['Violation']==7)&(df.TitleText.str.len()>=50), 0, df['Violation'])
    print('Mislabeled emojis re-labeled.')

    # Fill missing values as there are some; but this information could be valuable
    df['LanguageCode'] = df['LanguageCode'].fillna('nu-LL')
    print('Language code missing values filled in.')

    # Add text features
    df = extract_string_features(df)
    print('Text features added.')

    # Add date features
    df = extract_date_features(df)
    print('Date features added.')

    # Return processed df
    return df

    # Print clean_data execution time 
    print("clean_data execution time: {}".format(time.time()-start_time))

    return df
