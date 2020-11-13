import pandas as pd

from preprocessing import extract_text_data
from feature_processing import get_general_features
from model import train_stacked_model

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
  '--s', dest='source_dir', default='.', help='Source directory')
parser.add_argument(
  '--d', dest='data_records', required=True, help='Path to data records')

args = parser.parse_args()

SOURCE_DIR = args.source_dir
OUTPUT_DIR = SOURCE_DIR + 'output/'
MODEL_DIR = OUTPUT_DIR + 'models/'
ERROR_DIR = SOURCE_DIR + 'errors/'

DATA_RECORDS = args.data_records
OUT_DATA_DIR = OUTPUT_DIR + 'data/'

PATTERN_PARAMS = {'patent_number': 
                  {'pattern': r'\b(\d{1}[,\s]\d{3}[,\s]\d{3})\b',
                   'n_chars': 140, 'no_below': 9, 'no_above': 0.9},
                  'patent':
                  {'pattern': r'(patent)',
                   'n_chars': 140, 'no_below': 11, 'no_above': 0.9}
                 }

print('SOURCE DIRECTORY:', SOURCE_DIR)
print('DATA RECORDS PATH:', DATA_RECORDS)

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

print('CREATED OUTPUT DIRECTORY:', OUTPUT_DIR)

if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)

print('CREATED MODEL DIRECTORY:', MODEL_DIR)

if not os.path.isdir(ERROR_DIR):
    os.mkdir(ERROR_DIR)

print('CREATED ERROR DIRECTORY:', ERROR_DIR)

print('READING DATA...')
df = pd.read_csv(DATA_RECORDS)

success_indices = extract_text_data(in_filepaths=df.filename.values,
                                    out_data_dir=OUT_DATA_DIR,
                                    error_dir=ERROR_DIR)

df = df.iloc[success_indices]
df['filename'] = df['filename'].apply(lambda f: OUT_DATA_DIR + f.split('/')[-1])

print('EXTRACTED TEXT HTML TO:', OUT_DATA_DIR)

print('GENERATING FEATURES...')
df = get_general_features(df,
                          save=True,
                          save_as=OUTPUT_DIR + 'train_features_df.csv')

df.reset_index(drop=True, inplace=True)
features = df.columns.drop(['url', 'cat', 'filename', 'html',])

print('TRAINING MODEL...')
train_stacked_model(df, PATTERN_PARAMS, features, MODEL_DIR,
                    n_splits=4, info=True, plot=False)

print('MODEL TRAINING COMPLETE.')
print('MODEL HAS BEEN SAVED AT {}'.format(MODEL_DIR))