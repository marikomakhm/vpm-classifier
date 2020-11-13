import numpy as np
import pandas as pd
import pickle

import spacy

from preprocessing import extract_text_data
from feature_processing import get_general_features
from vector_processing import get_pattern_vectors

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
  '--s', dest='source_dir', default='.', help='Source directory')
parser.add_argument(
  '--d', dest='data_predict', required=True, help='Path to prediction metadata')
parser.add_argument(
  '--m', dest='model_dir', required=True, help='Path to trained model directory')

args = parser.parse_args()

SOURCE_DIR = args.source_dir
OUTPUT_DIR = SOURCE_DIR + 'output/'
ERROR_DIR = SOURCE_DIR + 'errors/'

# csv file containing rows: (filename, url)
DATA_PREDICT = args.data_predict
OUT_DATA_DIR = OUTPUT_DIR + 'data/'
MODEL_DIR = args.model_dir

PATTERN_PARAMS = {'patent_number': 
                  {'pattern': r'\b(\d{1}[,\s]\d{3}[,\s]\d{3})\b', 
                   'n_chars': 140, 'no_below': 9, 'no_above': 0.9},
                  'patent':
                  {'pattern': r'(patent)',
                   'n_chars': 140, 'no_below': 11, 'no_above': 0.9}
                 }

print('SOURCE DIRECTORY:', SOURCE_DIR)

if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

print('CREATED OUTPUT DIRECTORY:', OUTPUT_DIR)

if not os.path.isdir(ERROR_DIR):
        os.mkdir(ERROR_DIR)

print('CREATED ERROR DIRECTORY:', ERROR_DIR)

print('READING DATA...')
df = pd.read_csv(DATA_PREDICT)

success_indices = extract_text_data(in_filepaths=df.filename.values,
                                    out_data_dir=OUT_DATA_DIR,
                                    error_dir=ERROR_DIR)

df = df.iloc[success_indices]
df['filename'] = df['filename'].apply(lambda f: OUT_DATA_DIR + f.split('/')[-1])
df.reset_index(drop=True, inplace=True)

print('EXTRACTED TEXT HTML TO:', OUT_DATA_DIR)

print('GENERATING FEATURES...')
df = get_general_features(df,
                          save=True,
                          save_as=OUTPUT_DIR + 'predict_features_df.csv')

# PREDICTED PROBA DF
pp_df = pd.DataFrame()

nlp  = spacy.load('en_core_web_md')

# VECTOR MODELS
vector_models = {}

for pattern_name, params in PATTERN_PARAMS.items():
    # load vector model
    vector_models[pattern_name] = pickle.load(
        open(MODEL_DIR + pattern_name + '_vector_model.sav', 'rb'))

    # get vector inputs
    vectors, non_null_indices = get_pattern_vectors(
        df.html, nlp, pattern_name, params['pattern'], False, MODEL_DIR,
        params['n_chars'], params['no_below'], params['no_above'], False)

    # set default probability to equal probability for each class
    pp = np.tile([0.33, 0.33, 0.33], (len(df), 1))
    
    non_null_vectors = np.array(list(vectors.loc[non_null_indices].values))

    if len(non_null_indices) == 1:
        non_null_vectors = non_null_vectors.reshape(
            (1, len(non_null_vectors[0])))
    
    if len(non_null_indices):
        # predict proba for non null rows
        pp_subset = vector_models[pattern_name].predict_proba(non_null_vectors)

        # replace default with predicted proba for non null rows
        pp[non_null_indices] = pp_subset
    
    # add columns corresponding to predicted probas
    pp_df = pd.concat(
        [pp_df,
         pd.DataFrame(pp, columns=[pattern_name + '_hvpm',
                                   pattern_name + '_nvpm',
                                   pattern_name + '_svpm'])], axis=1)

# FEATURE MODEL

# load best features
best_features = pickle.load(open(MODEL_DIR + 'best_features.txt', 'rb'))

# load feature model
feature_model = pickle.load(open(MODEL_DIR + 'feature_model.sav', 'rb'))

# get predicted proba for feature model
pp = feature_model.predict_proba(df[best_features])

# add to predicted proba df
pp_df = pd.concat(
    [pp_df,
    pd.DataFrame(pp, columns=['feature_hvpm',
                              'feature_nvpm',
                              'feature_svpm'])], axis=1)


# LOAD MODEL
model = pickle.load(open(MODEL_DIR + 'stacked_model.sav', 'rb'))

print('Number of features expected by model: %d' % model.n_features_)
print('Number of features in dataframe: %d' % len(pp_df.columns))

# GET PREDICTIONS
out = pd.DataFrame(model.predict_proba(pp_df),
                   columns=['pred_hvpm', 'pred_nvpm', 'pred_svpm'])

out['pred'] = model.predict(pp_df)

out['filename'] = df['filename']

# SAVE PREDICTIONS
out.to_csv(OUTPUT_DIR + 'predictions.csv', encoding='utf-8', index=False)

print('PREDICTIONS SAVED!')