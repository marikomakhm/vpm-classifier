import numpy as np
import pandas as pd

from vector_processing import get_pattern_vectors
from performance import performance_report, display_confusion_matrix

import spacy
import spacy.lang.en as en

from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier

import pickle

from typing import List, Tuple, Dict

CLASSES = ['hvpm', 'nvpm', 'svpm']

# best parameters for each model with random forest classifier
BEST_PARAMS_RFC = {
    'feature':
    {'class_weight': 'balanced', 'max_depth': 10, 'min_samples_leaf': 1,
     'min_samples_split': 16, 'n_estimators': 250, 'random_state': 0},
    'patent_number':
    {'class_weight': 'balanced', 'max_depth': 15, 'min_samples_leaf': 5,
     'min_samples_split': 20, 'n_estimators': 500, 'random_state': 0},
    'patent':
    {'class_weight': 'balanced', 'max_depth': 10, 'min_samples_leaf': 2,
     'min_samples_split': 18, 'n_estimators': 600, 'random_state': 0},
    'stacked':
    {'class_weight': 'balanced', 'max_depth': 16, 'min_samples_leaf': 2,
     'min_samples_split': 2, 'n_estimators': 300, 'random_state': 0}
}

# best parameters for stacked model with lightgbm
BEST_PARAMS_LGBM = {
    'stacked':
    {'class_weight': 'balanced', 'max_depth': 5, 'n_estimators': 35,
     'num_leaves': 12, 'random_state': 0}
}

def feature_pred_proba(df: pd.DataFrame,
                       features: List[str],
                       train_indices: List[int],
                       test_indices: List[int],
                       info: bool = True,
                       plot: bool = False) -> List[List[int]]:
    """ Gets predicted probabilities on test set of model trained on features.

    Trains and tests model using specified features and target column. Returns
    a list of predicted probabilities for each sample in the test set.

    Args:
        df: DataFrame to get training/testing data from.
        features: list of column names corresponding to training features.
        train_indices: list of training indices.
        test_indices: list of test indices.
        info: prints debugging info. Defaults to True.
        plot: plots confusion matrix. Defaults to False.

    Returns:
        List of predicted probabilities for each class for each test set sample.
    """
    X = df[features].values
    y = df['cat'].values
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    clf = RandomForestClassifier(**BEST_PARAMS_RFC['feature'])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # save model
    pickle.dump(clf, open(MODEL_DIR + 'feature_model.sav', 'wb'))    
    
    if info:
        print('FEATURES:', ', '.join(features))
        performance_report(y_test, y_pred)
    
    if plot:
        display_confusion_matrix(y_test, y_pred, CLASSES, figsize=(5, 3))
    
    return clf.predict_proba(X_test)
    
def vector_pred_proba(df: pd.DataFrame,
                      pattern_name: str,
                      params: Dict[str, str],
                      nlp: en.English,
                      train_indices: List[int],
                      test_indices: List[int],
                      info: bool = True,
                      plot: bool = False) -> List[List[int]]:
    """ Gets predicted proba on model test set trained using pattern vectors.

    Trains and tests model using GloVe embedding vectors from HTML text
    surrounding regex pattern. Returns a list of predicted probabilities for
    each sample in the test set.

    Args:
        df: DataFrame to get training/testing data from.
        pattern_name: name of regex pattern (e.g. "patent_number").
        params: dict containing regex pattern parameters (pattern name, n_chars,
                no_below, no_above)
        nlp: pretrained spacy model to use for embedding vectors.
        train_indices: list of training indices.
        test_indices: list of test indices.
        info: prints debugging info. Defaults to True.
        plot: plots confusion matrix. Defaults to False.

    Returns:
        List of predicted probabilities for each class for each test set sample.
    """

    vectors, non_null_indices = get_pattern_vectors(
        df.html, nlp, pattern_name, params['pattern'], True, MODEL_DIR,
        params['n_chars'], params['no_below'], params['no_above'], info)
    
    train_indices_subset = [i for i in train_indices if i in non_null_indices]
    test_indices_subset_dict = {
        i: x for i, x in enumerate(test_indices) if x in non_null_indices}
    test_indices_subset = list(test_indices_subset_dict.values())

    if info:
        print('dataframe length:', len(df))
        print('vectors length (should be same):', len(vectors))
        print('number of non null indices:', len(non_null_indices))
        print('train indices start/remaining: %d / %d' % (
            len(train_indices_subset), len(train_indices)))
        print('test indices start/remaining: %d / %d' % (
            len(test_indices_subset), len(test_indices)))
    
    y = df['cat'].values

    X_train = list(vectors.loc[train_indices_subset].values)
    X_test = list(vectors.loc[test_indices_subset].values)
    y_train, y_test = y[train_indices_subset], y[test_indices_subset]

    clf = RandomForestClassifier(**BEST_PARAMS_RFC[pattern_name])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # save model
    pickle.dump(clf, open(MODEL_DIR + pattern_name + '_vector_model.sav', 'wb'))
    
    if info:
        performance_report(y_test, y_pred)
    
    if plot:
        display_confusion_matrix(y_test, y_pred, CLASSES, figsize=(5, 3))
    
    pred_proba_subset = clf.predict_proba(X_test)
    
    # set default probability to equal probability for each class
    pred_proba = np.tile([0.33, 0.33, 0.33], (len(test_indices), 1))
    pred_proba[list(test_indices_subset_dict.keys())] = pred_proba_subset
    
    return pred_proba

def best_RFECV(estimator: object,
               X: pd.DataFrame,
               y: pd.Series,
               features: List[str],
               random_state:int = 0,
               info: bool = False):
    """ Determines best subset of feature columns to use for classification.

    Uses recursive feature elimination and cross-validated selection to
    determine the best subset of feature columns to use for classification.

    Args:
        estimator: supervised learning estimator with a fit function that gives
                   info about feature importance.
        X: input feature columns.
        y: target column.
        features: list of feature column names.
        random_state: random state to set. Defaults to 0.
        info: prints debugging info. Defaults to False.

    Returns:
        List of subset of best feature columns.
    """

    selector = RFECV(estimator, verbose=0)
    selector = selector.fit(X, y)

    feature_rankings = dict(zip(features, selector.ranking_))
    best_features = [k for k, v in feature_rankings.items() if v == 1]

    # save best features
    pickle.dump(best_features, open(MODEL_DIR + 'best_features.txt', 'wb'))
    
    if info:
        print('N_FEATURES: %d' % selector.n_features_)
        print('BEST FEATURES:')
        print(', '.join(best_features))

    return best_features

def full_pred_proba(df: pd.DataFrame,
                    pattern_params: Dict[str, Dict[str, str]],
                    features: List[str],
                    train_indices: List[int],
                    test_indices: List[int],
                    info: bool = True,
                    plot: bool = False):
    """ Gets predicted proba from models trained on features and pattern models.

    Trains a separate model for each pattern in pattern_params, using vector
    embeddings on surrounding strings, and outputs the predicted probabilities
    for the test set. Also trains a model based on features specified in the
    features attribute, outputs the predicted probabilities for the test set.
    Outputs a DataFrame containing all the resulting predicted probabilities.

    Args:
        df: DataFrame to get training/testing data from.
        pattern_params: dictionary of regex patterns names and their parameters
                        (pattern name, n_chars, no_below, no_above)
        features: list of column names to use as inputs for feature model.
        train_indices: list of training indices.
        test_indices: list of test indices.
        info: prints debugging info. Defaults to True.
        plot: plots confusion matrix. Defaults to False.

    Returns:
        DataFrame containing predicted probabilities for each class for every
        sample, from various models.
    """

    nlp  = spacy.load('en_core_web_md')
    
    pps = []        # list of dataframes containing predicted proba
    out_cols = []   # names of output dataframe columns

    # vector model predicted proba
    for pattern_name, params in pattern_params.items():
        if info:
            print('collecting \'{}\' vector model predicted proba...'.format(
                pattern_name))
        
        pp = vector_pred_proba(
            df, pattern_name, params, nlp, train_indices, test_indices, 
            info, plot)
        pps.append(pp)
        out_cols.extend([pattern_name + '_hvpm',
                         pattern_name + '_nvpm',
                         pattern_name + '_svpm'])
    
    if info:
        print('collecting feature model predicted proba...')

    # feature model predicted proba
    X = df[features].values
    y = df['cat'].values

    # select best features
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, stratify=y, random_state=0)
    estimator = RandomForestClassifier(class_weight='balanced', random_state=0)
    best_features = best_RFECV(estimator, X_train, y_train, features, info)

    # store best features

    pp = feature_pred_proba(
        df, best_features, train_indices, test_indices, info, plot)
    pps.append(pp)
    out_cols.extend(['feature_hvpm', 'feature_nvpm', 'feature_svpm', 'cat'])

    if info:
        print('Collected all predicted proba.')
    
    # reshape output dataframe
    pps.append(y[test_indices].reshape(len(test_indices), -1))
    pred_probas = np.concatenate(pps, axis=1)
    
    return pd.DataFrame(pred_probas, columns=out_cols)

def get_train_test_splits(indices: List[int],
                          n_splits: int = 4) -> Tuple[List[int], List[int]]:
    """ Splits indices into a list of n_splits groups of train/test sets.

    Splits indices into n_splits equal length sections. For each split,
    n_split - 1 sections go to train indices, remaining goes to test indices.
    The resulting list contains all n_split combinations.

    Args:
        indices: list of indices.
        n_splits: number of splits. Defaults to 4.

    Raises:
        AttributeError: if number of splits is less than 2 or greater than 6.

    Returns:
        List of train indices, test indices pairs, has length n_splits.
    """
    if n_splits < 2 or n_splits > 6:
        raise AttributeError('number of splits should in a range of 2 to 6.')

    trains, tests = [], []

    np.random.shuffle(indices)
    splits = np.array_split(indices, n_splits)

    for s in splits:
        trains.append(sorted([x for x in indices if x not in s]))
        tests.append(sorted(s))
    
    return trains, tests

def train_stacked_model(original_df: pd.DataFrame,
                        pattern_params: Dict[str, Dict[str, str]],
                        features: List[str],
                        model_dir: str,
                        n_splits: int = 4,
                        info: bool = True,
                        plot: bool = False) -> None:
    """ Trains model using intermediate model predicted proba outputs as inputs.
    
    Trains and saves stacked model that uses the output predicted probabilities
    of intermediate models (feature-input, vector-input) as inputs.

    Args:
        original_df: DataFrame to get training/testing data from.
        pattern_params: dictionary of regex patterns names and their parameters
                        (pattern name, n_chars, no_below, no_above)
        features: list of column names to use as inputs for feature model.
        model_dir: output directory to save models in.
        n_splits: number of train/test splits. Defaults to 4.
        info: prints debugging info. Defaults to True.
        plot: plots confusion matrix. Defaults to False.
    """

    global MODEL_DIR
    MODEL_DIR = model_dir

    df = original_df.copy()
    indices = df.index.copy().values
    trains, tests = get_train_test_splits(indices, n_splits)
    
    agg_df = pd.DataFrame() # aggregated DataFrame
    for m in range(n_splits):
        print('----- TRAINING SUBSET %d -----' % (m + 1))
        pp_df = full_pred_proba(
            df, pattern_params, features, trains[m], tests[m], info, plot)
        agg_df = agg_df.append(pp_df)
        if info:
            print(agg_df.columns)
            print('aggregate df length: %d' % len(agg_df))
    
    inputs = agg_df.columns[:-1]
    X = agg_df[inputs].values
    y = agg_df['cat'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, stratify=y, random_state=0)

    # can use random forest classifier as an alternative to lgbm classifier
    # clf = RandomForestClassifier(**BEST_PARAMS_RFC['stacked'])

    clf = LGBMClassifier(**BEST_PARAMS_LGBM['stacked'])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    performance_report(y_test, y_pred)
    if plot:
        display_confusion_matrix(y_test, y_pred, CLASSES, figsize=(5, 3))

    # save model
    pickle.dump(clf, open(MODEL_DIR + 'stacked_model.sav', 'wb'))