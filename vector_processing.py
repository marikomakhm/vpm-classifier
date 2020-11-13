import numpy as np
import pandas as pd
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils import sparse2full

import spacy.lang.en as en

import pickle

from typing import List, Tuple

STOP_WORDS = set(stopwords.words('english'))

def get_pattern_substrings(init_text: str,
                           pattern: str,
                           n_chars: int = 140) -> List[str]:
    """ Returns a list of substrings containing regex pattern in text.

    Identifies pattern occurrences in text, and returns a list of substrings for
    each occurrence, with the surrounding complete words that fit in n_chars
    characters around the pattern.

    Args:
        init_text: text to search for pattern substrings in.
        pattern: regex pattern to find in text.
        n_chars: number of characters to select around pattern. Defaults to 140.

    Returns:
        List of substrings that are matched on regex pattern in text.
    """
    # in case text is nan
    if type(init_text) == float:
        return []

    substrings = []
    text = init_text.replace('\n', ' ')

    for m in re.finditer(pattern, text):
        start = m.start(0)
        s = text[max(start-n_chars, 0):min(start+n_chars, len(text))]
        # selects only complete "words" around identified pattern
        split = s.split(' ', 1)
        if len(split) > 1:
            s = split[1]
            s = s[:s.rfind(' ')]
            substrings.append(s)
    
    return substrings

def parse_texts(texts: List[str]) -> List[str]:
    """ Parses specific patent/patent number related terms to standardize terms.

    Args:
        texts: list of texts to parse.

    Returns:
        List of parsed texts.
    """
    parsed = []
    for t in texts:
        new_t = re.sub(r'\b(\d{1}[,\s]\d{3}[,\s]\d{3})\b', 'patent number', t)
        new_t = new_t.replace('patent nos', 'patent number')
        new_t = new_t.replace('patent no', 'patent number')
        parsed.append(new_t)
    return parsed

def stem_texts(texts: List[str]) ->  List[str]:
    """ Stems and removes stop words from each text in list.

    Args:
        texts: list of texts.

    Returns:
        List of stemmed texts with stop words removed.
    """
    stemmer = PorterStemmer()
    stemmed = []
    for t in texts:
        tokens = word_tokenize(t)
        stemmed_text = [stemmer.stem(w) for w in tokens if w not in STOP_WORDS]
        stemmed.append(stemmed_text)
    return stemmed

def valid_token(token: str) -> bool:
    """ Determines if a token is valid (keeps only words).

    Args:
        token: token to validate.

    Returns:
        True if token is valid, otherwise False.
    """
    return token.is_alpha and \
        not (token.is_space or token.is_punct or token.like_num)

def lemmatize_doc(doc: List[str]) -> List[str]:
    """ Lemmatizes each valid token in doc.

    Args:
        doc: list of tokens.

    Returns:
        List of lemmatized valid tokens.
    """
    return [token.lemma_ for token in doc if valid_token(token)]

def search(series: pd.Series,
           pattern: str,
           n_chars: int = 140,
           info: bool = True) -> pd.Series:
    """ Generates Series of list of strings containing pattern found in input.

    Searches input Series containing HTML to extract all occurrences of pattern
    and surrounding words within a character limit (specified by n_chars), and
    then parses these strings. Returns a Series containing a list of parsed
    strings identified using the pattern.

    Args:
        series: series containing HTML.
        pattern: regex pattern to search for.
        n_chars: number of characters to select around pattern. Defaults to 140.
        info: Prints debugging info. Defaults to True.

    Returns:
        Series containing a list of parsed strings identified using the pattern.
    """

    if info:
        print('getting {} substrings...'.format(pattern))

    pattern_series = series.apply(
        lambda x: get_pattern_substrings(x, pattern, n_chars))

    if info:
        print('collected {} substrings.'.format(pattern))
        
    pattern_series_parsed = pattern_series.apply(parse_texts)
    pattern_series_parsed = pattern_series_parsed.apply(stem_texts)
    
    return pattern_series_parsed

def get_docs_emb_train(docs: List[str],
                 nlp: en.English,
                 pattern_name: str,
                 model_dir: str,
                 no_below: int = 5,
                 no_above: int = 0.5) -> List[str]:
    """ Gets list of GloVe vectors for each string in docs using TF-IDF model.

    Computes list of GloVe vectors for each string in list of docs using a
    TF-IDF model and saves both the dictionary generated from docs and TF-IDF
    model for future reference.

    Note: follows a similar approach as described in the following tutorial
    http://dsgeek.com/2018/02/19/tfidf_vectors.html

    Args:
        docs: list of strings to get embeddings for.
        nlp: pretrained spacy model to use for embedding vectors.
        pattern_name: regex pattern name that generates list of docs (only to
                      be used for naming purposes of dictionary/model).
        model_dir: model directory to save dictionary and TF-IDF model in.
        no_below: keep tokens that are contained in at least no_below docs.
                  Defaults to 5.
        no_above: keep tokens which are contained in no more than no_above docs.
                  Defaults to 0.5.

    Returns:
        List of GloVe embedded vectors. Has shape (len(docs), 300)
    """

    # creates dictionary of terms in docs
    docs_dict = Dictionary(docs)

    # filters dictionary terms
    docs_dict.filter_extremes(
        no_below=no_below, no_above=no_above, keep_tokens=['patent', 'number'])
    docs_dict.compactify()
    corpus = [docs_dict.doc2bow(doc) for doc in docs]

    # creates TF-IDF model based on docs corpus
    model = TfidfModel(corpus, id2word=docs_dict)
    docs_tfidf = model[corpus]

    # extracts vector representation for each document from bag of words
    docs_vecs = np.vstack([sparse2full(c, len(docs_dict)) for c in docs_tfidf])

    # extracts nlp vector using pretrained model for each term in dictionary
    tfidf_emb_vecs = np.vstack([nlp(docs_dict[i]).vector for i in range(len(docs_dict))])

    # gets nlp vector embedding of each doc
    docs_emb = np.dot(docs_vecs, tfidf_emb_vecs)

    # save dictionary
    docs_dict.save(model_dir + pattern_name + '_dict.dict')

    # save model
    pickle.dump(model, open(model_dir + pattern_name + '_tfidf_model.sav', 'wb'))
    
    return docs_emb

def get_docs_emb_trained(docs: List[str],
                         nlp: en.English,
                         pattern_name: str,
                         model_dir: str) -> List[str]:
    """ Gets list of GloVe vectors for each string in docs using TF-IDF model.

    Loads previously stored doc dictionary and TF-IDF model to determine BOW
    representation of corpus.

    Args:
        docs: list of strings to get embeddings for.
        nlp: pretrained spacy model to use for embedding vectors.
        pattern_name: regex pattern name that generates list of docs (only to
                      be used for naming purposes of dictionary/model).
        model_dir: model directory to load dictionary and TF-IDF model from.

    Returns:
        List of GloVe embedded vectors. Has shape (len(docs), 300)
    """

    loaded_dict = Dictionary.load(model_dir + pattern_name + '_dict.dict')
    model = pickle.load(
        open(model_dir + pattern_name + '_tfidf_model.sav', 'rb'))
    
    corpus = [loaded_dict.doc2bow(doc) for doc in docs]

    if corpus:
        docs_tfidf = model[corpus]

        # extracts vector representation for each document from bag of words
        docs_vecs = np.vstack(
            [sparse2full(c, len(loaded_dict)) for c in docs_tfidf])

        # extracts nlp vector using pretrained model for each term in dictionary
        tfidf_emb_vecs = np.vstack(
            [nlp(loaded_dict[i]).vector for i in range(len(loaded_dict))])

        # gets nlp vector embedding of each doc
        docs_emb = np.dot(docs_vecs, tfidf_emb_vecs)

        return docs_emb

    # if corpus is empty according to dictionary
    return []

def get_docs(series: pd.DataFrame,
             nlp: en.English,
             pattern_name: str,
             pattern: str,
             n_chars:int = 140,
             info: bool = True) -> Tuple[List[str], List[int]]:
    """ Identifies regex pattern in HTML and returns vector representation.

    Identifies surrounding strings in each row of HTML of the input series
    containing regex pattern and calculates vector representation of the
    surrounding terms identified using GloVe embeddings of the surrounding
    terms.

    Args:
        series: series containing HTML.
        nlp: pretrained spacy model to use for embedding vectors.
        pattern_name: name of regex pattern (e.g. "patent_number").
        pattern: regex pattern to identify.
        n_chars: number of characters to select around pattern. Defaults to 140.
        no_below: keep tokens that are contained in at least no_below docs.
                  Defaults to 5.
        no_above: keep tokens which are contained in no more than no_above docs.
                  Defaults to 0.5.
        info: prints debugging info. Defaults to True.

    Returns:
        pd.Series of the same length as the original series containing the 
        vector embeddings of identified pattern and list of pids (page ids, 
        corresponding to rows in the series) that have non-null values (meaning
        that the pattern has been identified at least once in the sample).
    """

    # search for strings of length 2*n_chars containing pattern in input series
    pattern_series_parsed = search(
        series, pattern=pattern, n_chars=n_chars, info=info)

    df = pd.concat([pattern_series_parsed, pattern_series_parsed.str.len()], axis=1)
    df.columns = ['pattern_strs_parsed', 'len']
    non_null_indices = df[df.len > 0].index.values
    nun_null_pattern_series_parsed = df.loc[df.len > 0, 'pattern_strs_parsed']

    nun_null_pattern_series_parsed = nun_null_pattern_series_parsed.apply(
        lambda l: [e for subl in l for e in subl])
    nun_null_pattern_series_parsed = nun_null_pattern_series_parsed.apply(
        lambda x: ' '.join(x))
    non_null_lemmatized = nun_null_pattern_series_parsed.apply(
        lambda r: lemmatize_doc(nlp(r)))

    return non_null_lemmatized.values, non_null_indices

def get_pattern_vectors(series: pd.DataFrame,
                        nlp: en.English,
                        pattern_name: str,
                        pattern: str,
                        train: bool,
                        model_dir: str = '',
                        n_chars:int = 140,
                        no_below: int = 5,
                        no_above: int = 0.5,
                        info: bool = True) -> Tuple[pd.Series, List[int]]:
    """ Identifies regex pattern in HTML and returns vector representation.

    Identifies surrounding strings in each row of HTML of the input series
    containing regex pattern and calculates vector representation of the
    surrounding terms identified using GloVe embeddings of the surrounding
    terms.

    Args:
        series: series containing HTML.
        nlp: pretrained spacy model to use for embedding vectors.
        pattern_name: name of regex pattern (e.g. "patent_number").
        pattern: regex pattern to identify.
        train: if True, creates dictionary and TF-IDF model for pattern. If
               False, loads previously stored dictionary and TF-IDF model.
        model_dir: TF-IDF model and dictionary directory to save in/load from.
        n_chars: number of characters to select around pattern. Defaults to 140.
        no_below: keep tokens that are contained in at least no_below docs.
                  Defaults to 5.
        no_above: keep tokens which are contained in no more than no_above docs.
                  Defaults to 0.5.
        info: prints debugging info. Defaults to True.

    Returns:
        pd.Series of the same length as the original series containing the 
        vector embeddings of identified pattern and list of pids (page ids, 
        corresponding to rows in the series) that have non-null values (meaning
        that the pattern has been identified at least once in the sample).
    """

    docs, non_null_indices = get_docs(
        series, nlp, pattern_name, pattern, n_chars, info)
    
    if train:
        docs_emb = get_docs_emb_train(
            docs, nlp, pattern_name, model_dir, no_below, no_above)
    else:
        if not model_dir:
            raise AttributeError('no model directory specified.')

        docs_emb = get_docs_emb_trained(docs, nlp, pattern_name, model_dir)
    
    col_name = pattern_name + '_vector'

    docs_emb_series = pd.Series([[]] * len(series))
    docs_emb_series.loc[non_null_indices] = list(docs_emb)

    return docs_emb_series, non_null_indices