import numpy as np
import pandas as pd
import re
import string

PATENT_NUMBER_PATTERN = re.compile('\b(\d{1}[,\s]\d{3}[,\s]\d{3})\b')
DATE_URL_PATTERN = re.compile('(\d{1,2}[./-]\d{1,2}[./-]\d{4}|'
                              '\d{4}[./-]\d{1,2}[./-]\d{1,2}|'
                              '20\d{2}[0-3]\d{1}[0-3]\d{1}|'
                              '[0-3]\d{1}[0-3]\d{1}20\d{2})')
DATE_HTML_PATTERN = re.compile('(\d{1,2}[./-]\d{1,2}[./-]\d{4}|'
                               '\d{4}[./-]\d{1,2}[./-]\d{1,2})')
COPYRIGHT_PATTERN = [
    r'(copyright 19\d{2})|(copyright 20\d{2})',
    r'(\(c\) copyright|copyright \(c\))',
    r'(\(c\) 19\d{2})|(\(c\) 20\d{2})|(\(c\)19\d{2})|(\(c\)20\d{2})']

US_CODE_TERMS = ['35 u.s.c. § 287', '35 usc § 287', '35 u.s.c. §287', 
                 '35 usc §287', '35 u.s.c.', 'section 287']
PRODUCT_TERMS = ['product manual', 'user guide', 'user manual',
                 'product specification', 'product details',
                 'product description']
NEWS_TERMS = ['news', 'blog', 'article']

def get_url_features(original_df: pd.DataFrame) -> pd.DataFrame:
    """ Uses URL column to add new columns related to URL properties.

    Args:
        original_df: original DataFrame.

    Returns:
        Original DataFrame with additional URL-related columns.
    """

    df = original_df.copy()

    df['url'] = df.url.apply(lambda s: s.lower())

    # "patent" in URL
    df.loc[:, 'url_patent'] = df.url.apply(lambda s: 'patent' in s)

    # "product" in URL
    df.loc[:, 'url_product'] = df.url.apply(lambda s: 'product' in s)

    # number of characters in URL
    df['len_url'] = df['url'].apply(lambda x: len(x))

    # news terms in URL
    df.loc[:, 'url_news'] = df.url.apply(
        lambda s: any(term in s for term in NEWS_TERMS))

    # date format detected in URL
    df.loc[:, 'url_date'] = df.url.apply(
        lambda s: len(re.findall(DATE_URL_PATTERN, s)) > 0)

    return df

def add_html_column(original_df: pd.DataFrame) -> pd.DataFrame:
    """ Adds column containing HTML for each sample.

    Args:
        original_df: original DataFrame containing filename column.

    Returns:
        Original DataFrame with additional HTML column.
    """

    df = original_df.copy()

    df['html'] = ''

    for i, r in df.iterrows():
        with open(r.filename, 'r', encoding='utf-8', errors='ignore') as f:
            df.loc[i, 'html'] = f.read()
    
    df['html'] = df.html.apply(lambda r: str(r).lower())
    
    print('Successfully added HTML column')
    return df

def get_html_features(original_df: pd.DataFrame) -> pd.DataFrame:
    """ Uses HTML column to add new columns related to HTML properties.

    Args:
        original_df: original DataFrame.

    Returns:
        Original DataFrame with additional HTML-related columns.
    """
    df = original_df.copy()

    if 'html' not in df.columns:
        raise AttributeError('HTML column missing from DataFrame.')
            
    # number of times "patent" appears in HTML
    df.loc[:, 'html_n_patent'] = df.html.apply(lambda s: s.count('patent'))

    # number of times US code 287 appears in HTML
    df['n_us_code'] = 0

    # number of times product-specific terms appear in HTML
    df['n_product'] = 0

    for i, r in df.iterrows():
        n_us_code = sum([r.html.lower().count(code) for code in US_CODE_TERMS])
        df.loc[i, 'n_us_code'] = n_us_code

        n_product = sum([r.html.lower().count(s) for s in PRODUCT_TERMS])
        df.loc[i, 'n_product'] = n_product

    # number of times the patent number pattern appears in HTML
    df.loc[:, 'n_patent_strings'] = df.html.apply(
        lambda s: len(re.findall(PATENT_NUMBER_PATTERN, s)))

    # number of date-like substrings in HTML
    df.loc[:, 'n_dates'] = df.html.apply(
        lambda s: len(re.findall(DATE_HTML_PATTERN, s)))

    # date format detected in HTML
    df['contains_date'] = df['n_dates'] > 0
    return df

def get_footer_features(original_df: pd.DataFrame) -> pd.DataFrame:
    """ Uses HTML column to detect footers and add columns related to footers.

    Args:
        original_df: original DataFrame.

    Returns:
        Original DataFrame with additional footer-related columns.
    """

    df = original_df.copy()

    # to adhere to copyright pattern
    df['html'] = df['html'].apply(lambda s: s.replace('©', '(c)'))

    non_alnum_allowed = set(string.punctuation).union(set(['\n', ' ']))
    df['html_stripped'] = df.html.apply(
        lambda s: ''.join([c for c in s if c.isalnum() or c in non_alnum_allowed]))

    def split_index_end(s):
        indices = []    
        for k in COPYRIGHT_PATTERN:
            for m in re.finditer(k, s):
                indices.append(m.start(0))
        return len(s) - indices[0] if len(indices) else 0

    df['split_idx'] = df.html_stripped.apply(
        lambda r: r[-1500:] if len(str(r)) > 1500 else r)\
            .apply(lambda r: str(r).replace('\n', ' ').replace('*', ''))\
                .apply(lambda r: split_index_end(str(r)))

    # adds footer column    
    df['footer'] = ''
    df.loc[df['split_idx'] > 0, 'footer'] = df[df['split_idx'] > 0].apply(
        lambda r: r['html_stripped'][-r['split_idx']:], axis=1)

    # True if "patent" in footer
    df['footer_patent'] = df.footer.str.contains('patent')

    df.drop(columns=['html_stripped', 'split_idx', 'footer'], inplace=True)

    return df

def get_general_features(original_df: pd.DataFrame,
                         save: bool = True,
                         save_as: str = None) -> pd.DataFrame:
    """ Generates general URL and HTML-specific features for DataFrame.

    Args:
        original_df: DataFrame containing filename, url, category columns
        save: Specifies if generated DataFrame should be saved. If True,
              saves as save_as filename. Defaults to True.
        save_as: Filename to save DataFrame as. Defaults to None.

    Raises:
        AttributeError: If save is True, but no save_as filename is specified.

    Returns:
        DataFrame containing additional URL and HTML-specific features.
    """

    df = original_df.copy()    
    df = get_url_features(df)
    df = add_html_column(df)
    df = get_html_features(df)
    df = get_footer_features(df)

    if save:
        if not save_as:
            raise AttributeError('Save directory is not specified.')
        df.to_csv(save_as,
                  encoding='utf-8',
                  header=True,
                  index=False)
        print('Saved general features DataFrame.')
    
    return df