import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_palette(sns.color_palette("pastel"))

from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix

import re
import string

CLASSES = ['hvpm', 'nvpm', 'svpm']

COPYRIGHT_PATTERN = [
    r'(copyright 19\d{2})|(copyright 20\d{2})',
    r'(\(c\) copyright|copyright \(c\))',
    r'(\(c\) 19\d{2})|(\(c\) 20\d{2})|(\(c\)19\d{2})|(\(c\)20\d{2})']

def plot_bar(data, xlabel, ylabel, title, figsize=(6,4)):
    plt.figure(figsize=figsize)
    data.plot(kind='bar')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, size=20)
    plt.show()

def performance_report(y_test, y_pred):
    vpm_test = [1 if x in ['svpm', 'hvpm'] else 0 for x in y_test]
    vpm_pred = [1 if x in ['svpm', 'hvpm'] else 0 for x in y_pred]
    
    fn = sum(vpm_test & np.invert(vpm_pred))

    print('TOTAL SAMPLES: %d' % len(y_test))
    print('VPM FN: %d' % fn)
    print('VPM recall (%% actual pos. identif. correctly): %.3f' % recall_score(
        vpm_test, vpm_pred))
    print('VPM accuracy (%% predictions correct): %.3f' % accuracy_score(
        vpm_test, vpm_pred))

    print('F1 SCORE (MACRO): %.3f' % f1_score(y_test, y_pred, average='macro'))
    print('F1 SCORE (MICRO)/ACCURACY: %.3f' % f1_score(y_test, y_pred, average='micro'))

def display_confusion_matrix(y_test, y_pred, figsize=(10, 7), fontsize=12):
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
    df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df, annot=True, fmt="d")
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def get_footer_features(original_df: pd.DataFrame) -> pd.DataFrame:
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

    df.drop(columns=['html_stripped', 'split_idx'], inplace=True)

    return df