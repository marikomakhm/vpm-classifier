import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix

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
    # accuracy is the same as f1 micro
    print('F1 SCORE (MICRO)/ACCURACY: %.3f' % f1_score(
        y_test, y_pred, average='micro'))

def display_confusion_matrix(y_test, y_pred, class_names, figsize = (10,7), 
                             fontsize=12, savefig=None):
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df, annot=True, fmt="d")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0,
                                 ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45,
                                 ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if savefig:
        plt.savefig(savefig)
    plt.show()