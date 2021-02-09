from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from sklearn import metrics


# for decompensation, in-hospital mortality
def print_metrics_binary(y_true, predictions, verbose=False):
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1), labels=[0, 1])
    if verbose:
        print("confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    try:
        auroc = metrics.roc_auc_score(y_true, predictions[:, 1])
    except:
        auroc = np.nan

    (precisions, recalls, _) = metrics.precision_recall_curve(
        y_true, predictions[:, 1]
    )
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    if verbose:
        print("accuracy = {}".format(acc))
        print("precision class 0 = {}".format(prec0))
        print("precision class 1 = {}".format(prec1))
        print("recall class 0 = {}".format(rec0))
        print("recall class 1 = {}".format(rec1))
        print("AUC of ROC = {}".format(auroc))
        print("AUC of PRC = {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))

    return {
        "accuracy": acc,
        "precision0": prec0,
        "precision1": prec1,
        "recall0": rec0,
        "recall1": rec1,
        "auroc": auroc,
        "auprc": auprc,
        "minpse": minpse,
    }


# for phenotyping


def print_metrics_multilabel(y_true, predictions, verbose=1):
    y_true = np.array(y_true)
    predictions = np.array(predictions)

    auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions, average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions, average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions, average="weighted")

    if verbose:
        print("ROC AUC scores for labels:", auc_scores)
        print("ave_auc_micro = {}".format(ave_auc_micro))
        print("ave_auc_macro = {}".format(ave_auc_macro))
        print("ave_auc_weighted = {}".format(ave_auc_weighted))

    return {
        "auc_scores": auc_scores,
        "ave_auc_micro": ave_auc_micro,
        "ave_auc_macro": ave_auc_macro,
        "ave_auc_weighted": ave_auc_weighted,
    }