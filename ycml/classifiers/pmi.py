__all__ = ['PMIFeatureSelector']

from .labels import MultiLabelsClassifier

import numpy as np


class PMIFeatureSelector(MultiLabelsClassifier):
    def __init__(self, algorithm='pmi', X_prior=0.001, Y_prior=0.1, XY_prior=1, **kwargs):
        super(MultiLabelsClassifier, self).__init__(**kwargs)

        self.algorithm = algorithm
        self.X_prior = X_prior
        self.Y_prior = Y_prior
        self.XY_prior = XY_prior
    #end def

    def fit_binarized(self, X_featurized, Y_binarized, **kwargs):
        log_P_x = np.asarray(X_featurized.sum(axis=0) + self.X_prior).ravel()
        log_P_x = np.log(log_P_x) - np.log(log_P_x.sum())

        log_P_y = Y_binarized.sum(axis=0) + self.Y_prior
        log_P_y = np.log(log_P_y) - np.log(log_P_y.sum())

        log_P_xy = np.zeros((X_featurized.shape[1], Y_binarized.shape[1]))
        for i in range(X_featurized.shape[1]):
            for j in range(Y_binarized.shape[1]):
                X_indexes = (X_featurized[:, i] > 0).todense().flatten()
                Y_indexes = Y_binarized[:, j] > 0
                log_P_xy[i, j] = np.logical_and(X_indexes, Y_indexes).sum() + self.XY_prior
            #end for
        #end for
        log_P_xy = np.log(log_P_xy) - np.log(log_P_xy.sum())

        pmi = log_P_xy
        for i in range(X_featurized.shape[1]):
            pmi[i, :] -= log_P_x[i]
        for j in range(Y_binarized.shape[1]):
            pmi[:, j] -= log_P_y[j]

        self.feature_scores_ = np.max(pmi, axis=1)
    #end def

    def feature_select(self, X_featurized, Y_labels):
        self.fit(X_featurized, Y_labels)

        return self.feature_scores_
    #end def
#end class
