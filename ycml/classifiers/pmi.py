__all__ = ['PMIFeatureSelector']

from .labels import MultiLabelsClassifier

import numpy as np


class PMIFeatureSelector(MultiLabelsClassifier):
    def __init__(self, algorithm='pmi', X_prior=0.001, Y_prior=0.1, XY_prior=0.001, **kwargs):
        super(MultiLabelsClassifier, self).__init__(**kwargs)

        self.algorithm = algorithm
        self.X_prior = X_prior
        self.Y_prior = Y_prior
        self.XY_prior = XY_prior
    #end def

    def fit_binarized(self, X_featurized, Y_binarized, **kwargs):
        log_Nx = np.log(np.asarray((X_featurized.sum(axis=0) + self.X_prior)).ravel())
        log_Ny = np.log(Y_binarized.sum(axis=0) + self.Y_prior)
        Y_indexes = [set(np.flatnonzero(Y_binarized[:, j] > 0).tolist()) for j in range(Y_binarized.shape[1])]
        pmi_XY = np.zeros((X_featurized.shape[1], Y_binarized.shape[1]))
        for i in range(X_featurized.shape[1]):
            X_indexes = set(np.flatnonzero(np.asarray((X_featurized[:, i] > 0).todense()).ravel().flatten()).tolist())
            for j in range(Y_binarized.shape[1]):
                log_Nxy = len(X_indexes & Y_indexes[j]) + self.XY_prior
                pmi_XY[i, j] = log_Nxy - log_Nx[i] - log_Ny[j]
            #end for
        #end for

        if self.algorithm == 'npmi':
            log_N = np.log(X_featurized.shape[0])
            for i in range(X_featurized.shape[1]):
                pmi_XY[i, :] /= log_Nx[i] - log_N

            assert (pmi_XY >= -1.0).all()
            assert (pmi_XY <= 1.0).all()
        #end if

        self.pmi_ = pmi_XY
        self.scores_ = np.max(pmi_XY, axis=1)
    #end def

    def feature_select(self, X_featurized, Y_labels, **kwargs):
        self.fit(X_featurized, Y_labels, **kwargs)

        return self.scores_
    #end def

    def transform(self, X, **kwargs):
        return X
#end class
