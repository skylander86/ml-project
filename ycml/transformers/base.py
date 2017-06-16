import logging

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import Timer

__all__ = ['PureTransformer', 'identity']


logger = logging.getLogger(__name__)


# Helper class. A transformer that only does transformation and does not need to fit any internal parameters.
class PureTransformer(BaseEstimator, TransformerMixin):
    FORCE_NP_1D_ARRAY = False

    def __init__(self, nparray=True, **kwargs):
        super(PureTransformer, self).__init__(**kwargs)

        self.nparray = nparray
    #end def

    def fit(self, X, y=None, **fit_params): return self

    def transform(self, X, **kwargs):
        timer = Timer()
        transformed = self._transform(X, **kwargs)
        if self.nparray:
            if self.FORCE_NP_1D_ARRAY: transformed = np.array(transformed, dtype=np.object)
            else: transformed = np.array(transformed)
            if transformed.ndim == 1:
                transformed = transformed.reshape(transformed.shape[0], 1)
        #end if
        logger.debug('Done <{}> transformation{}.'.format(type(self).__name__, timer))

        return transformed
    #end def

    def _transform(self, X, y=None):
        return [self.transform_one(row) for row in X]
    #end def

    def transform_one(self, x):
        raise NotImplementedError('transform_one method needs to be implemented.')
#end class


def identity(x): return x
