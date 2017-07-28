import inspect

import pandas as pd

class InputIsNotPandas(Exception):
    """Raised when the input is not a DataFrame or Series."""


class AlexanderBaseEstimator(object):
    """Base class for Alexander's Estimator.
    """

    def _check_input_is_pandas(self, X, y=pd.DataFrame()):
        """Makes sure input is a Pandas structure."""
        if not (isinstance(X, pd.DataFrame)
                or isinstance(X, pd.Series)):
            raise InputIsNotPandas('Your X is not a Pandas object.')
        if not y.empty:
            if not (isinstance(y, pd.DataFrame)
                    or isinstance(y, pd.Series)):
                raise InputIsNotPandas('Your y is not a Pandas object.')

    def _get_coparent_class(self):
        mro = inspect.getmro(self.__class__)
        coparent = mro[mro.index(AlexanderBaseEstimator) + 1]
        return coparent

    def fit(self, X, y, sample_weight=None):
        """Fit the model on some training set (X, y)."""
        self._check_input_is_pandas(X, y)
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                n_rows = y.shape[0]
                y = y.values.reshape(n_rows,)
        self._get_coparent_class().fit(self, X, y)
        try:
            self.features = X.columns  # if pd.DataFrame()
        except AttributeError:
            self.features = X.name  # if pd.Series()

    def fit_predict(self, X, y, sample_weight=None):
        """Fit the data, then predict class for X."""
        self.fit(X, y, sample_weight)
        return self.predict(X)

    def predict(self, X):
        """Output predictions for X."""
        self._check_input_is_pandas(X)
        predictions = self._get_coparent_class().predict(self, X)
        if (len(predictions.shape) == 1 or predictions.shape[1] == 1):
            return pd.DataFrame(predictions,
                columns=['prediction'], index=X.index)
        else:  # If more than one output
            columns_names = ['prediction_{}'.format(ix) 
                             for ix in range(predictions.shape[1])]
            return pd.DataFrame(predictions, columns=columns_names,
                index=X.index)