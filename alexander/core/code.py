import pandas as pd


class InputIsNotPandas(Exception):
    """Raised when the input is not a DataFrame or Series."""


class AlexanderBaseEstimator():
    """Base class for Alexander's Estimator.
    """

    def fit(self, X, y, sample_weight=None):
        """Fit the model on some training set (X, y)."""
        if not (isinstance(X, pd.DataFrame)
                or isinstance(X, pd.Series)):
            raise InputIsNotPandas('Your X is not a Pandas object.')
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                n_rows = y.shape[0]
                y = y.values.reshape(n_rows,)
        else:
            if not isinstance(y, pd.Series):
                raise InputIsNotPandas('Your y is not a Pandas object')
        super(self.__class__, self).fit(X, y, sample_weight)
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
        if not (isinstance(X, pd.DataFrame)
                or isinstance(X, pd.Series)):
            raise InputIsNotPandas('Your X is not a Pandas object.')
        predictions = super(self.__class__, self).predict(X)
        if (len(predictions.shape) == 1 or predictions.shape[1] == 1):
            return pd.DataFrame(predictions, columns=['prediction'])
        # If more than one output
        else:
            columns_names = ['prediction_{}'.format(ix) 
                             for ix in range(predictions.shape[1])]
            return pd.DataFrame(predictions, columns=columns_names)