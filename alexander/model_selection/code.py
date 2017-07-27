import sklearn.model_selection

class GridSearchCV(sklearn.model_selection.GridSearchCV):
    """Wrapper for scikit-learn GridSearchCV."""

    def fit(self, X, y, sample_weight=None):
        """Fit model on X and y."""
        y = y.values.reshape(y.shape[0],)
        super(GridSearchCV, self).fit(X, y, sample_weight)