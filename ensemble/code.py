import pandas as pd
import sklearn.ensemble


class RandomForestClassifier(sklearn.ensemble.RandomForestClassifier):
    """Wrapper for scikit-learn RandomForestClassifier.

    Parameters
    ----------
    None

    Attributes
    ----------
    estimator : sklearn.ensemble.RandomForestClassifier()
        The actual classifier.
    features : list
        A list containing all the features of the pd.DataFrame passed as
        argument to the .fit() method.
    feature_importances : [tuple]
        A list of tuples, each containing the name of the feature and its
        importance for the classifier
    """

    def __init__(self):
        self.estimator = sklearn.ensemble.RandomForestClassifier()
        self.features = []
        self.feature_importances = []

    def fit(self, X, y, sample_weight=None):
        """Fit self.estimator on X (features) and y (targets).

        Parameters
        ----------
        X : pd.DataFrame, shape [n_samples, n_feature]
            The pd.DataFrame containing the features.
        y : pd.DataFrame, shape [n_samples, 1] or [n_samples, n_outputs]
            The pd.DataFrame containg the targets.

        Returns
        -------
        self
        """
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                n_rows = y.values.shape[0]
                y = y.values.reshape(n_rows,)
        self.estimator.fit(X, y, sample_weight)
        self.features = X.columns
        self.feature_importances = list(
            zip(self.features, self.estimator.feature_importances_))

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : pd.DataFrame, shape [n_samples, n_feature]
            The pd.DataFrame containing the features.

        Returns
        -------
        The predicted classes as pd.DataFrame.
        """
        result = pd.DataFrame(
            self.estimator.predict(X), index=X.index, columns=['prediction'])
        return result