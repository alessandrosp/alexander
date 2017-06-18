import pandas as pd
import sklearn.ensemble


class RandomForestClassifier(sklearn.ensemble.RandomForestClassifier):
    """Wrapper for scikit-learn RandomForestClassifier.

    Parameters
    ----------
    Same as sklearn.ensemble.RandomForestClassifier()

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

    def __init__(self, *args, **kwargs):
        super(RandomForestClassifier, self).__init__(*args, **kwargs)
        self.estimator = sklearn.ensemble.RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_split=self.min_impurity_split,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
            class_weight=self.class_weight)
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