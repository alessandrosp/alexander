import pandas as pd
import sklearn.ensemble

class RandomForestClassifier(sklearn.ensemble.RandomForestClassifier):
    """Wrapper for scikit-learn RandomForestClassifier.

    Parameters
    ----------
    Same as sklearn.ensemble.RandomForestClassifier()

    Attributes
    ----------
    features : list
        A list containing all the features of the pd.DataFrame passed as
        argument to the .fit() method.
    feature_importances : [tuple]
        A list of tuples, each containing the name of the feature and its
        importance for the classifier
    """

    def __init__(self,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_split=1e-7,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(RandomForestClassifier, self).__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)
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
        super(RandomForestClassifier, self).fit(X, y, sample_weight)
        self.features = X.columns
        self.feature_importances = list(
            zip(self.features, self.feature_importances_))

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
            super(RandomForestClassifier, self).predict(X),
            index=X.index,
            columns=['prediction'])
        return result