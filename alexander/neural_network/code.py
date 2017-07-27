import sklearn.neural_network

class MLPClassifier(sklearn.neural_network.MLPClassifier):
    """Wrapper for scikit-learn MLPClassifier."""
    def fit(self, X, y):
    	y = y.values.reshape(y.shape[0],)
    	super(MLPClassifier, self).fit(X, y)

    def fit_predict(self, X, y):
    	self.fit(X, y)
    	return self.predict(X)