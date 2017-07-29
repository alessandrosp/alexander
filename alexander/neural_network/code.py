from ..core import AlexanderBaseEstimator
import sklearn.neural_network


class BernoulliRBM(AlexanderBaseEstimator, sklearn.neural_network.BernoulliRBM):
    """Wrapper for sklearn.neural_network.BernoulliRBM()."""
    pass


class MLPClassifier(AlexanderBaseEstimator, neural_network.MLPClassifier):
    """Wrapper for sklearn.neural_network.MLPClassifier()."""
    pass


class MLPRegressor(AlexanderBaseEstimator, neural_network.MLPRegressor):
    """Wrapper for sklearn.neural_network.MLPRegressor()."""
    pass

