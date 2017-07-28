from ..core import AlexanderBaseEstimator
import sklearn.ensemble

class AdaBoostClassifier(AlexanderBaseEstimator, sklearn.ensemble.AdaBoostClassifier):
    """Wrapper for sklearn.ensemble.AdaBoostClassifier()."""
    pass


class AdaBoostRegressor(AlexanderBaseEstimator, sklearn.ensemble.AdaBoostRegressor):
    """Wrapper for sklearn.ensemble.AdaBoostRegressor()."""
    pass


class BaggingClassifier(AlexanderBaseEstimator, sklearn.ensemble.BaggingClassifier):
    """Wrapper for sklearn.ensemble.BaggingClassifier()."""
    pass


class BaggingRegressor(AlexanderBaseEstimator, sklearn.ensemble.BaggingRegressor):
    """Wrapper for sklearn.ensemble.BaggingRegressor()."""
    pass


class ExtraTreesClassifier(AlexanderBaseEstimator, sklearn.ensemble.ExtraTreesClassifier):
    """Wrapper for sklearn.ensemble.ExtraTreesClassifier()."""
    pass


class ExtraTreesRegressor(AlexanderBaseEstimator, sklearn.ensemble.ExtraTreesRegressor):
    """Wrapper for sklearn.ensemble.ExtraTreesRegressor()."""
    pass


class GradientBoostingClassifier(AlexanderBaseEstimator, sklearn.ensemble.GradientBoostingClassifier):
    """Wrapper for sklearn.ensemble.GradientBoostingClassifier()."""
    pass


class GradientBoostingRegressor(AlexanderBaseEstimator, sklearn.ensemble.GradientBoostingRegressor):
    """Wrapper for sklearn.ensemble.GradientBoostingRegressor()."""
    pass


class IsolationForest(AlexanderBaseEstimator, sklearn.ensemble.IsolationForest):
    """Wrapper for sklearn.ensemble.IsolationForest()."""
    pass


class RandomForestClassifier(AlexanderBaseEstimator, sklearn.ensemble.RandomForestClassifier):
    """Wrapper for sklearn.ensemble.RandomForestClassifier()."""
    pass


class RandomTreesEmbedding(AlexanderBaseEstimator, sklearn.ensemble.RandomTreesEmbedding):
    """Wrapper for sklearn.ensemble.RandomTreesEmbedding()."""
    pass


class RandomForestRegressor(AlexanderBaseEstimator, sklearn.ensemble.RandomForestRegressor):
    """Wrapper for sklearn.ensemble.RandomForestRegressor()."""
    pass


class VotingClassifier(AlexanderBaseEstimator, sklearn.ensemble.VotingClassifier):
    """Wrapper for sklearn.ensemble.VotingClassifier()."""
    pass
