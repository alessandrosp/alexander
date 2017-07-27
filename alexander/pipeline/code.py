import pandas as pd

class NoneTransformer():
    """Convenience class for no transformation.

    It can be though as a identity transformer: transform is called on X and it
    returns X without any transformation. This is used as a replacement for None
    values in the pipeline (which are themselves used to signify no
    transformation is needed for the selected columns)."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """It does nothing. It returns self."""
        pass

    def transform(self, X):
        """Takes X as input and returns X as output."""
        return X

    def fit_transform(self, X):
        """Fit to X, then transform X. Same as self.fit(X).transform(X)."""
        return X


class Pipeline():
    """Used to concatenate a series of transformation.

    Alternative to sklearn Pipeline. It uses a syntax similar to sklearn-pandas
    DataFrameMapper(). Every step has two components: a selector and some
    actions. A selector is either a string of a list of string. The selector
    is used to select a subset of columns from a pd.DataFrame (the parameter of
    .fit() and .transform()). Actions is either a single transformer or a list
    of transformers (objects with a .fit_transform() method). Each step returns
    a pd.DataFrame. All the pd.DataFrames are then combinated into a single
    pd.DataFrame, that is returned by .transform() and .fit_transform().

    Note: Alexander's pipelines must accept a pd.DataFrame as input and must
        return a pd.DataFrame as ouput. Thus, out-of-the-box sklearn
        transformers won't work. Use Alexander's transformers or write your
        own wrappers.

    Note 2: Contrary to sklean's Pipeline, Alexander's Pipeline shouldn't
        have a classifier as last step. A pipeline is supposed to be a set
        of transformations. Model fitting should not happen as part of this
        process. This is especially true given that Pipeline.fit() should be
        called on the data before splitting them into train- and test-set.

    Parameters
    ----------
    steps : [Tuple], i.e. ('Sex', alexander.preprocessing.FeaturesEncoder())
        List of tuples. The tuple's first element is the selector, the tuple's
        second element is the actions. This will result in:

        action().fit_transform(X[selector])

        for each action in the actions list. Each action will use as input the
        output of the action before. 
    """

    def __init__(self, steps):
        self.steps = self._format_steps(steps)

    def _format_steps(self, steps):
        """It formats steps.

        It guarantees that steps[1] is a list rather than a single object. Also,
        it replaces every None with a NoneTransformer.
        """
        formatted_steps = []
        for step in steps:
            selector = step[0]
            actions = step[1]
            if not isinstance(actions, list):
                actions = [actions]
            new_actions = []
            for action in actions:
                if action is None:
                    new_actions.append(NoneTransformer())
                else:
                    new_actions.append(action)
            new_step = (selector, new_actions)
            formatted_steps.append(new_step)
        return formatted_steps

    def fit(self, X, y=None):
        """It calls .fit_transform() on each action but the last one.

        For each step, for each action, it calls .fit_transform() but for the
        last action of each step; the last action of each step is only fitted.

        Parameters
        ----------
        X : Pandas DataFrame, shape [n_samples, n_feature]
            The pd.DataFrame containing the input to process.

        Returns
        -------
        self
        """
        for step in self.steps:
            selector = step[0]
            cX = pd.DataFrame(X[selector])
            for transformer in step[1][0:-1]:  # All but the last one
                cX = transformer.fit_transform(cX)
            step[1][-1].fit(cX)

    def transform(self, X):
        """For each step, for each action, it calls .transform().

        Parameters
        ----------
        X : Pandas DataFrame, shape [n_samples, n_feature]
            The pd.DataFrame containing the input to process.

        Returns
        -------
        result : pd.DataFrame
            It returns the output of each step combined into a single DataFrame.
        """
        new_dfs = []
        for step in self.steps:
            selector = step[0]
            cX = pd.DataFrame(X[selector])
            # Here we have multiple transformers (list)
            if isinstance(step[1], list):
                for transformer in step[1]:
                    cX = transformer.transform(cX)
            # Here we have only one transformer (single object)
            else:
                cX = step[1].transform(cX)
            new_dfs.append(cX)
        result = pd.concat(new_dfs, axis=1)
        return result

    def fit_transform(self, X):
        """Fit to X, then transform X.

        Equivalent to self.fit(X).transform(X), but more convenient and more
        efficient. See fit for the parameters, transform for the return value.
        """
        self.fit(X)  # TODO(): This is inefficient
        result = self.transform(X)
        return result
