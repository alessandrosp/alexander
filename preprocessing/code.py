import copy
import collections
import math

import numpy as np
import pandas as pd
import sklearn.preprocessing



class LabelEncoder(sklearn.preprocessing.LabelEncoder):
    """Encode labels with value between 0 and n_classes-1.

    Same as sklearn LabelEncoder but (a) expects a DataFrame or Series, (b)
    returns a DataFrame and (c) retains all the information of the original
    DataFrame, like indexes and column names.

    Parameters
    ----------
    None

    Attributes
    ----------
    encoders : dict
        A dict containing the encoders used on each column, where the keys in
        the dictionary are the name of the columns of the df used to fit the
        Alexander encoder. Empty before fit().
    encodings : dict, e.g., {'Sex': ['female', 'male']}
        The encoding used to map each value to an int. The keys in the dict are
        the name of the columns of the df used to fit the Alexander encoder.
        Empty before fit().
    """

    def __init__(self):
        self.encoders = {}
        self.encodings = {}
    
    def fit(self, X, y=None):
        """For each column in pd.DataFrame an encoder is fitted."""
        if isinstance(X, pd.DataFrame):
            cX = copy.deepcopy(X)
        elif isinstance(X, pd.Series):
            cX = pd.DataFrame(X)
        for column in cX.columns:
            self.encoders[column] = sklearn.preprocessing.LabelEncoder()
            self.encoders[column].fit(cX[column])
            self.encodings[column] = self.encoders[column].classes_.tolist()
         
    def transform(self, X):
        """Given a pd.DataFrame it returns a df with transformed features."""
        if isinstance(X, pd.DataFrame):
            cX = copy.deepcopy(X)
        elif isinstance(X, pd.Series):
            cX = pd.DataFrame(X)
        for column in cX.columns:
            cX[column] = self.encoders[column].transform(cX[column])
        cX.encodings = self.encodings
        return cX

    def fit_transform(self, X):
        self.fit(X)
        result = self.transform(X)
        return result


class FeaturesEncoder(LabelEncoder):
    """Encode features with value between 0 and n_classes-1.

    Exactly the same as LabelEncoder. Original sklearn LabelEncoder was supposed
    to only be used on labels (hence the name), while Alexander version of it
    can be used on any DataFrame, regardless of whether it contains the features
    or the labels. Intuitevely enough, it's reccomended to use LabelEncoder on
    labels and FeaturesEncoder on features.
    """
    pass


class OneHotEncoder(sklearn.preprocessing.OneHotEncoder):
    """Same as sklearn OneHotEncoder"""

    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        """For each column in pd.DataFrame an encoder is fitted."""
        if isinstance(X, pd.DataFrame):
            cX = copy.deepcopy(X)
        elif isinstance(X, pd.Series):
            cX = pd.DataFrame(X)
        for column in cX.columns:
            self.encoders[column] = sklearn.preprocessing.OneHotEncoder(
                sparse=False)
            self.encoders[column].fit(pd.DataFrame(cX[column]))

    def transform(self, X):
        """Given a pd.DataFrame it returns a pd.DataFrame with transformed features."""
        if isinstance(X, pd.DataFrame):
            cX = X.copy()
        elif isinstance(X, pd.Series):
            cX = pd.DataFrame(X)
        new_dfs = []
        for column in cX.columns:
            one_hot_encoded = self.encoders[column].transform(pd.DataFrame(cX[column]))
            new_columns = ['is_'+value for value in X.encodings[column]]
            new_df = pd.DataFrame(one_hot_encoded,
                                  index=cX.index,
                                  columns=new_columns)
            new_dfs.append(new_df)

        result = pd.concat(new_dfs, axis=1)
        return result

    def fit_transform(self, X):
        self.fit(X)
        result = self.transform(X)
        return result


class MissingValuesFiller(object):
    """Alternative to sklearn Imputer."""

    def __init__(self, missing_values=None, strategy='most_frequent', replacing_value=0):
        self.missing_values = missing_values 
        self.strategy = strategy
        self.replacing_value = replacing_value
        self.replacing_values = {}

    # TODO(): None != NaN!

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            cX = X.copy()
        elif isinstance(X, pd.Series):
            cX = pd.DataFrame(X)
        for column in cX.columns:
            if self.missing_values:
                no_missing_values = [value for value in cX[column].values 
                                     if value != self.missing_values]
            else:
                no_missing_values = [value for value in cX[column].values 
                                     if not pd.isnull(value)]
            if self.strategy == 'most_frequent':
                most_common = (collections
                               .Counter(no_missing_values)
                               .most_common(1)[0][0])
                self.replacing_values[column] = most_common
                
            elif self.strategy == 'mean':
                mean = np.nanmean(no_missing_values)
                self.replacing_values[column] = mean

            elif self.strategy == 'median':
                median = np.nanmedian(no_missing_values)
                self.replacing_values[column] = median

            elif self.strategy == 'value':
                self.replacing_values[column] = self.replacing_value

            else:
                pass  # TODO(): Raise an error


    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            cX = X.copy()
        elif isinstance(X, pd.Series):
            cX = pd.DataFrame(X)
        for column in cX.columns:
            if self.missing_values:
                bool_mask = (cX[column] == self.missing_values)
            else:
                bool_mask = pd.isnull(cX[column])
            cX[column][bool_mask] = self.replacing_values[column]
        return cX

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        result = self.transform(X)
        return result
