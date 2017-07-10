import collections

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
    transformers : dict
        A dict containing the transformers used on each column, where the keys in
        the dictionary are the name of the columns of the df used to fit the
        Alexander encoder. Empty before fit().
    encodings : dict, e.g., {'Sex': ['female', 'male']}
        The encoding used to map each value to an int. The keys in the dict are
        the name of the columns of the df used to fit the Alexander encoder.
        Empty before fit().
    """

    def __init__(self):
        self.transformers = {}
        self.encodings = {}
    
    def fit(self, X, y=None):
        """For each column in pd.DataFrame an encoder is fitted.

        Parameters
        ----------
        X : Pandas DataFrame, shape [n_samples, n_feature]
            The pd.DataFrame containing the input to process.

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            cX = X.copy()
        elif isinstance(X, pd.Series):
            cX = pd.DataFrame(X)
        for column in cX.columns:
            self.transformers[column] = sklearn.preprocessing.LabelEncoder()
            self.transformers[column].fit(cX[column])
            self.encodings[column] = self.transformers[column].classes_.tolist()
         
    def transform(self, X):
        """Given a DataFrame it returns a DataFrame with transformed features.

        Parameters
        ----------
        X : pd.DataFrame, shape [n_samples, n_feature]
            The pd.DataFrame containing the input to process.

        Returns
        -------
        cX : pd.DataFrame, shape same as X
            Same as X but with processed data.
        """
        if isinstance(X, pd.DataFrame):
            cX = X.copy()
        elif isinstance(X, pd.Series):
            cX = pd.DataFrame(X)
        for column in cX.columns:
            cX[column] = self.transformers[column].transform(cX[column])
        cX.encodings = self.encodings
        return cX

    def fit_transform(self, X):
        """Fit to X, then transform X.

        Equivalent to self.fit(X).transform(X), but more convenient and more
        efficient. See fit for the parameters, transform for the return value.
        """
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
    """Encode categorical values using one-hot encoding.

    Same as sklearn OneHotEncoder but (a) expects a DataFrame or Series, (b)
    returns a DataFrame and (c) retains all the information of the original
    DataFrame, like indexes. The new column names are inferred from the
    encodings attribute of the pd.DataFrame.

    Parameters
    ----------
    None

    Attributes
    ----------
    transformers : dict
        A dict containing the transformers used on each column, where the keys in
        the dictionary are the name of the columns of the df used to fit the
        Alexander encoder. Empty before fit().
    """

    def __init__(self):
        self.transformers = {}

    def fit(self, X, y=None):
        """For each column in pd.DataFrame an encoder is fitted.

        Parameters
        ----------
        X : Pandas DataFrame, shape [n_samples, n_feature]
            The pd.DataFrame containing the input to process.

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            cX = X.copy()
        elif isinstance(X, pd.Series):
            cX = pd.DataFrame(X)
        for column in cX.columns:
            self.transformers[column] = sklearn.preprocessing.OneHotEncoder(
                sparse=False)
            self.transformers[column].fit(pd.DataFrame(cX[column]))

    def transform(self, X):
        """Given a DataFrame it returns a DataFrame with transformed features.

        Parameters
        ----------
        X : pd.DataFrame, shape [n_samples, n_feature]
            The pd.DataFrame containing the input to process.

        Returns
        -------
        result : pd.DataFrame
            The DataFrame cointaining the one-hot encoded values.
        """
        if isinstance(X, pd.DataFrame):
            cX = X.copy()
        elif isinstance(X, pd.Series):
            cX = pd.DataFrame(X)
        new_dfs = []
        # TODO(): as per now, if pd.DataFrame doesn't have attribute encodings
        #         this will raise an error; this shouldn't happen
        for column in cX.columns:
            one_hot_encoded = self.transformers[column].transform(pd.DataFrame(cX[column]))
            new_columns = ['is_'+value for value in X.encodings[column]]
            new_df = pd.DataFrame(one_hot_encoded,
                                  index=cX.index,
                                  columns=new_columns)
            new_dfs.append(new_df)

        result = pd.concat(new_dfs, axis=1)
        return result

    def fit_transform(self, X):
        """Fit to X, then transform X.

        Equivalent to self.fit(X).transform(X), but more convenient and more
        efficient. See fit for the parameters, transform for the return value.
        """
        self.fit(X)
        result = self.transform(X)
        return result


class MissingValuesFiller(object):
    """Replace missing values with othe values.

    Alternative to sklearn Imputer. It (a) expects a DataFrame or Series, (b)
    returns a DataFrame and (c) retains all the information of the original
    DataFrame, like indexes and column names.

    Parameters
    ----------
    missing_values : None or any type (default = None)
        The placeholder for the missing value. By default, None and NaN are
        expected to be signifying missing values but this may not be the case.
    strategy : string (default = 'most_frequent')
        The strategy used to replace the missing value. Contrary to Imputer, the
        default strategy is most_frequent as this won't raise any error for
        String fields. Other accepted values are:

        - 'mean', to replace missing values with the mean
        - 'median', to replace missing values with the median
        - 'value', to replace missing values with a specific value; when this
            strategy is chosen, then replacing_value is expected
    replacing_value : any type (default = 0)
        When the strategy is 'value', then this is the value used to replace
        missing values in the pd.DataFrame

    Attributes
    ----------
    replacing_values : dict
        A dictionary containing the values used for replacement for each column.
        Column names are used as dict keys. It's empty before .fit(). 
    """

    def __init__(self, missing_values=None, strategy='most_frequent', replacing_value=0):
        self.missing_values = missing_values 
        self.strategy = strategy
        self.replacing_value = replacing_value
        self.replacing_values = {}

    # TODO(): None != NaN!

    def fit(self, X, y=None):
        """For each column in X, the correct value to replace NaN is found.

        Parameters
        ----------
        X : Pandas DataFrame, shape [n_samples, n_feature]
            The pd.DataFrame containing the input to process.

        Returns
        -------
        self
        """
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
        """Given a DataFrame it returns a DataFrame without missing values.

        Parameters
        ----------
        X : pd.DataFrame, shape [n_samples, n_feature]
            The pd.DataFrame containing the input to process.

        Returns
        -------
        cX : pd.DataFrame, shape same as X
            Same as X but the missing values (by default NaN and None) are now
            being replaced by some other values, as specified during fitting.
        """
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
        """Fit to X, then transform X.

        Equivalent to self.fit(X).transform(X), but more convenient and more
        efficient. See fit for the parameters, transform for the return value.
        """
        self.fit(X, y)
        result = self.transform(X)
        return result
