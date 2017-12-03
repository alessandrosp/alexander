from .. import core

import collections
import warnings

import mock
import numpy as np
import pandas as pd
import sklearn.preprocessing

# TODO(): Make sure to call check_and_preprocess_input
# TODO(): Refactor/change LabelEncoder()
# TODO(): Most of these classes are identical - refactor?
# TODO(): Generate our own warnings

class Binarizer(sklearn.preprocessing.Binarizer):
	"""Wrapper for sklearn.preprocessing.Binarizer()."""

	def fit(self, X, y=None):
		"""Do nothing and return the estimator unchanged."""
		_ = core.check_and_preprocess_input(X)

	def fit_transform(self, X, y=None):
		"""Fit to data, then transform it."""
		self.fit(X, y)
		return self.transform(X)

	def transform(self, X):
		"""Binarize each element of X."""
		X = core.check_and_preprocess_input(X)
		X_index = X.index
		X_columns = X.columns
		result = super().transform(X)
		return pd.DataFrame(result, index=X_index, columns=X_columns)


class FunctionTransformer(sklearn.preprocessing.FunctionTransformer):
	"""Wrapper for sklearn.preprocessing.FunctionTransformer()."""

	def fit(self, X, y=None):
		"""Fit transformer by checking X."""
		X = core.check_and_preprocess_input(X)
		super().fit(X, y)

	def fit_transform(self, X, y=None):
		"""Fit to data, then transform it."""
		self.fit(X, y)
		return self.transform(X)

	def inverse_transform(self, X):
		"""Transform X using the inverse function."""
		X = core.check_and_preprocess_input(X)
		X_index = X.index
		X_columns = X.columns
		result = super().inverse_transform(X)
		return pd.DataFrame(result, index=X_index, columns=X_columns)

	def transform(self, X):
		"""Transform X using the forward function."""
		X = core.check_and_preprocess_input(X)
		X_index = X.index
		X_columns = X.columns
		result = super().transform(X)
		return pd.DataFrame(result, index=X_index, columns=X_columns)


class Imputer(sklearn.preprocessing.Imputer):
	"""Wrapper for sklearn.preprocessing.Imputer()."""

	def fit(self, X, y=None):
		"""Fit transformer by checking X."""
		X = core.check_and_preprocess_input(X)
		super().fit(X, y)

	def fit_transform(self, X, y=None):
		"""Fit to data, then transform it."""
		self.fit(X, y)
		return self.transform(X)

	def transform(self, X):
		"""Impute all missing values in X."""
		X = core.check_and_preprocess_input(X)
		X_index = X.index
		X_columns = X.columns
		result = super().transform(X)
		return pd.DataFrame(result, index=X_index, columns=X_columns)


class MaxAbsScaler(sklearn.preprocessing.MaxAbsScaler):
	"""Wrapper for sklearn.preprocessing.MaxAbsScaler()."""

	def fit(self, X, y=None):
		"""Compute the maximum absolute value to be used for later scaling."""
		X = core.check_and_preprocess_input(X)
		super().fit(X, y)

	def fit_transform(self, X, y=None):
		"""Fit to data, then transform it."""
		self.fit(X, y)
		return self.transform(X)

	def inverse_transform(self, X):
		"""Scale back the data to the original representation."""
		X = core.check_and_preprocess_input(X)
		X_index = X.index
		X_columns = X.columns
		result = super().inverse_transform(X)
		return pd.DataFrame(result, index=X_index, columns=X_columns)

	def transform(self, X):
		"""Scale the data."""
		X = core.check_and_preprocess_input(X)
		X_index = X.index
		X_columns = X.columns
		if not self.copy:
			message = 'Alexander does not allow inplace scaling or normalization'
			warnings.warn(message, UserWarning)
		result = super().transform(X)
		return pd.DataFrame(result, index=X_index, columns=X_columns)


class MinMaxScaler(sklearn.preprocessing.MinMaxScaler):
	"""Wrapper for sklearn.preprocessing.MinMaxScaler()."""

	def fit(self, X, y=None):
		"""Compute the minimum and maximum to be used for later scaling."""
		X = core.check_and_preprocess_input(X)
		super().fit(X, y)

	def fit_transform(self, X, y=None):
		"""Fit to data, then transform it."""
		self.fit(X, y)
		return self.transform(X)

	def inverse_transform(self, X):
		"""Undo the scaling of X according to feature_range."""
		X = core.check_and_preprocess_input(X)
		X_index = X.index
		X_columns = X.columns
		result = super().inverse_transform(X)
		return pd.DataFrame(result, index=X_index, columns=X_columns)

	def transform(self, X):
		"""Scaling features of X according to feature_range."""
		X = core.check_and_preprocess_input(X)
		X_index = X.index
		X_columns = X.columns
		if not self.copy:
			message = 'Alexander does not allow inplace scaling or normalization'
			warnings.warn(message, UserWarning)
		result = super().transform(X)
		return pd.DataFrame(result, index=X_index, columns=X_columns)


class Normalizer(sklearn.preprocessing.Normalizer):
	"""Wrapper for sklearn.preprocessing.Normalizer()."""

	def fit(self, X, y=None):
		"""Compute the minimum and maximum to be used for later scaling."""
		X = core.check_and_preprocess_input(X)
		super().fit(X, y)

	def fit_transform(self, X, y=None):
		"""Fit to data, then transform it."""
		self.fit(X, y)
		return self.transform(X)

	def transform(self, X):
		"""Scaling features of X according to feature_range."""
		X = core.check_and_preprocess_input(X)
		X_index = X.index
		X_columns = X.columns
		if not self.copy:
			message = 'Alexander does not allow inplace scaling or normalization'
			warnings.warn(message, UserWarning)
		result = super().transform(X)
		return pd.DataFrame(result, index=X_index, columns=X_columns)


class OneHotEncoder(sklearn.preprocessing.OneHotEncoder):
	"""Wrapper for sklearn.preprocessing.OneHotEncoder()."""

	def fit(self, X, y=None):
		"""Fit the imputer on X."""
		X = core.check_and_preprocess_input(X)
		with mock.patch.object(self, 'fit_transform',
			                   side_effect=super().fit_transform):
			super().fit(X, y)
		
	def fit_transform(self, X, y=None):
		"""Fit OneHotEncoder to X, then transform X."""
		self.fit(X, y)
		return self.transform(X)

	def transform(self, X):
		"""Transform X using one-hot encoding."""
		X = core.check_and_preprocess_input(X)
		result = super().transform(X)
		return pd.DataFrame(result)

class PolynomialFeatures(sklearn.preprocessing.PolynomialFeatures):
	"""Wrapper for sklearn.preprocessing.PolynomialFeatures()."""

	def fit(self, X, y=None):
		"""Compute number of output features."""
		X = core.check_and_preprocess_input(X)
		super().fit(X, y)

	def fit_transform(self, X, y=None):
		"""Fit to data, then transform it."""
		self.fit(X, y)
		return self.transform(X)

	def transform(self, X):
		"""Transform data to polynomial features."""
		X = core.check_and_preprocess_input(X)
		X_index = X.index
		X_columns = self.get_feature_names(X.columns)
		result = super().transform(X)
		return pd.DataFrame(result, index=X_index, columns=X_columns)


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


# class FeaturesEncoder(LabelEncoder):
#     """Encode features with value between 0 and n_classes-1.

#     Exactly the same as LabelEncoder. Original sklearn LabelEncoder was supposed
#     to only be used on labels (hence the name), while Alexander version of it
#     can be used on any DataFrame, regardless of whether it contains the features
#     or the labels. Intuitevely enough, it's reccomended to use LabelEncoder on
#     labels and FeaturesEncoder on features.
#     """
#     pass


# class OneHotEncoder(sklearn.preprocessing.OneHotEncoder):
#     """Encode categorical values using one-hot encoding."""

#     def fit(self, X, y=None):
#         """Fit to some data and save the columns names."""
#         cX = core.check_and_preprocess_input(X)
#         columns_names = []
#         # TODO(): Check whether original columns names exists in DataFrame
#         for column in cX.columns:
#             for value in list(cX[column].unique()):
#                 columns_names.append(str(value))
#         self.columns_names = columns_names
#         with mock.patch.object(self, 'fit_transform',
#             side_effect=super().fit_transform):
#             super().fit(cX)

#     def transform(self, X):
#         """Transform X using the information acquired during fit()."""
#         cX = core.check_and_preprocess_input(X)
#         transformed = super().transform(cX)
#         new_columns = ['is_' + value for value in self.columns_names]
#         if self.sparse:
#             results = pd.SparseDataFrame(
#                 transformed, columns=new_columns, index=cX.index
#             )
#         else:
#             results = pd.DataFrame(
#                 transformed, columns=new_columns, index=cX.index
#             )
#         return results

#     def fit_transform(self, X):
#         """Fit to X, then transform X."""
#         self.fit(X)
#         return self.transform(X)


# class OneHotEncoderBackup(sklearn.preprocessing.OneHotEncoder):
#     """Encode categorical values using one-hot encoding.

#     Same as sklearn OneHotEncoder but (a) expects a DataFrame or Series, (b)
#     returns a DataFrame and (c) retains all the information of the original
#     DataFrame, like indexes. The new column names are inferred from the
#     encodings attribute of the pd.DataFrame.

#     Parameters
#     ----------
#     None

#     Attributes
#     ----------
#     transformers : dict
#         A dict containing the transformers used on each column, where the keys in
#         the dictionary are the name of the columns of the df used to fit the
#         Alexander encoder. Empty before fit().
#     """

#     def __init__(self):
#         self.transformers = {}

#     def fit(self, X, y=None):
#         """For each column in pd.DataFrame an encoder is fitted.

#         Parameters
#         ----------
#         X : Pandas DataFrame, shape [n_samples, n_feature]
#             The pd.DataFrame containing the input to process.

#         Returns
#         -------
#         self
#         """
#         cX = core.check_and_preprocess_input(X)
#         for column in cX.columns:
#             self.transformers[column] = sklearn.preprocessing.OneHotEncoder(
#                 sparse=False)
#             self.transformers[column].fit(pd.DataFrame(cX[column]))

#     def transform(self, X):
#         """Given a DataFrame it returns a DataFrame with transformed features.

#         Parameters
#         ----------
#         X : pd.DataFrame, shape [n_samples, n_feature]
#             The pd.DataFrame containing the input to process.

#         Returns
#         -------
#         result : pd.DataFrame
#             The DataFrame cointaining the one-hot encoded values.
#         """
#         cX = core.check_and_preprocess_input(X)
#         new_dfs = []
#         # TODO(): as per now, if pd.DataFrame doesn't have attribute encodings
#         #         this will raise an error; this shouldn't happen
#         for column in cX.columns:
#             one_hot_encoded = self.transformers[column].transform(pd.DataFrame(cX[column]))
#             new_columns = ['is_'+value for value in X.encodings[column]]
#             new_df = pd.DataFrame(one_hot_encoded,
#                                   index=cX.index,
#                                   columns=new_columns)
#             new_dfs.append(new_df)

#         result = pd.concat(new_dfs, axis=1)
#         return result

#     def fit_transform(self, X):
#         """Fit to X, then transform X.

#         Equivalent to self.fit(X).transform(X), but more convenient and more
#         efficient. See fit for the parameters, transform for the return value.
#         """
#         self.fit(X)
#         result = self.transform(X)
#         return result


# class MissingValuesFiller(object):
#     """Replace missing values with othe values.

#     Alternative to sklearn Imputer. It (a) expects a DataFrame or Series, (b)
#     returns a DataFrame and (c) retains all the information of the original
#     DataFrame, like indexes and column names.

#     Parameters
#     ----------
#     missing_values : None or any type (default = None)
#         The placeholder for the missing value. By default, None and NaN are
#         expected to be signifying missing values but this may not be the case.
#     strategy : string (default = 'most_frequent')
#         The strategy used to replace the missing value. Contrary to Imputer, the
#         default strategy is most_frequent as this won't raise any error for
#         String fields. Other accepted values are:

#         - 'mean', to replace missing values with the mean
#         - 'median', to replace missing values with the median
#         - 'value', to replace missing values with a specific value; when this
#             strategy is chosen, then replacing_value is expected
#     replacing_value : any type (default = 0)
#         When the strategy is 'value', then this is the value used to replace
#         missing values in the pd.DataFrame

#     Attributes
#     ----------
#     replacing_values : dict
#         A dictionary containing the values used for replacement for each column.
#         Column names are used as dict keys. It's empty before .fit(). 
#     """

#     def __init__(self, missing_values=None, strategy='most_frequent', replacing_value=0):
#         self.missing_values = missing_values 
#         self.strategy = strategy
#         self.replacing_value = replacing_value
#         self.replacing_values = {}

#     # TODO(): None != NaN!

#     def fit(self, X, y=None):
#         """For each column in X, the correct value to replace NaN is found.

#         Parameters
#         ----------
#         X : Pandas DataFrame, shape [n_samples, n_feature]
#             The pd.DataFrame containing the input to process.

#         Returns
#         -------
#         self
#         """
#         cX = core.check_and_preprocess_input(X)
#         for column in cX.columns:
#             if self.missing_values:
#                 no_missing_values = [value for value in cX[column].values 
#                                      if value != self.missing_values]
#             else:
#                 no_missing_values = [value for value in cX[column].values 
#                                      if not pd.isnull(value)]
#             if self.strategy == 'most_frequent':
#                 most_common = (collections
#                                .Counter(no_missing_values)
#                                .most_common(1)[0][0])
#                 self.replacing_values[column] = most_common
                
#             elif self.strategy == 'mean':
#                 mean = np.nanmean(no_missing_values)
#                 self.replacing_values[column] = mean

#             elif self.strategy == 'median':
#                 median = np.nanmedian(no_missing_values)
#                 self.replacing_values[column] = median

#             elif self.strategy == 'value':
#                 self.replacing_values[column] = self.replacing_value

#             else:
#                 pass  # TODO(): Raise an error


#     def transform(self, X):
#         """Given a DataFrame it returns a DataFrame without missing values.

#         Parameters
#         ----------
#         X : pd.DataFrame, shape [n_samples, n_feature]
#             The pd.DataFrame containing the input to process.

#         Returns
#         -------
#         cX : pd.DataFrame, shape same as X
#             Same as X but the missing values (by default NaN and None) are now
#             being replaced by some other values, as specified during fitting.
#         """
#         if isinstance(X, pd.DataFrame):
#             cX = X.copy()
#         elif isinstance(X, pd.Series):
#             cX = pd.DataFrame(X)
#         for column in cX.columns:
#             if self.missing_values:
#                 bool_mask = (cX[column] == self.missing_values)
#             else:
#                 bool_mask = pd.isnull(cX[column])
#             cX[column][bool_mask] = self.replacing_values[column]
#         return cX

#     def fit_transform(self, X, y=None):
#         """Fit to X, then transform X.

#         Equivalent to self.fit(X).transform(X), but more convenient and more
#         efficient. See fit for the parameters, transform for the return value.
#         """
#         self.fit(X, y)
#         result = self.transform(X)
#         return result
