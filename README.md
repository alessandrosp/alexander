# Alexander
**Alexander** is a Python wrapper that aims to make [**scikit-learn**](http://scikit-learn.org/) fully compatible with [**pandas**](http://pandas.pydata.org/pandas-docs/stable/index.html).

# Example
Alexander mirrors sklearn's API and structure. Most classes are either estimators (they have `fit`, `predict` and `fit_predict` methods) or transformers (they have a `fit`, `transform` and `fit_transform` methods). Transformers takes pd.DataFrame as input and returns pd.DataFrame as output, and are expected to be concatenated into pipelines.

```python

import alexander.datasets
import alexander.ensemble
import alexander.pipeline
import alexander.preprocessing
import pandas as pd

data, target = alexander.datasets.load_titanic(return_X_y=True)

rf_pipeline = alexander.pipeline.Pipeline([
    (['Pclass', 'Fare'], None),
    ('Age', alexander.preprocessing.MissingValuesFiller(strategy='median')),
    ('Sex', [alexander.preprocessing.LabelEncoder(),
             alexander.preprocessing.OneHotEncoder()]),
    ('Embarked', [alexander.preprocessing.MissingValuesFiller(strategy='most_frequent'),
                  alexander.preprocessing.LabelEncoder(),
                  alexander.preprocessing.OneHotEncoder()])
])
rf_pipeline.fit(data)  # Note how the pipeline is trained on the whole dataset

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, target, test_size=0.2))

rf_train = rf_pipeline.transform(X_train)
rf_clf = alexander.ensemble.RandomForestClassifier(n_estimators=100)
rf_clf.fit(rf_train, y_train)  # The model is ready to be used!

```

Alexander follows these principles:

- Alexander has the same interface as scikit-learn
- Transformers and estimators expect pd.DataFrame as input
- Every method or function that returns something, should return a pd.DataFrame

Guidelines for Alexander usage:

- Pretty much all data transformations should be done as part of an alexander.pipeline.Pipeline()
- Pipelines should be fitted to the whole dataset, before any split occures


# Status
These are the modules there are currently implemented in Alexander and their status:

| Module                  | Status           | Notes                                                |
|-------------------------|------------------|------------------------------------------------------|
| sklearn.datasets        | Work in progress | Added a new function to load the Titanic dataset     |
| sklearn.ensemble        | Fully wrapped    |                                                      |
| sklearn.model_selection | Work in progress |                                                      |
| sklearn.neural_network  | Fully wrapped    |                                                      |
| sklearn.pipeline        | Fully replaced   | Pipeline works differently in sklearn and Alexander  |
| sklearn.preprocessing   | Work in progress | This will have to be re-designed                     |

Please, refrain from using any module whose status is not either `Fully wrapped` or `Fully replaced`.

# Documentation

## sklearn.preprocessing

### Binarizer

Done. It behaves as in scikit-learn except for the fact both transform() and fit_transform() return pd.DataFrame(). The returned DataFrame has the same index and columns names as the input. Note: even though fit() does not really do anything here, the input is still checked and an error is returned if X is not a Pandas object.

### FunctionTransformer

Done. It behaves as in scikit-learn except for the fact that transform(), inverse_transform() and fit_transform() return pd.DataFrame(). The returned DataFrame has the same index and columns names as the input.


### Imputer

Done. It behaves as in scikit-learn except for the fact both transform() and fit_transform() return pd.DataFrame().

### KernelCenterer

Not implemented yet.

### LabelBinarizer

Not implemented yet.

### LabelEncoder

Not implemented yet.

### MultiLabelBinarizer

Not implemented yet.

### MaxAbsScaler

Done. It behaves as in scikit-learn except for the fact that transform(), inverse_transform() and fit_transform() return pd.DataFrame(). Note: Alexander does not allow inplace scaling or normalization, so parameter *copy* is de facto always set to True.

### MinMaxScaler

Done. It behaves as in scikit-learn except for the fact that transform(), inverse_transform() and fit_transform() return pd.DataFrame(). Note: Alexander does not allow inplace scaling or normalization, so parameter *copy* is de facto always set to True.

### Normalizer

Done. It behaves as in scikit-learn except for the fact that transform(), inverse_transform() and fit_transform() return pd.DataFrame(). Note: Alexander does not allow inplace scaling or normalization, so parameter *copy* is de facto always set to True.

### OneHotEncoder

Done. It behaves as in scikit-learn except for the fact both transform() and fit_transform() return pd.DataFrame(). Currently the columns names are just integers (starting from 0).

### PolynomialFeatures

Done. It behaves as in scikit-learn except for the fact both transform() and fit_transform() return pd.DataFrame(). Alexander uses get_feature_names() to get the right names for the DataFrame columns.

# Notes

- Usage of mock: if you look into the code from time to time you'll encounter non-orthodox usages of the mock library. This is because Alexander uses mock functionalities to repeat infinite loops. Imagine for example an Alexander class that implements a fit(), a transform() and a fit_transform() methods. The first two methods call their sklearn equivalents (via super()) while fit_transform call self.fit() and self.transform() one after the other. However, sklearn fit() often call self.fit_transform() behind the curtains so what happens is: self.fit() calls super().fit(), which calls self.fit_transform(), which has now been replaced by a different method whose first action is to simply call self.fit() et voilà we have an infinite loop. Mock solves all this.

# Extra
If you find this library useful, consider dropping me a message or starring this repo so that I know there are people interested in the project out there. (: