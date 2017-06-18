# Alexander
**Alexander** is a Python wrapper that aims to make [**scikit-learn**](http://scikit-learn.org/) fully compatible with [**pandas**](http://pandas.pydata.org/pandas-docs/stable/index.html).

# Example
Alexander mirrors sklearn's API and structure. Most classes are either estimators (they have `fit`, `predict` and `fit_predict` methods) or transformers (they have a `fit`, `transform` and `fit_transform` methods). Transformers takes pd.DataFrame as input and returns pd.DataFrame as output, and are expected to be concatenated into a pipelines.

```python

import alexander.ensemble
import alexander.pipeline
import alexander.preprocessing
import pandas as pd

df = pd.read_csv('train.csv')  # We load the Titanic dataset
df = df.set_index('PassengerId')

rf_pipeline = alexander.pipeline.Pipeline([
    (['Pclass', 'Fare'], None),
    ('Age', alexander.preprocessing.MissingValuesFiller(strategy='median')),
    ('Sex', [alexander.preprocessing.LabelEncoder(),
             alexander.preprocessing.OneHotEncoder()]),
    ('Embarked', [alexander.preprocessing.MissingValuesFiller(strategy='most_frequent'),
                  alexander.preprocessing.LabelEncoder(),
                  alexander.preprocessing.OneHotEncoder()])
])
rf_pipeline.fit(df)  # Note how the pipeline is trained on the whole dataset

X_all = df.loc[:, df.columns != 'Survived']
y_all = df.loc[:, ['Survived']]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_all, y_all, test_size=0.2))

rf_train = rf_pipeline.transform(X_train)
rf_clf = alexander.ensemble.RandomForestClassifier(max_features=None)
rf_clf.fit(rf_train, y_train)  # The model is ready to be used!
print(RandomForestClassifier.feature_importances)

```

Alexander follows these principles:

- Both trasnformers and estimators expect pd.DataFrame as structure for the data
- Pretty much all data transformations should be done as part of an alexander.pipeline.Pipeline()
- Where possible, Alexander tries to perfectly mirror scikit-learn's API
- Alexander's transformers have a self.transformers attribute where the actual transformers are stored (normally, one for column; before `fit` this attribute is empty)
- Alexander's estimators have a self.estimator where the actual estimator is stored

# What have been done so far
The following scikit-learn classes have been either wrapped or replaced:

- FeaturesEncoder, newly created, same as LabelEncoder
- Imputer, replaced by MissingValuesFiller
- LabelEncoder, wrapped
- MissingValuesFiller, newly created, it replaces Imputer
- OneHotEncoder, wrapped
- Pipeline, replaced by class with the same name
- RandomForestClassifier, wrapped