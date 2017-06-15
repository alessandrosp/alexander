# Alexander
Alexander is a Python wrapper that aims to make scikit-learn fully compatible with pandas.

# Example
Alexander mirrors sklearn's API and structure. Most classes are either estimators (they have a .fit(), .predict() and .fit_predict() methods) or transformers (they have a .fit(), .transform() and .fit_transform() methods). Transformers takes pd.DataFrame as input and returns pd.DataFrame, and are expected to be concatenated into a pipelines.

```python

import alexander.pipeline
import alexander.preprocessing
import pandas as pd
import sklearn.ensemble

df = pd.read_csv('train.csv')
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

# What have been done so far
The following sklearn classes have been either wrapped or replace:

- FeaturesEncoder, newly created, same as LabelEncoder
- Imputer, replaced by MissingValuesFiller
- LabelEncoder, wrapped
- MissingValuesFiller, newly created, it replaces Imputer
- OneHotEncoder, wrapped
- Pipeline, replaced by class with the same name
- RandomForestClassifier, wrapped