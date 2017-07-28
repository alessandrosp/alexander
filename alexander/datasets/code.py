import os

import pandas as pd

def load_titanic(return_X_y=False):
    """Load and return the Titanic dataset (classification).

    The Titanic dataset is a classic classification dataset.

    More info here: https://www.kaggle.com/c/titanic

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a dictionary.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : dict
        Dictionary, the interesting attributes are:
        - 'data', the data to learn
        - 'target', the classification labels
        - 'target_names', the meaning of the labels 
        - 'feature_names', the meaning of the features
        - 'DESCR', the full description of the dataset
    (data, target) : tuple, if ``return_X_y`` is True
    """
    module_path = os.path.dirname(__file__)
    dataset_path = os.path.join(module_path, 'titanic-train.csv')
    df = pd.read_csv(dataset_path)
    df = df.set_index('PassengerId')

    raw_data = df.loc[:, df.columns != 'Survived']
    target = df.loc[:, ['Survived']]
    target_names = ['Dead', 'Survived']

    if return_X_y:
        return (raw_data, target)
    else:
        data = {}
        data['data'] = raw_data
        data['target'] = target
        data['target_names'] = target_names
        data['feature_names'] = None
        data['DESCR'] = ('Full description available here: '
                         'https://www.kaggle.com/c/titanic')
        return data