# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:55:56 2019

@author: Erin Balabukha
"""

import pandas as pd
import numpy as np
from sklearn import model_selection


def generate(nb_obs):

    # Sample dataset with two actual and predicted variables
    n = nb_obs

    data = {'var1': np.random.normal(20, 3, n),
            'var2': np.random.normal(96, 2, n),
            'var3': np.random.choice(['blue', 'green', 'yellow', 'white'],
                                      size = n, p = [0.70, 0.2, 0.07, 0.03]),
            'actual_cat': np.random.choice([1, 0], size = n, p = [0.8, 0.2]),
            'actual_cont': np.random.normal(20, 3, n),

            'predicted_cat': np.random.choice([1, 0], size = n, p = [0.8, 0.2]),
            'predicted_cont': np.random.normal(20, 3, n)
            }

    df = pd.DataFrame(data=data)

    train, test = model_selection.train_test_split(df, test_size=0.3, random_state=123)
    valid = df.sample(int(n/4))


    return train, valid, test
