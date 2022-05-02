# -*- coding: utf-8 -*-

import numpy as np

class featureEngineering:

    '''
    Methods to clean the data and prepare for modelling.

    To instantiate the class, pass your dataframe as an argument.
    features = featureEngineering(df)
    '''

    def __init__(self, df):

        self.df = df


    def reduce_categories(self, var, threshold = None):

        '''
        Input: variable from a dataframe.
        Output: numpy ndarray that replaces variable category names with "Other", if they occur in less then 1% of cases.
                Otherwise, returns the same category name as in the original variable.
        '''

        if threshold == None:
            n_categories = self.df[var].nunique()
            cut_off = np.log(n_categories)/n_categories
        else:
            cut_off = threshold

        tmp = self.df[var].value_counts(normalize = True).reset_index()
        categories_list = tmp[tmp[var] > cut_off]['index'].tolist()
        new_var = np.where(np.isin(self.df[var], categories_list) == True, self.df[var], 'Other')

        return new_var

# TODO 
# Methods to add:
#       - calculate proportion of missing values ranked from max missing to the least missing
