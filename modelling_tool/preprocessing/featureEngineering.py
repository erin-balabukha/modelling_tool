# -*- coding: utf-8 -*-
"""

To instantiate the class, pass your dataframe as an argument.
features = featureEngineering(df)

"""

import numpy as np

class featureEngineering:

    '''
    Methods to clean the data and prepare for modelling
    '''

    def __init__(self, df):
        self.df = df


    def reduce_categories(self, var, percent = None):

        '''
        Input: variable from a dataframe.
        Output: numpy ndarray that replaces variable category names with "Other", if they occur in less then 1% of cases.
                Otherwise, returns the same category name as in the original variable.
        '''

        if percent == None:
            n_categories = self.df[var].nunique()
            cut_off = np.log(n_categories)/n_categories
        else:
            cut_off = percent

        tmp = self.df[var].value_counts(normalize = True).reset_index()
        categories_list = tmp[tmp[var] > cut_off]['index'].tolist()
        new_var = np.where(np.isin(self.df[var], categories_list) == True, self.df[var], 'Other')

        return new_var

# =============================================================================
# Methods to add:
#       - select variables with the number of levels between range, more than...
#       - table with proportion of missing values ranked from max missing to least
#
# =============================================================================

# change_var_f <- function(df, x) {
#   x_new <- paste0(x, "_Change")
#   x_end <- paste0(x, "_Last_Mod")
#   df[x_new] <-  ifelse(df[["n_mod_by_pol"]] == 0, 0,
#                                   ifelse(df[[x]] == df[[x_end]], 0, 1))
#   return(df)
# }
# for (i in last.mod.vars[-(1:3)]) {
#   policies <- change_var_f(policies, i)
#   print(names(policies[length(policies)]))
# }
