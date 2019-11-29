# -*- coding: utf-8 -*-

import pandas as pd

class descriptiveStatistics:

    '''
    Methods to explore data and look at summary statistics.

    To instantiate the class, pass your dataframe as an argument.

    Example:
    desc_analysis = descriptiveStatistics(df)

    '''

    def __init__(self, df):
        self.df = df


    def mean_by_category(self, target, list_continuous_vars):
        return pd.DataFrame(self.df.groupby([target])[list_continuous_vars].agg({'mean'}))

# =============================================================================
# Methods to add:
#       - summary table with number of obs. in each category and mean value (e.g., to see when the mean value difference is big if the data is balanced) + missing
#         similar to my mean.by.var function in R
#
# =============================================================================
