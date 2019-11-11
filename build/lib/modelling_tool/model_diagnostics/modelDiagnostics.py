# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

class modelDiagnostics:
    '''
    Methods to explore the model output (actual vs. predicted).

    To instantiate the class, pass actual and predicted values of the train set
    as the arguments.
    You can also simultaneously pass the values for validation and test sets in
    addition to train to explore model performance across all datasets.

    If target variable is categorical, predicted value should be a probability
    score from 0 to 1.

    Example:
    my_res = modelDiagnostics(train['actual_cat'], train['predicted_cat'],
                              test['actual_cat'], test['predicted_cat'],
                              valid['actual_cat'], valid['predicted_cat'])
    '''

    def __init__(self, actual_train,        predicted_score_train,
                        actual_test = None,  predicted_score_test = None,
                        actual_val = None,   predicted_score_val = None):
        self.actual_train = actual_train
        self.predicted_score_train = predicted_score_train

        self.actual_test = actual_test
        self.predicted_score_test = predicted_score_test

        self.actual_val = actual_val
        self.predicted_score_val = predicted_score_val


    def binary_bucket_summary(self, set_type, n_buckets = 10):

        '''
        Target variable: dichotomous in numeric format (1 for 'yes' and 0 for 'no')

      - sorts actual and predicted score values on predicted score variable from
        highest to lowest probability.
      - groups the actual and predicted values in n buckets with equal sizes
      - provides summary statistics on ratio of actual to predicted, false positives, lift, etc.
        '''

        if set_type == 'train':
            actual = self.actual_train
            predicted_score = self.predicted_score_train
        elif set_type == 'test':
            actual = self.actual_test
            predicted_score = self.predicted_score_test
        elif set_type == 'valid':
            actual = self.actual_val
            predicted_score = self.predicted_score_val
        else:
            return 'Please enter a valid dataset name: "train", "test", or "valid"'


        #Load actual and predicted score values:
        result_ind = pd.DataFrame({'actual': actual, 'predicted': predicted_score})

        #Sort the dataframe:
        result_ind = result_ind.sort_values(['predicted'], ascending = False)

        #Add variable that would indicate a bucket:
            #Create a list from 1 to 100:
        x = list(range(1, n_buckets+1))

            #Calculate how many times each number should be repeated:
        y = [int(result_ind.shape[0]/n_buckets)] * (len(x))

            #Generate a list of numbers based on x and y:
        bucket = [item for item, count in zip(x, y) for i in range(count)]

            #If we cannot divide dataframe to 100 without a remainder, generate extra numbers for the last bucket
        bucket_last = [n_buckets] * (result_ind.shape[0] - len(bucket))

            #Add bucket variable to the dataframe:
        result_ind['bucket'] = bucket + bucket_last

        total_actual = sum(result_ind['actual'])
        rate = total_actual/result_ind.shape[0]

        #Calculate summary statistics:
        summary_act_pred = result_ind.groupby(['bucket']).agg({'actual':['sum'], 'predicted':['count']})
        summary_act_pred.columns = summary_act_pred.columns.get_level_values(0)
        summary_act_pred.reset_index(level=0, inplace=True)
        summary_act_pred['cum_act'] = summary_act_pred['actual'].cumsum()
        summary_act_pred['cum_pred'] = summary_act_pred['predicted'].cumsum()
        summary_act_pred['ratio_true'] = summary_act_pred['cum_pred']/total_actual * 100
        summary_act_pred['false_positive'] = summary_act_pred['predicted'] - summary_act_pred['actual']
        summary_act_pred['cum_false_positive'] = summary_act_pred['false_positive'].cumsum()
        summary_act_pred['ratio_false'] = summary_act_pred['cum_false_positive']/summary_act_pred['cum_pred'] * 100
        summary_act_pred['bucket_rate'] = summary_act_pred['actual']/summary_act_pred['predicted']
        summary_act_pred['cum_rate'] = summary_act_pred['cum_act']/summary_act_pred['cum_pred']
        summary_act_pred['lift_res'] = summary_act_pred['cum_rate']/rate

        return summary_act_pred
