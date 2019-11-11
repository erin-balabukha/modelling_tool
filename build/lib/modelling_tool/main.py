
import sys
import os
import pandas as pd
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter



# In order to include functions from other files, need to tell jupyter notebook where the path is
my_path = os.path.abspath(os.path.join('..'))
my_path = 'W:\Developpement\_PROJECTS\_STATISTICAL_DEVELOPMENT\SMART\z. Perso\Erin\My functions'
if my_path not in sys.path:
    sys.path.append(my_path)

from data.generate import generate
from exploration.descriptiveStatistics import descriptiveStatistics as ds
#from exploration.visualization import visualization as vis
from exploration.visualization import visualization as vis
from preprocessing.featureEngineering import featureEngineering as fe
from model_diagnostics.modelDiagnostics import modelDiagnostics

#DATA___________________________________________________________________________

train, valid, test = generate(200)

#PREPROCESSING__________________________________________________________________
feature_engineering = fe(train)
print(feature_engineering)
print(train['var3'].value_counts())
train['var3_new'] = feature_engineering.reduce_categories('var3', percent = 0.1)
train['var3_new'] = feature_engineering.reduce_categories('var3')
train['var3_new'].value_counts()



#VALIDATION_____________________________________________________________________
#Classification model diagnostics class:
my_res = modelDiagnostics(train['actual_cat'], train['predicted_cat'],
                          test['actual_cat'], test['predicted_cat'],
                          valid['actual_cat'], valid['predicted_cat'])
print(my_res)
print(my_res.binary_bucket_summary('train', 5))

#Testing errors:
print(my_res.binary_bucket_summary('wrong dataset', 5))


#EXPLORATION____________________________________________________________________

#descriptiveStatistics..........................................................
my_summary = ds(train)
my_summary.mean_by_category('predicted_cat', ['var1', 'var2'])
print(ds.mean_by_category(my_summary,'predicted_cat', ['var1', 'var2']))

#visualization..................................................................
my_vis = vis(train, 'actual_cat')
print(my_vis)

#Testing with default and manual titles:
my_vis.percent_obs_by_cat('var3')
my_vis.percent_obs_by_cat('var3',
                          x_label = 'Predictor', y_label = 'Target percentage',
                          title = 'Big title', subtitle = 'Small title')

#Testing with default and manual titles:
my_vis.mean_by_cat('var1')
my_vis.mean_by_cat('var1', title = 'Big title', subtitle = 'Small title')

#Testing for continuous variable:
my_vis.act_vs_pred('var1', n_bins = 3)

#Testing for categorical variable:
my_vis.act_vs_pred('var3', title = 'My title',
                   x_label = 'Predictor', y_label ='Target')

#Visualization of model performance by buckets: lift (default)
my_vis.buckets_vis(train_summary = train_summary,
                   valid_summary = valid_summary,
                   test_summary = test_summary)

#Visualization of model performance by buckets: false positives
my_vis.buckets_vis(train_summary = train_summary,
                   valid_summary = valid_summary,
                   test_summary = test_summary,
                   performance_indicator = 'false_positive')
