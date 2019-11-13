# -*- coding: utf-8 -*-
"""

This script serves to test the package and demostrate its functionality.

"""

from modelling_tool.data.generate import generate
from modelling_tool.preprocessing.featureEngineering import featureEngineering as fe

from modelling_tool.exploration.descriptiveStatistics import descriptiveStatistics as ds
from modelling_tool.exploration.visualization import visualization as vis

from modelling_tool.model_diagnostics.modelDiagnostics import modelDiagnostics as md


#PACKAGE INFORMATION___________________________________________________________

#To see the documentation about classes
#and how to instantiate them, use print(class.__doc__)
print(generate.__doc__)
print(fe.__doc__)
print(ds.__doc__)
print(vis.__doc__)
print(md.__doc__)

#To see the methods included in a class, use help(class)
help(md)


#DATA___________________________________________________________________________
train, valid, test = generate(200)


#PREPROCESSING__________________________________________________________________
feature_engineering = fe(train)
print(feature_engineering)
print(train['var3'].value_counts())
train['var3_new'] = feature_engineering.reduce_categories('var3', percent = 0.1)
train['var3_new'] = feature_engineering.reduce_categories('var3')
train['var3_new'].value_counts()


#MODEL DIAGNOSTICS______________________________________________________________
#Classification model diagnostics class:
my_res = md(train['actual_cat'], train['predicted_cat'],
            test['actual_cat'], test['predicted_cat'],
            valid['actual_cat'], valid['predicted_cat'])
print(my_res)
print(my_res.binary_bucket_summary('train', 5))

#Testing errors:
print(my_res.binary_bucket_summary('wrong dataset', 5))

train_summary = my_res.binary_bucket_summary('train')
valid_summary = my_res.binary_bucket_summary('valid')
test_summary = my_res.binary_bucket_summary('test')



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
my_vis.buckets_vis(train_summary_example = train_summary,
                   valid_summary_example = valid_summary,
                   test_summary_example = test_summary)

#Visualization of model performance by buckets: false positives
my_vis.buckets_vis(train_summary = train_summary,
                   valid_summary = valid_summary,
                   test_summary = test_summary,
                   performance_indicator = 'false_positive')
