# Modelling tool

The tool is designed to provide helper functions for a modelling project and cover the key steps of a project:
- data preprocessing and feature engineering
- data exploration and descriptive analysis
- model diagnostics
- data visualization


## Details about the package structure and classes

To see the documentation about classes and how to instantiate them, import the modules and run the following code:

```
from modelling_tool.data.generate import generate, buckets_data
from modelling_tool.preprocessing.featureEngineering import featureEngineering as fe

from modelling_tool.exploration.descriptiveStatistics import descriptiveStatistics as ds
from modelling_tool.exploration.visualization import visualization as vis

from modelling_tool.model_diagnostics.modelDiagnostics import modelDiagnostics as md


help(md)

print(generate.__doc__)
print(fe.__doc__)
print(ds.__doc__)
print(vis.__doc__)
print(md.__doc__)
```

## Code Examples
```
#DATA___________________________________________________________________________
train, valid, test = generate(200)
train_summary_example, valid_summary_example, test_summary_example = buckets_data()


#PREPROCESSING__________________________________________________________________
feature_engineering = fe(train)
print(train['var3'].value_counts())
train['var3_new'] = feature_engineering.reduce_categories('var3', threshold = 0.1)
train['var3_new'].value_counts()


#MODEL DIAGNOSTICS______________________________________________________________
#Classification model diagnostics class:
my_res = md(train['actual_cat'], train['predicted_cat'],
            test['actual_cat'], test['predicted_cat'],
            valid['actual_cat'], valid['predicted_cat'])
print(my_res.binary_bucket_summary('train', 5))


#EXPLORATION____________________________________________________________________

#descriptiveStatistics..........................................................
my_summary = ds(train)
my_summary.mean_by_category('predicted_cat', ['var1', 'var2'])
print(ds.mean_by_category(my_summary,'predicted_cat', ['var1', 'var2']))

#visualization..................................................................
my_vis = vis(train, 'actual_cat')

#Testing with default and manual titles:
my_vis.percent_obs_by_cat('var3')
my_vis.percent_obs_by_cat('var3',
                          x_label = 'Predictor', y_label = 'Target percentage',
                          title = 'Big title', subtitle = 'Small title')

#Visualization of model performance by buckets: lift (default)
my_vis.buckets_vis(train_summary = train_summary_example,
                   valid_summary = valid_summary_example,
                   test_summary = test_summary_example)

```

<img src="/modelling_tool/data/vis_example.png" alt="vis_example"
	title="buckets_vis output (model performance by bucket)" width="150" height="100" />

## More details on package functionality
If you want to see more examples, open and run the examples.py (located in modelling-tool folder).

## Package installation
Download .whl file from dist folder and run pip install 'C:/...your_path.../modelling_tool-0.0.16-py3-none-any.whl'

## Status
Package development is in progress.
The key blocks have been built and more functionality will be added over time.

## Contact
Created by Erin Balabukha
