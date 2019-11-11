# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 14:48:05 2019

@author: Erin Balabukha

"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter

from model_diagnostics import modelDiagnostics as md

class styling(object):

    color_dark_green = "#237522"
    color_light_green = "#35B233"
    color_pale_green = "#93DF91"
    color_light_gray = '#D3D3D3'
    color_gray = '#808080'
    selected_font_size = 10

    def plot_style(self, fig, ax, title, subtitle, x_label, y_label):

        #Background style:
        #To do: write ax.spines in one line
        ax.grid(axis = 'both', linestyle='-', color = self.color_light_gray)
        ax.spines['right'].set_color(self.color_gray)
        ax.spines['left'].set_color(self.color_gray)
        ax.spines['top'].set_color(self.color_gray)
        ax.spines['bottom'].set_color(self.color_gray)
        #ax.spines['right'].set_visible(False)
        #ax.spines['top'].set_visible(False)

        fig.suptitle(title, fontsize = self.selected_font_size + 4, fontweight='bold')
        ax.set_title(subtitle, size = self.selected_font_size, fontstyle='italic')
        ax.set_xlabel(x_label, size = self.selected_font_size + 2)
        ax.set_ylabel(y_label, size = self.selected_font_size + 2)



class visualization(styling):

    '''
    Methods to to visualize the data
    '''


    def __init__(self, df, target):

        self.df = df
        self.target = target


    def percent_obs_by_cat(self, predictor, rotation = 0,
                           title = 'Percentage of positive obs. by category',
                           subtitle = '',
                           x_label = None, y_label = None):

        '''
        Target variable: dichotomous in numeric format (1 for 'yes' and 0 for 'no')

        Input: categorical variable from a dataframe (as a string)
        Output: plot that shows the percentage of instances by category
                Black dotted line shows the average percentage of instances
        '''

        #DATA PREPARATION
        tmp = pd.DataFrame(self.df.groupby([predictor])[self.target].agg({'count', 'sum'}))
        tmp.reset_index(inplace=True)
        tmp['prop'] = tmp['sum']/tmp['count']
        target_mean = self.df[self.target].mean()

        fig = plt.figure()
        ax = fig.add_subplot(111)

        #DATA PLOTTING
        ax.axhline(y = target_mean, linestyle = '--', color = '#708090', zorder=1) #zorder to send line back
        ax.bar(tmp[predictor], tmp['prop'], color = 'green', zorder=2)

        #Numeric values inside the bars
        for p, i in zip(ax.patches, tmp['sum']):
            left, bottom, width, height = p.get_bbox().bounds
            ax.annotate(str(i), xy=(left+width/2, bottom+height*0.9),
                        ha='center', va='center', size = 12, color = 'white')

        #STYLING AND FORMATTING
        # Default names for axes labels are variables names.
        if x_label == None:
            x_label = str(predictor)
        if y_label == None:
            y_label = str(self.target)

        plt.gca().yaxis.set_major_formatter(PercentFormatter(1)) #display y-axis in %
        plt.xticks(rotation = rotation)
        self.plot_style(fig, ax, title, subtitle, x_label, y_label)


    def mean_by_cat(self, predictor, rotation = 0,
                           title = 'Mean values for two categories',
                           subtitle = '',
                           x_label = None, y_label = None):

        '''
        Target variable: dichotomous in numeric format (1 for 'yes' and 0 for 'no')

        Input: continuous variable from a dataframe (as a string)
        Output: plot that shows the mean value for each category
        '''

        #DATA PREPARATION
        tmp = self.df.groupby([self.target])[predictor].agg({'mean'}).reset_index()

        #DATA PLOTTING
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(tmp[self.target].astype(str), tmp['mean'], color = 'green', zorder=2)

        #Numeric values inside the bars
        for p, i in zip(ax.patches, round(tmp['mean'], 2)):
            left, bottom, width, height = p.get_bbox().bounds
            ax.annotate(str(i), xy=(left+width/2, bottom+height*0.9),
                        ha='center', va='center', size = 12, color = 'white')

        #STYLING AND FORMATTING
        # Default names for axes labels are variables names.
        if x_label == None:
            x_label = str(predictor)
        if y_label == None:
            y_label = str(self.target)
        self.plot_style(fig, ax, title, subtitle, x_label, y_label)
        plt.xticks(rotation = rotation)



    def act_vs_pred(self, predictor, rotation = 0,
                           title = 'Mean actual and predicted values by category',
                           subtitle = '',
                           x_label = None, y_label = None,
                           n_bins = 5,
                           bin_precision = 2):

        '''
        The goal of the function is to display how close the actual and predicted values are
        for a specific potential predictor across different categories or bins.
        The function can be used before and/or after the variable is added to the model.

        Target variable: continuous

        Input: potential predictor from a dataframe (can be either categorical or continuous)
        Output: two plots:
                Main plot shows the difference between actual and predicted values
                Smaller plot shows the distribution of observations for
                actual values by the selected categorical predictor.
        '''

        #DATA PREPARATION
        #If the variable is categorical, do nothing. If it is continuous, create bins:
        if self.df[predictor].dtype == 'O':
            self.df['predictor_transformed'] = self.df[predictor]
        else:
            self.df['predictor_transformed'] = pd.cut(self.df[predictor], precision = bin_precision, bins = n_bins)

        #Data for the main plot:
        tmp = self.df.groupby('predictor_transformed').agg(
        {'actual_cont': 'mean',
         'predicted_cont': 'mean'}
        ).reset_index()
        tmp['predictor_transformed'] = tmp['predictor_transformed'].astype(str)

        #Data for the additional plot:
        volume = self.df['predictor_transformed'].value_counts(normalize = True).reset_index()
        volume = volume.sort_values('index')
        volume['index'] = volume['index'].astype(str)

        #DATA PLOTTING
        fig = plt.figure()
        ax = fig.add_subplot(111)
        gs = gridspec.GridSpec(3,1)

        ax1 = ax
        ax1.set_position(gs[0:2].get_position(fig))
        ax1.plot(tmp['predictor_transformed'], tmp['predicted_cont'], color = 'red', zorder=2, label = "Predicted values")
        ax1.plot(tmp['predictor_transformed'], tmp['actual_cont'], color = 'green', zorder=2, label = "Actual values")
        ax1.set_subplotspec(gs[0:2])

        ax2 = fig.add_subplot(gs[2])
        ax2.bar(volume['index'], volume['predictor_transformed'], color = self.color_dark_green, zorder=2)

        #STYLING AND FORMATTING
        # Default names for axes labels are variables names.
        if x_label == None:
            x_label = str(predictor)
        if y_label == None:
            y_label = str(self.target)

        plt.xticks(rotation = rotation)

        self.plot_style(fig, ax, title, subtitle, x_label, y_label)

        fig_title = fig.suptitle(title, fontsize = self.selected_font_size + 4, fontweight='bold')
        fig_title.set_position([0.55, 1.05])

        ax1.set_xlabel(x_label, size = self.selected_font_size + 2)
        ax1.set_ylabel(y_label, size = self.selected_font_size + 2)
        ax1.legend(loc = 'best')
        #bbox_to_anchor=(1.1, .5)
        fig.tight_layout()
        #plt.save('test.jpg')
        plt.show()


    def buckets_vis(self,  train_summary,
                           valid_summary = None, test_summary = None,
                           performance_indicator = 'lift_res',
                           title = 'Model performance by bucket',
                           subtitle = None,
                           x_label = 'Buckets', y_label = None):

        '''
        Goal: visualize summary results produced by binary_bucket_summary method from modelDiagnostics class
        Target variable: dichotomous in numeric format (1 for 'yes' and 0 for 'no')

        Input: requires train, valid, and test summary tables produced by binary_bucket_summary method.
               Performance indicator can be selected from the summary tables (e.g., lift_res)
        Output: plot that shows model performance by bucket.
        '''

        #DATA PLOTTING
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(train_summary['bucket'].astype(str), train_summary[performance_indicator], color = self.color_dark_green, zorder=2, label = "Train")
        ax.plot(valid_summary['bucket'].astype(str), valid_summary[performance_indicator], color = self.color_light_green, zorder=2, label = "Validation")
        ax.plot(test_summary['bucket'].astype(str), test_summary[performance_indicator], color = self.color_gray, zorder=2, label = "Test")

        #STYLING AND FORMATTING
        # Default names for axes labels are variables names.
        if y_label == None:
            y_label = performance_indicator
        if subtitle == None:
            subtitle = f"Indicator: {performance_indicator}"

        ax.legend(loc = 'best')

        self.plot_style(fig, ax, title, subtitle, x_label, y_label)
        plt.show()
