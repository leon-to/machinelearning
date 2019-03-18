# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:28:14 2019

@author: khoit
"""

from recorder import Recorder
import matplotlib.pyplot as plt

class Plotter(object):
    def __init__(self, recorder):
        self.rc = recorder
    
    # plot a single graph
    def plot( self, 
              df, #dataframe to plot
              ax=None, # axes this plot belong to
              title=None, xlabel=None, ylabel=None,
              txt=None, x=None, y=None  # text in plot
              ):
        if ax==None:
            plt.figure()
        
        plot = df.plot(ax=ax, title=title)
        
        if xlabel!=None:
            plot.set_xlabel(xlabel)
        if ylabel!=None:
            plot.set_ylabel(ylabel)
        
        if txt!=None:    
            plot.text(
                x, y, 
                txt, 
                horizontalalignment='center', verticalalignment='center', 
                transform=plot.transAxes
            )
        
    def plot_loss_and_accuracy(self, 
                               df_loss, df_accuracy,
                               figsize=(12,6), padding=3):
        plt.figure()
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        fig.tight_layout(pad=3)
        
        self.plot(
            df=df_loss,
            ax=axes[0],
            title='Loss over %d epochs' % df_loss.size,
            xlabel='Epochs', ylabel='Loss',
            txt='Final: %.3f, Min: %.3f' % (df_loss.iloc[-1], df_loss.min()),
            x=0.8,
            y=0.9
        )
        self.plot(
            df=df_accuracy,
            ax=axes[1],
            title='Accuracy over %d epochs' % df_accuracy.size,
            xlabel='Epochs', ylabel='Accuracy',
            txt='Final: %.3f, Peak: %.3f' % (df_accuracy.iloc[-1], df_accuracy.max()),
            x=0.8,
            y=0.1
        )
        
        
    def plot_train_valid_test(self, 
                              fname='noname', # to save plots
                              figsize=(12,6), padding=3):
        # train
        self.plot_loss_and_accuracy(
            self.rc.train['loss'], self.rc.train['accuracy'],
            figsize, padding
        )
        plt.savefig('%s_train' % fname)
        
        self.plot_loss_and_accuracy(
            self.rc.valid['loss'], self.rc.valid['accuracy'],
            figsize, padding
        )
        plt.savefig('%s_valid' % fname)
        
        self.plot_loss_and_accuracy(
            self.rc.test['loss'], self.rc.test['accuracy'],
            figsize, padding
        )
        plt.savefig('%s_test' % fname)
