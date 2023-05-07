import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_results(results_df, plot_metrics=[], rot=0):
    """
    Evaluate `results_df` on a common set of metrics.
    Metrics: ['Annualized Mean Return', 'Annualized Risk', 'Sharpe Ratio', 'Sortino Ratio']
    """
    metrics = {}
    metrics['Annualized Mean Return'] = (1+results_df).prod()**(252/len(results_df))-1
    metrics['Annualized Risk'] = np.sqrt(results_df.var()*252)
    metrics['Sharpe Ratio'] = metrics['Annualized Mean Return'] / metrics['Annualized Risk']
    
    def sortino_ratio(series, N=252, rf=0):
        mean = series.mean() * N -rf
        std_neg = series[series<0].std()*np.sqrt(N)
        return mean/std_neg
    
    metrics['Sortino Ratio'] = results_df.apply(sortino_ratio, axis=0)
    
    if len(plot_metrics)>0:
        fig, axes = plt.subplots(1, len(plot_metrics), figsize=(len(plot_metrics)*4, 4))
        for i,metric in enumerate(plot_metrics):
            metrics[metric].plot.bar(ax=axes[i], title=metric, rot=rot, color='#F4C430')
    
    return metrics