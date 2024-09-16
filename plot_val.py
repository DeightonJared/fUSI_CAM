import numpy as np
import matplotlib.pyplot as plt


metrics = np.load('mk_results/mk1_5/validation_metrics60.npy',allow_pickle='TRUE').item()
CAMS = np.load('mk_results/mk1_5/validation_CAMS60.npy',allow_pickle='TRUE').item()

metrics['Accuracy'] = metrics.pop('accuracy')
metrics['AUC'] = metrics.pop('roc_auc_score')
metrics['Precision'] = metrics.pop('precision')
metrics['Recall'] = metrics.pop('recall')
metrics['Confusion Matrix'] = metrics.pop('confusion_matrix')
metrics['F1 Score'] = metrics.pop('f1_score')

step_size = 60

times = np.arange(0, (len(list(metrics['Accuracy'].keys())) * step_size)/60, (step_size)/60)

def plot_metric_std(metric):
    means = 100*np.array([np.mean(metrics[metric][i]) for i in metrics[metric].keys()])
    stds = 100*np.array([np.std(metrics[metric][i]) for i in metrics[metric].keys()])
    

    plt.figure(figsize=(12,8))

    with plt.style.context('seaborn-v0_8-whitegrid'):
        plt.plot(times, means, label='Mean ' + metric, color='midnightblue', linewidth = 2)
        plt.fill_between(times, means -  stds ,means + stds, color='lightblue', alpha=0.75, label='Standard Deviation')
        plt.xlabel('Time(m)', fontsize = 18)
        plt.ylabel(metric, fontsize = 18)
        # plt.title('Mean Validation ' + metric + ' with Min/Max', fontsize = 20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.axvline(x = 5, color = 'red')
        plt.legend(loc  = 'lower right', frameon=True, facecolor='white', edgecolor='black', fontsize = 18, ncol = 1)
        plt.grid(True, linewidth=0.5, alpha=0.7)
        
        
        plt.show()
        
def plot_metric_se(metric):
    means = 100*np.array([np.mean(metrics[metric][i]) for i in metrics[metric].keys()])
    
    ses = 100*np.array([np.std(metrics[metric][i]) / len(metrics[metric][i]) for i in metrics[metric].keys()])

    plt.figure(figsize=(12,8))

    with plt.style.context('seaborn-v0_8-whitegrid'):
        plt.plot(times, means, label='Mean ' + metric, color='midnightblue', linewidth = 2)
        plt.fill_between(times, means -  ses ,means + ses, color='lightblue', alpha=0.75, label='Standard Error')
        plt.xlabel('Time(m)', fontsize = 18)
        plt.ylabel(metric, fontsize = 18)
        # plt.title('Mean Validation ' + metric + ' with Min/Max', fontsize = 20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.axvline(x = 5, color = 'red')
        plt.legend(loc  = 'lower right', frameon=True, facecolor='white', edgecolor='black', fontsize = 18, ncol = 1)
        plt.grid(True, linewidth=0.5, alpha=0.7)
        
        
        plt.show()
        
def plot_metric_minmax(metric): 
    
    means = 100*np.array([np.mean(metrics[metric][i]) for i in metrics[metric].keys()])
    mins = 100*np.array([np.min(metrics[metric][i]) for i in metrics[metric].keys()])
    maxs = 100*np.array([np.max(metrics[metric][i]) for i in metrics[metric].keys()])
    
    plt.figure(figsize=(12,8))

    with plt.style.context('seaborn-v0_8-whitegrid'):
        plt.plot(times, means,label='Mean ' + metric, color='midnightblue', linewidth = 2)
        plt.fill_between(times, mins ,maxs, color='lightblue', alpha=0.75, label='Min/Max ' + metric)
        plt.xlabel('Time(s)', fontsize = 18)
        plt.ylabel(metric, fontsize = 18)
        # plt.title('Mean Validation ' + metric + ' with Min/Max', fontsize = 20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(loc  = 'lower right', frameon=True, facecolor='white', edgecolor='black', fontsize = 18, ncol = 1)
        plt.grid(True, linewidth=0.5, alpha=0.7)
        
        
        plt.show()

def plot_all_metrics(metrics, times, style='std'):
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    metric_names = ['Accuracy', 'F1 Score', 'Precision', 'AUC', 'Recall']

    plt.figure(figsize=(12,8))
    with plt.style.context('seaborn-v0_8-whitegrid'):
        for metric, color in zip(metric_names, colors):
            means = 100 * np.array([np.mean(metrics[metric][i]) for i in metrics[metric].keys()])
            
            if style == 'std':
                stds = 100 * np.array([np.std(metrics[metric][i]) for i in metrics[metric].keys()])
                plt.fill_between(times, means - stds, means + stds, color=color, alpha=0.3)
            elif style == 'minmax':
                mins = 100 * np.array([np.min(metrics[metric][i]) for i in metrics[metric].keys()])
                maxs = 100 * np.array([np.max(metrics[metric][i]) for i in metrics[metric].keys()])
                plt.fill_between(times, mins, maxs, color=color, alpha=0.3)

            plt.plot(times, means, label=metric, color=color, linewidth=2)

        plt.xlabel('Time (m)', fontsize = 18)
        plt.ylabel('Metrics', fontsize = 18)
        # plt.title('Validation Metrics Over Time', fontsize = 16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(loc  = 'lower right', frameon=True, facecolor='white', edgecolor='black', fontsize=18, ncol=1)
        plt.grid(True, linewidth=0.5, alpha=0.7)
        plt.show()


#%%
plot_metric_std('Accuracy')
plot_metric_se('Accuracy')
plot_metric_minmax('Accuracy')


plot_metric_std('AUC')
plot_metric_minmax('AUC')

plot_all_metrics(metrics, times, style='std')


#%%
mk_CAMS_all = CAMS['MK']
Sal_CAMS_all = CAMS['Saline']

plt.figure(figsize= (16,12))
for i, mk_idx in enumerate(mk_CAMS_all.keys()):
    accuracies = np.array([metrics['Accuracy'][step][i] for step in list(metrics['Accuracy'].keys())])
    plt.plot(100*accuracies, label = mk_idx)
    
plt.xlabel('Time (m)', fontsize = 18)
plt.ylabel('Accuracy', fontsize = 18)
plt.legend(fontsize = 18)
    
#%%
def plot_metric_se(metric, ax):
    means = 100 * np.array([np.mean(metrics[metric][i]) for i in metrics[metric].keys()])
    ses = 100 * np.array([np.std(metrics[metric][i]) / np.sqrt(len(metrics[metric][i])) for i in metrics[metric].keys()])
    
    ax.plot(times, means, label='Mean ' + metric, color='midnightblue', linewidth=2)
    ax.fill_between(times, means - ses, means + ses, color='lightblue', alpha=0.75, label='Standard Error')
    ax.set_xlabel('Time (m)', fontsize=16)
    ax.set_ylabel(metric, fontsize=16)
    ax.axvline(x=5, color='red')
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='black', fontsize=12)
    ax.grid(True, linewidth=0.5, alpha=0.7)

fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
metric_names = ['Accuracy', 'F1 Score', 'Precision', 'AUC', 'Recall']

for ax, metric in zip(axs.flat, metric_names):
    plot_metric_se(metric, ax)

# Hide the empty subplot (the 6th one)
fig.delaxes(axs[2, 1])
fig.suptitle('MK-801 CNN Classification Metrics Across Time', fontsize = 20)

plt.tight_layout()
plt.show()