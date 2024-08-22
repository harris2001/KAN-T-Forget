import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot(filenames, title, save_path=None):
    accuracies = []
    forgetting = []
    titles = []
    for filename in filenames:
        titles.append(filename.split('/')[-2])
        with open(filename) as f:
            df = pd.read_csv(f)
        accuracies.append(np.array(df.loc[df['training_exp']==19]['eval_accuracy']))
        forgetting.append(np.array(df.loc[df['training_exp']==19]['forgetting']))
        print(accuracies,forgetting)
        titles.append(filename.split('/')[-2]+"_KAN")
        with open(filename.replace('/eval','_KAN/eval')) as f:
            df = pd.read_csv(f)
        accuracies.append(np.array(df.loc[df['training_exp']==19]['eval_accuracy']))
        forgetting.append(np.array(df.loc[df['training_exp']==19]['forgetting']))
        print(accuracies,forgetting)

    for i in range(len(accuracies)):
        plt.plot(accuracies[i], label=titles[i])
    plt.xlabel('Experience')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    for i in range(len(forgetting)):
        plt.plot(forgetting[i], label=titles[i])
    plt.xlabel('Experience')
    plt.ylabel('Forgetting')
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path.replace('.png','_forgetting.png'))
    plt.show()


plot(['../Experiments/results/csv/lamaml_scifar100/eval_results.csv']
      ,'La-MAML on Split CIFAR-100', 'CIFAR-100.png')