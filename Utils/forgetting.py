import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def plot(filename):

    # Load the data from a CSV file
    df = pd.read_csv(filename)

    # Pivot the dataframe to create a matrix for heatmap
    heatmap_data = df.pivot(index="training_exp", columns="eval_exp", values="forgetting")

    # Plotting the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=False, fmt=".4f", cmap="coolwarm", cbar=True)
    plt.title("Forgetting Heatmap")
    plt.xlabel("Evaluation Experience")
    plt.ylabel("Training Experience")
    plt.savefig(filename.replace('.csv', '_heatmap.png'))

plot('../Experiments/results/csv/lamaml_scifar100_KAN/eval_results.csv')

def plot_accuracy(filename):
    # Load the data from a CSV file
    df = pd.read_csv(filename)

    # Pivot the dataframe to create a matrix for heatmap
    heatmap_data = df.pivot(index="training_exp", columns="eval_exp", values="eval_accuracy")

    # Plotting the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=False, fmt=".4f", cmap="coolwarm", cbar=True)
    plt.title("Accuracy Heatmap")
    plt.xlabel("Evaluation Experience")
    plt.ylabel("Training Experience")
    plt.savefig(filename.replace('.csv', '_accuracy_heatmap.png'))

plot_accuracy('../Experiments/results/csv/lamaml_scifar100_KAN/eval_results.csv')