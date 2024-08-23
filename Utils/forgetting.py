import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def plot_forgetting(filename, folder):

    # Load the data from a CSV file
    df = pd.read_csv(filename)

    # Pivot the dataframe to create a matrix for heatmap
    heatmap_data = df.pivot(index="training_exp", columns="eval_exp", values="forgetting")

    # Plotting the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="coolwarm", cbar=True)
    plt.title("Forgetting Heatmap")
    plt.xlabel("Evaluation Experience")
    plt.ylabel("Training Experience")
    plt.savefig(folder+'_forgetting_heatmap.png')


def plot_accuracy(filename,folder):
    # Load the data from a CSV file
    df = pd.read_csv(filename)

    # Pivot the dataframe to create a matrix for heatmap
    heatmap_data = df.pivot(index="training_exp", columns="eval_exp", values="eval_accuracy")

    # Plotting the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="coolwarm", cbar=True)
    plt.title("Accuracy Heatmap")
    plt.xlabel("Evaluation Experience")
    plt.ylabel("Training Experience")
    plt.savefig(folder+'_accuracy_heatmap.png')

folder = "../Experiments/results/csv/gen_replay_pmnist_KAN"
file = folder+"/eval_results.csv"
plot_accuracy(file,folder)
plot_forgetting(file,folder)
