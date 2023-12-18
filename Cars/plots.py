import os
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

current_directory = os.getcwd()
relative_path_plot_style = 'deeplearning.mplstyle'

file_path_plot_style = os.path.join(current_directory, relative_path_plot_style)
plt.style.use(file_path_plot_style)


def plot_with_column(df, column):
    """
    Plots x against y
    
    Args:
      df                    : dataframe
      column (string)       : name of the column in the dataframe to plot against Y
      
    """
    plt.figure()
    plt.scatter(df[column].to_numpy(), df["Price"], marker='o', c='b', s=10)
    plt.title(f"{column}/Price")
    plt.ylabel("Price")
    plt.xlabel(column)
    plt.show()


def plotHistograms(dataframe, rows, cols, column_names, numerical_data_names, title="Distribution of numerical data",
                   height=1400, width=800):
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=column_names)
    for i in range(0, rows):
        for j in range(0, cols):
            if column_names[i * cols + j] != 'Price':
                fig.add_trace(
                    go.Histogram(x=dataframe[str(column_names[i * cols + j])], name=numerical_data_names[i * cols + j]),
                    row=i + 1, col=j + 1) if j + i * cols < len(column_names) else None
            else:
                fig.add_trace(
                    go.Histogram(x=np.exp(dataframe[str(column_names[i * cols + j])]), name=numerical_data_names[i * cols + j]),
                    row=i + 1, col=j + 1) if j + i * cols < len(column_names) else None

    fig.update_layout(height=height, width=width, title_text=title)
    fig.show()


def plotBarGraphs(dataframe, rows, cols, column_names, title="Distribution of categorical data", height=1400,
                  width=1400):
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=column_names)
    for i in range(0, rows):
        for j in range(0, cols):
            if j + i * cols < len(column_names):
                count = dataframe[str(column_names[i * cols + j])].value_counts().reset_index()
                fig.add_trace(go.Bar(y=count['count'], x=count[str(column_names[i * cols + j])]), row=i + 1, col=j + 1)

    fig.update_layout(height=height, width=width, title_text=title)
    fig.show()


def showHeatMap(dataframe, columns):
    sns.heatmap(dataframe[columns].corr(), annot=True, cmap='coolwarm')
    plt.show()


def showResult(X, y_target, y_predicated, num_windows=10):
    # Ensure X, y_target, and y_predicated have the same number of samples
    assert X.shape[0] == y_target.shape[0] == y_predicated.shape[0], "Number of samples must be the same"

    num_features = X.shape[1]

    # Plot 10 windows with 10 graphs each
    for window_index in range(min(num_windows, num_features // 10)):
        fig, axs = plt.subplots(2, 5, figsize=(15, 6), sharey=True)
        fig.suptitle(f'Linear Regression Model Weights Visualization - Window {window_index + 1}')

        for i in range(2):
            for j in range(5):
                weight_index = window_index * 10 + i * 5 + j
                x_column = X[:, weight_index]
                axs[i, j].scatter(x_column, y_target, label='True values')
                axs[i, j].scatter(x_column, y_predicated, color='red', label=f'Weight {weight_index + 1}')
                axs[i, j].set_title(f'Weight {weight_index + 1}')

        # Adjust layout and show plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    # Plot 1 window with 1 graph for the remaining weights
    if num_features % 10 != 0:
        fig, axs = plt.subplots(1, num_features % 10, figsize=(15, 3), sharey=True)
        fig.suptitle(f'Linear Regression Model Weights Visualization - Window {num_windows + 1}')

        for j in range(num_features % 10):
            weight_index = num_windows * 10 + j
            x_column = X[:, weight_index]
            axs[j].scatter(x_column, y_target, label='True values')
            axs[j].scatter(x_column, y_predicated, color='red', label=f'Weight {weight_index + 1}')
            axs[j].set_title(f'Weight {weight_index + 1}')

        # Adjust layout and show plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
