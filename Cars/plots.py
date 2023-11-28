import os
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

current_directory = os.getcwd()
relative_path_plot_style = 'deeplearning.mplstyle'

file_path_plot_style = os.path.join(current_directory, relative_path_plot_style)
plt.style.use(file_path_plot_style)

def plot_with_column(df, column, title='Line plot'):
    """
    Plots x against y 
    
    Args:
      df                    : dataframe
      column (string)       : name of the column in the dataframe to plot against Y
      x_label (string)      : label for X axis
      title (string)        : title for the graph. "Line plot" by default
      
    """

    plt.scatter(df[column].to_numpy(), df["Price"], marker='o', c='b')
    plt.title(title)
    plt.ylabel("Price")
    plt.xlabel(column)
    plt.show()

def plotHistograms(dataframe, rows, cols, column_names, numerical_data_names, title="Distribution of numerical data", height=1400, width=800):
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=column_names)
    for i in range(0, rows):
        for j in range(0, cols):
            fig.add_trace(go.Histogram(x=dataframe[str(column_names[i*cols+j])], name=numerical_data_names[i*cols+j]), row=i+1, col=j+1) if j + i*cols < len(column_names) else None

    fig.update_layout(height=height, width=width, title_text=title)
    fig.show()

def plotBarGraphs(dataframe, rows, cols, column_names, title="Distribution of categorical data", height=1400, width=1400):
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=column_names)
    for i in range(0, rows):
        for j in range(0, cols):
            if j + i * cols < len(column_names):
                count = dataframe[str(column_names[i*cols+j])].value_counts().reset_index()
                fig.add_trace(go.Bar(y=count['count'], x=count[str(column_names[i*cols+j])]), row=i+1, col=j+1)

    fig.update_layout(height=height, width=width, title_text=title)
    fig.show()

def showHeatMap(dataframe):
    sns.heatmap(dataframe.corr(), annot=True, cmap='coolwarm')
    plt.show()