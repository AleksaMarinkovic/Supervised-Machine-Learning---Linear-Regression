import os
import matplotlib.pyplot as plt

current_directory = os.getcwd()
relative_path_plot_style = 'deeplearning.mplstyle'

file_path_plot_style = os.path.join(current_directory, relative_path_plot_style)
plt.style.use(file_path_plot_style)

def plot_with_column(x, y, column, x_label, title='Line plot'):
    """
    Plots x against y 
    
    Args:
      x (ndarray (m,n))     : input data, m examples, n features
      y (ndarray (m,1))     : input data, m examples
      column (int)          : which column from X to plot against Y
      x_label (string)      : label for X axis
      title (string)        : title for the graph. "Line plot" by default
      
    """
    plt.scatter(x[:,column], y, marker='o', c='b')
    plt.title(title)
    plt.ylabel("Price")
    plt.xlabel(x_label)
    plt.show()