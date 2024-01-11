
import numpy as np
import torch
import matplotlib.pyplot as plt
import random

def plot_array_on_axis(array, axis, x_limits, y_limits):
    """
    Draws the input 1D PyTorch array on the provided Matplotlib axis using a scatter plot.

    :param array: A 1D PyTorch array of numbers to be plotted.
    :param axis: A Matplotlib axis object on which to draw the scatter plot.
    :param x_limits: A tuple containing the minimum and maximum x-axis limits.
    :param y_limits: A tuple containing the minimum and maximum y-axis limits.
    """
    x = torch.arange(len(array))  # Generate x-coordinates as indices
    y = array  # y-coordinates are the array values

    axis.scatter(x.numpy(), y.numpy(), s=500, alpha=0.5, edgecolors='none')   # Plotting the scatter plot on the provided axis
    axis.set_xlim(x_limits)  # Set x-axis limits
    axis.set_ylim(y_limits)  # Set y-axis limits
    axis.set_xlabel('Index')
    axis.set_ylabel('Value')
    #axis.set_title('Scatter Plot')

def create_grid_plot(arrays, outfile="test.png", plt_show = False, plot_data_func = None):
    """
    Creates a grid plot for a list of 1D PyTorch arrays, each plotted on a separate axis.

    :param arrays: A list of 1D PyTorch arrays.
    """
    plot_data_func = plot_data_func or plot_array_on_axis

    # Find global x and y limits
    max_length = max(len(arr) for arr in arrays)
    max_value = max(arr.max() for arr in arrays)
    min_value = min(arr.min() for arr in arrays)
    x_limits = (-0.5, max_length - 0.5)
    y_limits = (min_value-0.5, max_value+0.5)

    # Determine the grid size
    num_plots = len(arrays)
    rows = int(np.ceil(np.sqrt(num_plots)))
    cols = int(np.ceil(num_plots / rows))

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    # Check if axes is a single Axes object
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])  # Wrap it in an array for consistent handling

    # Flatten the axes array
    axes = axes.flatten()

    # Plot each array on its respective axis
    for i, array in enumerate(arrays):
        plot_data_func(array, axes[i], x_limits, y_limits)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    try:
        if plt_show:
            plt.show()
    except:
        pass
    try:
        if outfile is not None:
            plt.savefig(outfile) 
    except:
        pass

if __name__ == "__main__":
    # Example usage
    arrays = list()

    for _ in range(100):
        arrays.append(torch.tensor([1, 2+random.random(), 3*+random.random(), 4]))

    create_grid_plot(arrays)
