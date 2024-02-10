import numpy as np
import torch
import matplotlib.pyplot as plt
import random

from graphdiffusion import utils


def to_numpy_array(input_array):
    # Check if the input_array is a PyTorch tensor
    if isinstance(input_array, torch.Tensor):
        # Detach the tensor from the current graph and move it to cpu
        return input_array.detach().cpu().numpy()
    # Check if the input_array is already a NumPy array
    elif isinstance(input_array, np.ndarray):
        return input_array
    else:
        raise TypeError("The input should be a PyTorch tensor or a NumPy array")


def plot_array_on_axis(array, axis, arrays):
    """
    Draws the input 1D PyTorch array on the provided Matplotlib axis using a scatter plot.

    :param array: A 1D PyTorch array of numbers to be plotted.
    :param axis: A Matplotlib axis object on which to draw the scatter plot.
    :param x_limits: A tuple containing the minimum and maximum x-axis limits.
    :param y_limits: A tuple containing the minimum and maximum y-axis limits.
    """

    array = to_numpy_array(array)
    for i in range(array.shape[0]):
        x = np.arange(len(array[i, :]))  # Generate x-coordinates as indices
        y = array[i, :]  # y-coordinates are the array values
        axis.plot(x, y, alpha=0.5)  # Plotting the scatter plot on the provided axis

    # axis.set_xlim(x_limits)  # Set x-axis limits
    # axis.set_ylim(y_limits)  # Set y-axis limits
    # min_x = np.min([np.min(x[:, :]) for x in arrays])
    # max_x = np.max([np.max(x[:, :]) for x in arrays])
    # min_y = 0.0
    # max_y = np.max([x.shape[1] for x in arrays])  # TODO fix
    # axis.set_xlim([min_x, max_x])
    # axis.set_ylim([min_y, max_y])

    axis.set_xlabel("Index")
    axis.set_ylabel("Value")
    # axis.set_title('Scatter Plot')


def plot_2darray_on_axis(array, axis, arrays):
    """
    Draws the input 2D NumPy array on the provided Matplotlib axis using a scatter plot.
    Each row in the 2D array is considered a separate point in the scatter plot.
    If the input array is 1D, it's treated as a single point with batch size 1.

    :param array: A 2D or 1D NumPy array of numbers to be plotted.
                  If 2D, shape should be (batch_size, 2).
                  If 1D, shape should be (2,).
    :param axis: A Matplotlib axis object on which to draw the scatter plot.
    :param arrays: List of all arrays that will be plotttet.
    """
    # If the input array is 1D, reshape it to 2D with batch size 1
    if array.ndim == 1:
        array = array.reshape(1, -1)

    # Ensure array is a 2D tensor after reshaping
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Input array must be a 2D or 1D array with shape (batch_size, 2) or (2,)")

    if np.isnan(array).any() or np.isinf(array).any():
        warnings.warn("Input array contains NaN or Inf values. These will be replaced with 0.")
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    # Plotting the scatter plot on the provided axis for all points
    axis.scatter(array[:, 0], array[:, 1], s=100, alpha=0.5, edgecolors="none")

    # Set x and y axis limits
    min_x = np.min([np.min(x[:, 0]) for x in arrays])
    max_x = np.max([np.max(x[:, 0]) for x in arrays])
    min_y = np.min([np.min(x[:, 1]) for x in arrays])
    max_y = np.max([np.max(x[:, 1]) for x in arrays])
    # axis.set_xlim([min_x,max_x])
    # axis.set_ylim([min_y,max_y])

    # Setting labels
    axis.set_xlabel("x")
    axis.set_ylabel("y")


def create_grid_plot(arrays, outfile="test.pdf", plt_show=False, plot_data_func=None):
    """
    Creates a grid plot for a list of 1D PyTorch arrays, each plotted on a separate axis.

    :param arrays: A list of 2D PyTorch arrays.
    """
    arrays = [to_numpy_array(array) for array in arrays]

    plot_data_func = plot_data_func or plot_array_on_axis

    # Find global x and y limits
    max_length = max(len(arr) for arr in arrays)
    max_value = max(arr.max() for arr in arrays)
    min_value = min(arr.min() for arr in arrays)
    x_limits = (-0.5, max_length - 0.5)
    y_limits = (min_value - 0.5, max_value + 0.5)

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
        plot_data_func(array, axes[i], arrays)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

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
        arrays.append(torch.tensor([1, 2 + random.random(), 3 * +random.random(), 4]))

    create_grid_plot(arrays)


def plot_data_as_normal_pdf(array, axis, arrays):
    """
    Plot the Kernel Density Estimate (KDE) of the data and the standard normal probability density function (PDF)
    for comparison.

    Parameters:
    - array: numpy.ndarray
        A 2D numpy array whose distribution is to be plotted.
    - axis: matplotlib.axes._subplots.AxesSubplot
        The matplotlib axis on which to plot the KDE and standard normal PDF.

    Returns:
    None
    """
    import warnings
    import seaborn as sns
    import scipy.stats

    # Suppress future warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # Function to calculate standard normal PDF
    def standard_normal_pdf(x):
        return scipy.stats.norm(0, 1).pdf(x)

    # Generate x values for plotting the standard normal PDF
    x_values = np.linspace(-4.5, 4.5, 30)

    # Plot the standard normal PDF
    axis.plot(x_values, [standard_normal_pdf(x) for x in x_values], c="black", alpha=0.3, lw=5, label="Reference Normal PDF", linestyle="--")

    # Plot the KDE of the input array
    sns.kdeplot(data=array.flatten(), ax=axis)

    # Add legend to the plot
    axis.legend()


def plot_image_on_axis(array, axis, arrays=None):
    # assumge images is in range [-1, 1]

    from graphdiffusion.utils import tensor_to_img

    array = array[0, :]
    array = array.squeeze()
    if array.ndim == 1:
        # assume three channels and square image
        img_size = int(np.sqrt(array.numel() // 3))
        array = array.reshape(3, img_size, img_size)

    array = tensor_to_img(torch.tensor(array))  # assumge images is in range [-1, 1]
    array = np.clip(array, 0, 255)  # should be redundant
    if array.dtype == np.float32 or array.dtype == np.float64:
        array = array / 255.0

    # Display the image
    axis.imshow(array.astype(np.uint8) if array.dtype == np.float32 or array.dtype == np.float64 else array)
