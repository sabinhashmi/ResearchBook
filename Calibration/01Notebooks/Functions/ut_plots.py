import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["green", "orange", "red"])

#Plotting UT Plane
def ut_plot(data_,path=None,filename=None,save_fig=False,show_fig=False,vmin_vmax_scaled=False):
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    if vmin_vmax_scaled:
        # sns.heatmap(data_, cmap=custom_cmap, cbar=True, linewidths=0.5, linecolor='black',vmin=0,vmax=0.000005) #This is for KLDiv ToyData
        sns.heatmap(data_, cmap=custom_cmap, cbar=True, linewidths=0.5, linecolor='black',vmin=0,vmax=1) #This is for KLRealRunData
    else:
        sns.heatmap(data_, cmap=custom_cmap, cbar=True, linewidths=0.5, linecolor='black')

    # Get the center row and column
    center_row, center_col = data_.shape[0] // 2, data_.shape[1] // 2

    # Add the central horizontal and vertical lines
    plt.axhline(y=center_row, color='black', linewidth=2)  # Horizontal Line
    plt.axvline(x=center_col, color='black', linewidth=2)  # Vertical Line

    # Plot the four lines around the center
    plt.plot([center_col-1, center_col+1], [center_row-1, center_row-1], color='black', linewidth=1.5)
    plt.plot([center_col-1, center_col+1], [center_row+1, center_row+1], color='black', linewidth=1.5)
    plt.plot([center_col-1, center_col-1], [center_row-1, center_row+1], color='black', linewidth=1.5)
    plt.plot([center_col+1, center_col+1], [center_row-1, center_row+1], color='black', linewidth=1.5)

    # Add titles and labels
    #UTaX is getting filtered from data_preparation_for_ut function
    plt.title("Heatmap of UTaX")
    plt.xlabel('Staves')
    plt.ylabel('Rows')

    # Save or show the plot
    if save_fig:
        os.makedirs(path, exist_ok=True)
        plt.savefig(f'{path}/{filename}.png')
    if show_fig:
        plt.show()

    # Close the plot
    plt.close()