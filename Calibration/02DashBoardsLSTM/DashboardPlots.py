import  pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap

file = ''

def individual_plots():
    loss_file = pd.read_csv(f'./Outputs/file/loss.csv')

    plt.figure(figsize=(12,6))
    sns.lineplot(loss_file['LossValues'],color='g',label='MAE Loss')
    plt.title("Loss Function")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    # plt.savefig(f"./Plots/Losses/{id}_loss.png")
    plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------------------- #

    staves_file = pd.read_csv(f'./Outputs/file/staves_prediction.csv')
    plt.figure(figsize=(12,6))
    sns.lineplot(staves_file['TrueValues'],label='True Value')
    sns.lineplot(staves_file['Predictions'],label='Predictions')
    plt.ylim(0,1)
    plt.title('UTaX Test Values and Predictions')
    plt.xlabel('Module ID')
    plt.ylabel('Value')
    plt.tight_layout()
    # plt.savefig(f"./Plots/Compare/{id}_compare.png")
    plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------------------- #

    training_file = pd.read_csv(f'./Outputs/file/train_sample.csv')

    plt.figure(figsize=(12, 6))
    sns.lineplot(x = training_file.index, y =training_file['PedestalValue'],label='TrainingData') #train_data for non-scaled data
    sns.lineplot(x=pd.RangeIndex(start=training_file.index.max(), stop=training_file.index.max()+256, step=1), y=staves_file['Sequence'], label='Sequence')
    sns.lineplot(x=pd.RangeIndex(start=training_file.index.max()+256, stop=training_file.index.max()+512, step=1), y=staves_file['TrueValues'], label='True Values')
    sns.lineplot(x=pd.RangeIndex(start=training_file.index.max()+256, stop=training_file.index.max()+512, step=1), y=staves_file['Predictions'], label='Forecast')
    plt.title('Forecasting Model Performance')
    plt.xlabel('Index')
    plt.ylabel('PedestalValue')
    plt.legend(ncols=2)
    # plt.savefig(f"./Plots/Forecast/{id}_forecast.png")
    plt.tight_layout()
    plt.show()

    return




def dashboard(file_name):


    fig, axes = plt.subplots(2,2,figsize=(16,10),gridspec_kw={'height_ratios': [1, 1],
                                                                            'width_ratios': [2, 1]},sharex=False)


    ax1, ax2, ax3, ax4 = axes.flatten()

    # --------------------------------------------------------------------------------------------------------------------------------------------------- #

    staves_file = pd.read_csv(f'./Outputs/{file_name}/staves_prediction.csv')

    staves_file['Module'] = staves_file['Staves'].str.cat(staves_file['Rows'],sep=',')

    sns.lineplot(data = staves_file,x = 'Module', y='TrueValues',label='True Value',ax=ax1)
    sns.lineplot(data = staves_file,x = 'Module', y='Predictions',label='Predictions',ax=ax1)

    ax1.set_title('UTaX Test Values and Predictions')
    ax1.set_xlabel('Module ID')
    ax1.set_ylabel('Value')
    ax1.set_ylim(0, 1)

    ax1.set_xticks(staves_file.index[::5])  # Set the positions of the ticks
    ax1.set_xticklabels(staves_file['Module'][::5], rotation=90, fontsize=8)  # Set the labels with rotation and font size


    ax1.legend(ncols=2)


    # --------------------------------------------------------------------------------------------------------------------------------------------------- #

    loss_file = pd.read_csv(f'./Outputs/{file_name}/loss.csv')

    sns.lineplot(loss_file['LossValues'],color='g',label='MAE Loss',ax=ax2)
    ax2.set_title("Loss Function")
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Loss')
    ax2.legend(ncols=2)

    # --------------------------------------------------------------------------------------------------------------------------------------------------- #

    training_file = pd.read_csv(f'./Outputs/{file_name}/train_sample.csv')

    sns.lineplot(x = training_file.index, y =training_file['PedestalValue'],label='TrainingData',ax=ax3)
    sns.lineplot(x=pd.RangeIndex(start=training_file.index.max(), stop=training_file.index.max()+256, step=1), y=staves_file['Sequence'], label='Sequence',ax=ax3)
    sns.lineplot(x=pd.RangeIndex(start=training_file.index.max()+256, stop=training_file.index.max()+512, step=1), y=staves_file['TrueValues'], label='True Values',ax=ax3)
    sns.lineplot(x=pd.RangeIndex(start=training_file.index.max()+256, stop=training_file.index.max()+512, step=1), y=staves_file['Predictions'], label='Forecast',ax=ax3)
    ax3.set_title('Forecasting Model Performance')
    ax3.set_xlabel('Index')
    ax3.set_ylabel('PedestalValue')

    ax3.set_ylim(-1, 1)
    ax3.legend(ncols=4)


    # --------------------------------------------------------------------------------------------------------------------------------------------------- #

    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["green", "orange", "red"])

    staves_file = pd.read_csv(f'./Outputs/{file_name}/staves_prediction.csv')

    data_ = staves_file.groupby(['Staves','Rows']).agg('mean').reset_index()

    row_order = ['S4T','M4T','S3T','M3T','S2T','M2T','S1T','M1T','M1B','S1B','M2B','S2B','M3B','S3B','M4B','S4B']
    stave_order = ['1C', '2C', '3C', '4C', '5C', '6C', '7C', '8C', '8A', '7A', '6A', '5A', '4A', '3A', '2A', '1A']

    data_['Rows'] = pd.Categorical(data_['Rows'], categories=row_order, ordered=True)
    data_['Staves'] = pd.Categorical(data_['Staves'], categories=stave_order, ordered=True)

    data_['Difference'] = np.abs(data_['Difference'])


    data_ = data_.pivot(index='Rows', columns='Staves', values='Difference')

    sns.heatmap(data_, cmap=custom_cmap, cbar=False, linewidths=0.5, linecolor='black',vmin=0,vmax=1.0,ax=ax4)

    # sns.heatmap(data_, cmap=custom_cmap, cbar=True, linewidths=0.5, linecolor='black',ax=ax4)

    center_row, center_col = data_.shape[0] // 2, data_.shape[1] // 2

    ax4.axhline(y=center_row, color='black', linewidth=2)  # Horizontal Line
    ax4.axvline(x=center_col, color='black', linewidth=2)  # Vertical Line

        # Plot the four lines around the center
    ax4.plot([center_col-1, center_col+1], [center_row-1, center_row-1], color='black', linewidth=1.5)
    ax4.plot([center_col-1, center_col+1], [center_row+1, center_row+1], color='black', linewidth=1.5)
    ax4.plot([center_col-1, center_col-1], [center_row-1, center_row+1], color='black', linewidth=1.5)
    ax4.plot([center_col+1, center_col+1], [center_row-1, center_row+1], color='black', linewidth=1.5)

    ax4.set_title("Absolute Difference in Forecast")
    ax4.set_xlabel('Staves')
    ax4.set_ylabel('Rows')
    

    # Adjust layout
    plt.tight_layout()

    dashboard_path = './Plots'
    os.makedirs(dashboard_path,exist_ok=True)
    plt.savefig(f'{dashboard_path}/{file_name}.png')
    plt.show()

