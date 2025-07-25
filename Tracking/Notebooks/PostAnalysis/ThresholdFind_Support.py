import matplotlib.pyplot as plt

def threshold_plots(data_for_plot,track_type,bar_width,directory,plot_name):

    fig, ax1 = plt.subplots(figsize=(16, 8))

#     bar_width = 0.4
    edge_color = 'black'

    bars1 = ax1.bar(data_for_plot['Threshold'], data_for_plot[f'True{track_type}Tracks'], 
                    label=f'True{track_type}Tracks', color='skyblue', 
                    edgecolor=edge_color, width=bar_width)
    
    bars2 = ax1.bar(data_for_plot['Threshold'], data_for_plot[f'{track_type}GhostTracks'], 
                    bottom=data_for_plot[f'True{track_type}Tracks'], 
                    label=f'Ghost{track_type}Tracks', color='lightcoral', 
                    edgecolor=edge_color, width=bar_width)
    
#     # Change the color of the first bar (base stack)
#     bars1[0].set_color('lightblue')  # Example color for the baseline

# # Change the color of the first upper stack bar
#     bars2[0].set_color('coral')  # Same color or different for the baseline




    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()

        # ax1.text(bar1.get_x() + bar1.get_width() / 2, 
        #         bar1.get_y() + height1 / 2, 
        #         f"{(data_for_plot[f'{track_type}Ratio'].iloc[i])}%", 
        #         ha='center', va='center', 
        #         color='black', fontsize=10, style='italic')

        # ax1.text(bar2.get_x() + bar2.get_width() / 2, 
        #         bar2.get_y() + height2 / 2, 
        #         f"{(data_for_plot[f'{track_type}GhostRatio'].iloc[i])}%", 
        #         ha='center', va='center', 
        #         color='black', fontsize=10, style='italic')

        # For the base stack (True{track_type}Tracks), place text at the base
        ax1.text(bar1.get_x() + bar1.get_width() / 2, 
                bar1.get_y() + height1 / 2, 
                f"{(data_for_plot[f'{track_type}Ratio'].iloc[i])}%", 
                ha='center', va='center', 
                color='black', fontsize=14, style='italic')

        # For the upper stack ({track_type}GhostTracks), place text above the bar
        ax1.text(bar2.get_x() + bar2.get_width() / 2, 
                bar2.get_y() + height2 + 25000,  # Adjust 0.05 for better visibility above the bar
                f"{(data_for_plot[f'{track_type}GhostRatio'].iloc[i])}%", 
                ha='center', va='center', 
                color='black', fontsize=14, style='italic')


    # Add a label for the first bar (base stack) with a green color
    ax1.text(bars1[0].get_x() + bars1[0].get_width() / 2, 
            0,  # Adjust 0.05 for visibility above the bar
            'Baseline', 
            ha='center', va='bottom', 
            color='black', fontsize=14, style='italic')

    ax2 = ax1.twinx()
    ax2.plot(data_for_plot['Threshold'], data_for_plot[f'{track_type}PseudoPurity'], label=f'{track_type}PseudoPurity', color='green', marker='o')
#     ax2.plot(data_for_plot['Threshold'], data_for_plot[f'{track_type}PseudoEfficiency'], label=f'{track_type}PseudoEfficiency', color='red', marker='o')


#     ax1.vlines(x=data_for_plot[data_for_plot['Threshold'] == '0.2,0.2']['Threshold'], 
#             ymin=-1e5, ymax=1e6, 
#             colors='blue', linestyles='dotted', 
#             linewidth=2.5, label='Recomended Threshold')

#     ax2.hlines(y=100, 
#             xmin=data_for_plot['Threshold'].min(), xmax=data_for_plot['Threshold'].max(),
#             colors='black', linestyles='dotted', 
#             linewidth=2.5, label='Baseline Efficiency and Purity')



    ax1.set_title('Working Point Optimisation', fontsize=18, style='italic')
    ax1.set_ylabel('Number of Tracks', fontsize=14)
    ax1.set_xlabel('Threshold', fontsize=14, style='normal')
   
    ax1.legend(loc='upper center', bbox_to_anchor=(0.15, -0.25), ncol=1, fontsize=16, labelspacing=1.5)

    ax1.grid(True)
    ax1.tick_params(axis='both', labelsize=14) 
    ax1.set_xticklabels(data_for_plot['Threshold'], rotation=45, ha='right')

    ax2.set_ylabel('KPI [%]', fontsize=14)
    ax2.tick_params(axis='both', labelsize=16) 

    ax2.legend(loc='upper center', bbox_to_anchor=(0.85, -0.25), ncol=1, fontsize=16, labelspacing=1.5)
  

    plt.tight_layout()

   
    plt.savefig(f"{directory}/{plot_name}.png",dpi=300,bbox_inches='tight')
    plt.show()