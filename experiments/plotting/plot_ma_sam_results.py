import sys
from pathlib import Path
from itertools import cycle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_results(results_directory_path: Path):
    """Plot the results of the experiments."""
    for file_path in results_directory_path.glob('*solving_aggregated*.csv'):
        df = pd.read_csv(file_path)

        algorithms = df['learning_algorithm'].unique()
        # Define color-blind friendly palette
        color_palette = sns.color_palette("colorblind", n_colors=len(df['learning_algorithm'].unique()))
        line_styles = cycle(['-', '--', '-.', ':', '--', '-.', ':'])
        color_cycle = cycle(color_palette)
        markers = cycle(['o', 's', 'D', 'v', '^', '>', '<', 'p', 'P', '*', 'X', 'd'])

        # Group the data by 'num_trajectories', 'learning_algorithm' and calculate the mean and std of 'percent_ok'
        labels_algorithm = {
            "ma_sam": "MA-SAM",
            "sam_learning": "SAM Learning",
            "sam_learning_soft": "SAM Learning - Soft",
            "sam_learning_hard": "SAM Learning - Hard",
            "ma_sam_soft": "MA-SAM - Soft",
            "ma_sam_hard": "MA-SAM - Hard",
        }

        grouped_data = df.groupby(['policy', 'num_trajectories', 'learning_algorithm']).agg(
            avg_percent_ok=('percent_ok', 'mean'),
            std_percent_ok=('percent_ok', 'std')).reset_index()

        # Plotting
        sns.set(style='whitegrid')
        plt.figure(figsize=(10, 6))

        # Plot a line for each learning algorithm
        for algo in algorithms:
            algo_data = grouped_data[grouped_data['learning_algorithm'] == algo]
            plt.errorbar(algo_data['num_trajectories'],
                         algo_data['avg_percent_ok'],
                         linestyle=next(line_styles),
                         label=labels_algorithm[algo],
                         marker=next(markers),
                         color=next(color_cycle),
                         lolims=True,
                         linewidth=3)

            # Plot standard deviation as shaded area around the mean line
            plt.fill_between(algo_data['num_trajectories'],
                             np.clip(algo_data['avg_percent_ok'] - algo_data['std_percent_ok'], 0, 100),
                             np.clip(algo_data['avg_percent_ok'] + algo_data['std_percent_ok'], 0, 100),
                             alpha=0.2)

        # Set plot labels and title
        plt.xlabel('# Observations', fontsize=24)
        plt.ylabel('AVG % of solved', fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(ticks=list(range(0, 101, 10)), fontsize=24)

        # Add a legend
        plt.legend(fontsize=24)
        plt.grid(True)

        output_file_path = file_path.parent / f'{file_path.stem}_plot.png'
        plt.savefig(output_file_path, bbox_inches='tight')

        # Show the plot
        plt.show()

if __name__ == '__main__':
    results_path = Path(sys.argv[1])
    plot_results(results_path)
