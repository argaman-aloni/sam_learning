import sys
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_results(results_directory_path: Path):
    """Plot the results of the experiments."""
    for file_path in results_directory_path.glob('*solving_aggregated*.csv'):
        df = pd.read_csv(file_path)

        # Define color-blind friendly palette
        color_palette = sns.color_palette("colorblind", n_colors=len(df['learning_algorithm'].unique()))
        line_styles = cycle(['-', '--', '-.', ':', '--', '-.', ':'])
        color_cycle = cycle(color_palette)
        markers = cycle(['o', 's', 'D', 'v', '^', '>', '<', 'p', 'P', '*', 'X', 'd'])

        # Group the data by 'num_trajectories', 'learning_algorithm' and calculate the mean and std of 'percent_ok'
        df = df[df['learning_algorithm'] != "incremental_nsam"]  # Remove max_percent_ok from the plot
        grouped_data = df.groupby(['num_trajectories', 'learning_algorithm']).agg(
            avg_max_percent_ok=('max_percent_ok', 'mean'),
            std_max_percent_ok=('max_percent_ok', 'std'),
            goal_not_achieved=('percent_goal_not_achieved', 'first')).reset_index()

        labels = {
            "numeric_sam": "NSAM*",
            "naive_nsam": "NSAM",
        }

        # Plotting
        sns.set(style='whitegrid')
        plt.figure(figsize=(10, 6))

        # Plot a line for each learning algorithm
        for algo in df['learning_algorithm'].unique():
            algo_data = grouped_data[grouped_data['learning_algorithm'] == algo]
            plt.plot(algo_data['num_trajectories'],
                         algo_data['avg_max_percent_ok'],
                         linestyle=next(line_styles),
                         label=labels[algo],
                         marker=next(markers),
                         color=next(color_cycle),
                         linewidth=3)

            # Plot standard deviation as shaded area around the mean line
            plt.fill_between(algo_data['num_trajectories'],
                             np.clip(algo_data['avg_max_percent_ok'] - algo_data['std_max_percent_ok'], 0, 100),
                             np.clip(algo_data['avg_max_percent_ok'] + algo_data['std_max_percent_ok'], 0, 100),
                             alpha=0.2)

        # Set plot labels and title
        plt.xlabel('# Observations', fontsize=24)
        plt.ylabel('AVG % of solved', fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(ticks=list(range(0, 101, 10)),fontsize=24)

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
