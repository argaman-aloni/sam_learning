import sys
from pathlib import Path
from itertools import cycle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main(results_directory_path: Path):
    """Plot the results of the experiments."""
    for file_path in results_directory_path.glob('*.csv'):
        df = pd.read_csv(file_path)

        # Define color-blind friendly palette
        color_palette = sns.color_palette("colorblind", n_colors=len(df['learning_algorithm'].unique()))
        line_styles = cycle(['-', '--', '-.', ':', '--', '-.', ':'])
        color_cycle = cycle(color_palette)

        # Group the data by 'num_trajectories', 'learning_algorithm' and calculate the mean and std of 'percent_ok'
        grouped_data = df.groupby(['num_trajectories', 'learning_algorithm'])['percent_ok'].agg(
            ['mean', 'std']).reset_index()

        labels = {
            "numeric_sam": "NSAM* with RV",
            "raw_numeric_sam": "NSAM* without RV",
            "naive_nsam": "NSAM with RV",
            "raw_naive_nsam": "NSAM without RV",
            "polynomial_sam": "NSAM* with RV",
            "raw_polynomial_nsam": "NSAM* without RV",
            "naive_polysam": "NSAM with RV",
            "raw_naive_polysam": "NSAM without RV",
        }

        # Plotting
        sns.set(style='whitegrid')
        plt.figure(figsize=(10, 6))

        # Plot a line for each learning algorithm
        for algo in df['learning_algorithm'].unique():
            algo_data = grouped_data[grouped_data['learning_algorithm'] == algo]
            plt.plot(algo_data['num_trajectories'],
                     algo_data['mean'],
                     linestyle=next(line_styles),
                     label=labels[algo],
                     color=next(color_cycle),
                     linewidth=3)

            # Plot standard deviation as shaded area around the mean line
            plt.fill_between(algo_data['num_trajectories'],
                             np.clip(algo_data['mean'] - algo_data['std'], 0, 100),
                             np.clip(algo_data['mean'] + algo_data['std'], 0, 100),
                             alpha=0.2)

        # Set plot labels and title
        plt.xlabel('# Observations', fontsize=16)
        plt.ylabel('Average % of problems solved', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # Add a legend
        plt.legend(fontsize=16)
        plt.grid(True)

        output_file_path = file_path.parent / f'{file_path.stem}_plot.png'
        plt.savefig(output_file_path, bbox_inches='tight')

        # Show the plot
        plt.show()


if __name__ == '__main__':
    results_path = Path(sys.argv[1])
    main(results_path)
