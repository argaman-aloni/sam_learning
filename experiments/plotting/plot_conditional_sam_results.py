from itertools import cycle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
file_path = "compressed_results.csv"
df = pd.read_csv(file_path)


relevant_labels = ["solved", "no-solution", "timeout", "solver-error"]
color_palette = sns.color_palette("colorblind", n_colors=len(relevant_labels))
line_styles = ["-", "--", "-.", ":", "--", "-.", ":"]
color_cycle = cycle(color_palette)

label_to_legend = {
    "solved": "Solved",
    "no-solution": "No Solution",
    "timeout": "Timeout",
    "solver-error": "Solver Error",
}


domains = df["domain"].unique()
fig, axs = plt.subplots(1, len(domains), sharey=True)
fig.set_size_inches(30, 5)
axs[0].set_ylabel("% of Problems", fontsize=24)
for dom_index, domain in enumerate(domains):
    domain_data = df[df["domain"] == domain]
    for index, label in enumerate(relevant_labels):
        axs[dom_index].plot(
            domain_data["num_trajectories"],
            domain_data[label],
            label=label_to_legend[label],
            color=color_palette[index],
            linestyle=line_styles[index],
            linewidth=3,
        )
        axs[dom_index].set_title(domain, fontsize=24)
        axs[dom_index].yaxis.set_tick_params(labelsize=20)
        axs[dom_index].xaxis.set_tick_params(labelsize=20)
        axs[dom_index].set_xlabel("# Trajectories", fontsize=24)


plt.tight_layout()
plt.legend(loc=0, mode="expand",  fontsize=20, bbox_transform=fig.transFigure, framealpha=0.4)
plt.show()


if __name__ == "__main__":
    pass
