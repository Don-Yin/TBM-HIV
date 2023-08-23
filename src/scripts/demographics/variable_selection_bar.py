import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

save_to = Path("results", "images", "demographics")
save_to.mkdir(parents=True, exist_ok=True)


steps = [
    "Initial variables",
    "Cutting irrelevant variables:",
    "Cutting variables without data file:",
    "Cutting variables not in data:",
    "Cutting systematic missing variables:",
    "Cutting variables with < 50% data available",
]

variables_left = [387, 209, 88, 84, 70, 70]  # variables left at each step

plt.figure(figsize=(10, 6))
sns.barplot(x=variables_left, y=steps, color="grey")  # Use a single color here
plt.xlabel("Number of Variables Left")
plt.ylabel("Selection Process")
plt.title("Variable Selection Process for TBM Prognosis Analysis")
plt.xlim(0, max(variables_left) * 1.1)  # Ensure the x-axis fits all the labels
plt.tight_layout()

# Align labels to the left
ax = plt.gca()
ax.tick_params(axis="y", left=False)  # Remove y-ticks
ax.yaxis.label.set_visible(False)  # Hide y-label
for patch in ax.patches:
    x_val = patch.get_width()
    y_val = patch.get_y() + patch.get_height() / 2
    left_margin = max(variables_left) * 0.02  # A small percentage of the total width
    ax.text(left_margin, y_val, int(x_val), va="center")

# -------- arrow --------
arrow_x = -max(variables_left) * 0.1  # X position for the arrow (left of the graph)
arrow_y_start = len(steps) - 0.5  # Y starting position (top of the graph)
arrow_y_end = -0.5  # Y ending position (bottom of the graph)

plt.annotate(
    "Order",
    xy=(arrow_x, arrow_y_end),
    xytext=(arrow_x, arrow_y_start),
    arrowprops=dict(arrowstyle="->", lw=2),
    va="center",
    ha="center",
    fontsize=12,
)


plt.savefig(save_to / "variable_selection_process.png", dpi=420)
