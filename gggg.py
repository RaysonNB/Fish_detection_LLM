import matplotlib.pyplot as plt
import numpy as np

# Define the confusion matrix
cm = np.array([[13, 1, 0, 2, 10],
               [12, 10, 6, 8, 9],
               [23, 0, 9, 0, 7],
               [30, 6, 0, 40, 10],
               [20, 5, 0, 20, 3]])

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the confusion matrix
im = ax.imshow(cm, cmap='Blues')

# Add labels and title
ax.set_xticks(np.arange(len(cm[0])))
ax.set_yticks(np.arange(len(cm[:, 0])))
ax.set_xticklabels(['Thilapa\ntests', 'Abramis\nbrama_tests', 'Carassius\nauratus_tests', 'Cyprinidae\ntests', 'Hypophthalmichthys\ntests'], fontsize=8)
ax.set_yticklabels(['Hypophthalmichthys', 'Cyprinidae','Carassius\nauratus', 'Abramis\nbrama','Thilapa'], fontsize=10)
ax.set_title('Confusion matrix, without normalization')

# Add colorba
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('True label', rotation=-90, va="bottom")

# Annotate the values
for i in range(len(cm)):
    for j in range(len(cm[0])):
        text = ax.text(j, i, cm[i, j],
                      ha="center", va="center", color="w")

# Set the layout
plt.tight_layout()
plt.show()