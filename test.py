import matplotlib.pyplot as plt

# Accuracy data for different fish species
accuracies1 = {
    'Abramis_brama': 87.27,
    'Cyprinidae': 86.36,
    'Hypophyhalmichthys': 47.27,
    'Carassius_auratus': 54.55,
    'Thilapa_fish': 77.27
}

accuracies2 = {
    'Abramis_brama2': 95.09,
    'Cyprinidae2': 100,
    'Hypophyhalmichthys2': 54.83,
    'Carassius_auratus2': 62.55,
    'Thilapa_fish2': 84.61
}

# Calculate the average accuracy for both sets
average_accuracy1 = sum(accuracies1.values()) / len(accuracies1)
average_accuracy2 = sum(accuracies2.values()) / len(accuracies2)

# Create a list of fish species and their corresponding accuracies
fish_species = list(accuracies1.keys())
accuracies_list1 = list(accuracies1.values())
accuracies_list2 = list(accuracies2.values())

# Set up the figure and axis
plt.figure(figsize=(12, 8))

# Define the width of the bars
bar_width = 0.35

# Create a bar chart for both sets of data
bars1 = plt.bar(range(len(fish_species)), accuracies_list1, width=bar_width, label='Before', color='skyblue')
bars2 = plt.bar([i + bar_width for i in range(len(fish_species))], accuracies_list2, width=bar_width, label='After', color='orange')

# Add horizontal dashed lines for the average accuracy of both sets
plt.axhline(y=average_accuracy1, color='red', linestyle='--', linewidth=2, label=f'Average Accuracy Set 1: {average_accuracy1:.2f}')
plt.axhline(y=average_accuracy2, color='green', linestyle='--', linewidth=2, label=f'Average Accuracy Set 2: {average_accuracy2:.2f}')

# Add title and labels
plt.title('Accuracy of Fish Species Identification (Five Common type of fishes in Macau) By Gemini')
plt.xlabel('Fish Species / Name')
plt.ylabel('Accuracy / %')

# Set xticks and labels
plt.xticks([i + bar_width / 2 for i in range(len(fish_species))], fish_species)

# Write the accuracy on each bar
for bar, acc in zip(bars1, accuracies_list1):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{acc:.2f}%', ha='center', va='bottom')

for bar, acc in zip(bars2, accuracies_list2):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{acc:.2f}%', ha='center', va='bottom')

# Add a legend
plt.legend()

# Show the plot
plt.ylim(0, 100)  # Set y-axis range
plt.tight_layout()  # Adjust layout to make room for the labels
plt.show()