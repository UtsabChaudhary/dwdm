from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Data points
points = [
    (3,7),  # A
    (4,6),  # B
    (5,5),  # C
    (6,4),  # D
    (7,3),  # E
    (6,2),  # F
    (7,2),  # G
    (8,4)   # H
]

labels = ['A','B','C','D','E','F','G','H']

# Apply DBSCAN
# Change eps and min_samples if needed
db = DBSCAN(eps=2.0, min_samples=2)
db.fit(points)

# Core points
core_indices = set(db.core_sample_indices_)

core_points = []
border_points = []
noise_points = []

for i in range(len(points)):
    if db.labels_[i] == -1:
        noise_points.append(labels[i])
    elif i in core_indices:
        core_points.append(labels[i])
    else:
        border_points.append(labels[i])

print("Core Points:", core_points)
print("Border Points:", border_points)
print("Noise Points:", noise_points)

# -----------------------------
# Plot the result
# -----------------------------
colors = ['red', 'blue', 'green', 'purple']

plt.figure(figsize=(7,6))

for i, point in enumerate(points):
    if labels[i] in core_points:
        plt.scatter(point[0], point[1], c='blue', s=150, marker='o')
    elif labels[i] in border_points:
        plt.scatter(point[0], point[1], c='green', s=150, marker='s')
    else:
        plt.scatter(point[0], point[1], c='red', s=150, marker='x')

    plt.text(point[0]+0.05, point[1]+0.05, labels[i], fontsize=12)

plt.title("DBSCAN Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

plt.scatter([], [], c='blue', marker='o', label='Core Point')
plt.scatter([], [], c='green', marker='s', label='Border Point')
plt.scatter([], [], c='red', marker='x', label='Noise Point')

plt.legend()
plt.show()