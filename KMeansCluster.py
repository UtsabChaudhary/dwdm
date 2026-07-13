import math
import matplotlib.pyplot as plt

# Data points
points = [(2,10), (2,5), (8,4), (5,8),
          (7,5), (6,4), (1,2), (4,9)]

# Number of clusters
k = int(input("Enter number of clusters (K): "))

# Input initial centroids
centroids = []
print("Enter initial centroids:")
for i in range(k):
    x = float(input(f"Centroid {i+1} X: "))
    y = float(input(f"Centroid {i+1} Y: "))
    centroids.append((x, y))

while True:
    clusters = [[] for _ in range(k)]

    # Assign points to nearest centroid
    for point in points:
        distances = []
        for centroid in centroids:
            distance = math.sqrt((point[0]-centroid[0])**2 +
                                 (point[1]-centroid[1])**2)
            distances.append(distance)

        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(point)

    # Calculate new centroids
    new_centroids = []
    for cluster in clusters:
        if len(cluster) == 0:
            new_centroids.append((0, 0))
        else:
            x_mean = sum(p[0] for p in cluster) / len(cluster)
            y_mean = sum(p[1] for p in cluster) / len(cluster)
            new_centroids.append((x_mean, y_mean))

    # Stop if centroids do not change
    if new_centroids == centroids:
        break

    centroids = new_centroids

# Display results
print("\nFinal Clusters:")
for i in range(k):
    print(f"Cluster {i+1}: {clusters[i]}")

print("\nFinal Centroids:")
for i in range(k):
    print(f"Centroid {i+1}: {centroids[i]}")

# -------------------------
# Plot Cluster Diagram
# -------------------------
colors = ['red', 'blue', 'green', 'purple', 'orange',
          'brown', 'pink', 'cyan', 'magenta', 'yellow']

plt.figure(figsize=(8,6))

# Plot each cluster
for i in range(k):
    if len(clusters[i]) > 0:
        x = [p[0] for p in clusters[i]]
        y = [p[1] for p in clusters[i]]
        plt.scatter(x, y,
                    color=colors[i % len(colors)],
                    s=100,
                    label=f'Cluster {i+1}')

# Plot centroids
for i, centroid in enumerate(centroids):
    plt.scatter(centroid[0], centroid[1],
                color='black',
                marker='X',
                s=250,
                edgecolors='white',
                label=f'Centroid {i+1}')

plt.title("K-Means Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
plt.show()