from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Data points
points = [
    (2,10),
    (2,5),
    (8,4),
    (5,8),
    (7,5),
    (6,4)
]

# Labels
labels = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']

# Apply Agglomerative Clustering
model = AgglomerativeClustering(n_clusters=2, linkage='single')
clusters = model.fit_predict(points)

# Display clusters
cluster1 = []
cluster2 = []

print("Point\tCluster")
for i in range(len(points)):
    print(f"{labels[i]} {points[i]}\t{clusters[i]+1}")

    if clusters[i] == 0:
        cluster1.append(points[i])
    else:
        cluster2.append(points[i])

print("\nCluster 1:", cluster1)
print("Cluster 2:", cluster2)

# Plot the clusters
colors = ['red', 'blue']

plt.figure(figsize=(7,6))

for i, point in enumerate(points):
    plt.scatter(point[0], point[1],
                color=colors[clusters[i]],
                s=120)
    plt.text(point[0]+0.1, point[1]+0.1,
             labels[i], fontsize=11)

plt.title("Agglomerative Clustering (2 Clusters)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()