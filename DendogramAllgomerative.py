import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Data points
points = [
    (2,10),
    (2,5),
    (8,4),
    (5,8),
    (7,5),
    (6,4),
    (1,2),
    (4,9)
]

# Labels for each point
labels = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']

# Perform Agglomerative Hierarchical Clustering
Z = linkage(points, method='single')   # You can also use 'complete', 'average', or 'ward'

# Plot dendrogram
plt.figure(figsize=(8,6))
dendrogram(
    Z,
    labels=labels,
    leaf_rotation=90,
    leaf_font_size=12
)

plt.title("Dendrogram using Agglomerative Clustering")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.grid(True)
plt.show()