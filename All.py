import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.DataFrame({
    'Marks': [35, 45, 55, 65, 75]
})

print("Original Data:")
print(df)

minmax_scaler = MinMaxScaler()
df['MinMax_Scaled'] = minmax_scaler.fit_transform(df[['Marks']])

standard_scaler = StandardScaler()
df['Standard_Scaled'] = standard_scaler.fit_transform(df[['Marks']])

print("\nData After Scaling:")
print(df)

-----------------------------------------------------------------

import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

x1 = float(input("Enter input x1: "))
x2 = float(input("Enter input x2: "))

w1 = float(input("Enter initial weight w1: "))
w2 = float(input("Enter initial weight w2: "))

b = float(input("Enter initial bias: "))

target = float(input("Enter target output: "))

learning_rate = float(input("Enter learning rate: "))

net = x1 * w1 + x2 * w2 + b
output = sigmoid(net)

error = target - output
delta = error * sigmoid_derivative(output)

w1 = w1 + learning_rate * delta * x1
w2 = w2 + learning_rate * delta * x2
b = b + learning_rate * delta

print("\n----- Results -----")
print(f"Neuron Output          : {output:.4f}")
print(f"Error                  : {error:.4f}")
print(f"Updated Weight w1      : {w1:.4f}")
print(f"Updated Weight w2      : {w2:.4f}")
print(f"Updated Bias           : {b:.4f}")

--------------------------------------------------------

import math
import matplotlib.pyplot as plt

points = [(2,10), (2,5), (8,4), (5,8),
          (7,5), (6,4), (1,2), (4,9)]

k = int(input("Enter number of clusters (K): "))

centroids = []
print("Enter initial centroids:")
for i in range(k):
    x = float(input(f"Centroid {i+1} X: "))
    y = float(input(f"Centroid {i+1} Y: "))
    centroids.append((x, y))

while True:
    clusters = [[] for _ in range(k)]

    for point in points:
        distances = []
        for centroid in centroids:
            distance = math.sqrt((point[0]-centroid[0])**2 +
                                 (point[1]-centroid[1])**2)
            distances.append(distance)

        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(point)

    new_centroids = []
    for cluster in clusters:
        if len(cluster) == 0:
            new_centroids.append((0, 0))
        else:
            x_mean = sum(p[0] for p in cluster) / len(cluster)
            y_mean = sum(p[1] for p in cluster) / len(cluster)
            new_centroids.append((x_mean, y_mean))

    if new_centroids == centroids:
        break

    centroids = new_centroids

print("\nFinal Clusters:")
for i in range(k):
    print(f"Cluster {i+1}: {clusters[i]}")

print("\nFinal Centroids:")
for i in range(k):
    print(f"Centroid {i+1}: {centroids[i]}")

colors = ['red', 'blue', 'green', 'purple', 'orange',
          'brown', 'pink', 'cyan', 'magenta', 'yellow']

plt.figure(figsize=(8,6))

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

-------------------------------------------------------------

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

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

labels = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']

Z = linkage(points, method='single')  

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

-------------------------------------------------------

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

points = [
    (3,7), 
    (4,6), 
    (5,5), 
    (6,4), 
    (7,3), 
    (6,2), 
    (7,2),
    (8,4)   
]

labels = ['A','B','C','D','E','F','G','H']

db = DBSCAN(eps=2.0, min_samples=2)
db.fit(points)

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

----------------------------------------------------------------

from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

points = [
    (2,10),
    (2,5),
    (8,4),
    (5,8),
    (7,5),
    (6,4)
]

labels = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']

model = AgglomerativeClustering(n_clusters=2, linkage='single')
clusters = model.fit_predict(points)

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