import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {
    "X":[2,2,8,5,7,6,1,4],
    "Y":[10,5,4,8,5,4,2,9]
}

df = pd.DataFrame(data)
csv_path = "data_for_kmeans.csv"
df.to_csv(csv_path, index  = False)

df = pd.read_csv("data_for_kmeans.csv")
X = df.to_numpy()

centroids = np.array([(2,10),(5,8),(1,2)])

def euclidean(a,b):
    return np.sqrt(np.sum((a - b)**2))

def kmeans(X, centroids, iterations = 10):
    K = len(centroids)
    for i in range(iterations):
        clusters = [[] for _ in range(K)]
        for point in X:
            distances = [euclidean(point,c) for c in centroids]
            clusters[np.argmin(distances)].append(point)
        
        clusters = [np.array(c) for c in clusters]

        new_centroids = []
        for i,cluster in enumerate(clusters):
            if len(cluster)>0:
                new_centroids.append(np.mean(cluster,axis = 0))
            else:
                new_centroids.append(centroids[i])

        new_centroids = np.array(new_centroids)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids
    
    return centroids, clusters

final_centroids, clusters = kmeans(X, centroids)
print("Final Centroids:\n", final_centroids)

print("\nClusters:")
for i, cluster in enumerate(clusters):
    print(f"\nCluster {i+1}:")
    print(cluster)

colors = ['red','green','blue']
for i, cluster in enumerate(clusters):
    if len(cluster)>0:
        plt.scatter(cluster[:,0],cluster[:,1],label = f"Cluster {i+1}",color=colors[i],s=80)

plt.scatter(final_centroids[:,0],final_centroids[:,1],marker = 'X',s=200,color = 'black',label = "centroids")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-Means Final Clusters")
plt.legend()
plt.grid(True)
plt.show()






