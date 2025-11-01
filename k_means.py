from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

df = pd.read_csv("income.csv")
print("Dataset Preview: ")
print(df.head())
plt.scatter(df['Age'], df['Income($)'])
plt.show()

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])
print(y_predicted)

df['cluster'] = y_predicted
print(df.head())

df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

plt.scatter(df1.Age, df1['Income($)'], color = "green")
plt.scatter(df2['Age'], df2['Income($)'], color = 'red')
plt.scatter(df3['Age'], df3['Income($)'], color = 'black')

plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend(labels = ['Cluster1', 'Cluster2', 'Cluster3'])
plt.show()

scaler = MinMaxScaler()

scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])
print(df.head())

scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])
print(df.head())

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age', 'Income($)']])
print(y_predicted)

df['cluster'] = y_predicted
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
plt.scatter(df1.Age, df1['Income($)'], color = "green")
plt.scatter(df2['Age'], df2['Income($)'], color = 'red')
plt.scatter(df3['Age'], df3['Income($)'], color = 'black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1], color = 'purple', marker ='*', label = 'centroid')

plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend(labels = ['Cluster1', 'Cluster2', 'Cluster3', 'Centroid'])
plt.show()

print("Cluster Centroids", km.cluster_centers_)

k_range = range(1,10)
sse = []
for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)

print("sum squared  errors: ")
print(sse)

plt.xlabel('K')
plt.ylabel('Sum of Squared Error')
plt.plot(k_range,sse)
plt.show()