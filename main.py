import sklearn.cluster
import sklearn.metrics.cluster
from sklearn.metrics import davies_bouldin_score
import numpy as np
import pandas as pd
import matplotlib.pyplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
from kneed import KneeLocator


def get_min_samples(X1, center, optEp):
    dbscan = DBSCAN(eps=optEp,min_samples=center)
    model = dbscan.fit(X1)
    score = silhouette_score(X1, model.labels_, metric='euclidean')
    return score


tripadvisor1 = pd.read_csv('Cust_Segmentation.csv')
tripadvisor = tripadvisor1.drop('Address', axis=1)
colors = ['purple', 'red', 'blue', 'green', 'yellow', 'orange', 'cyan', 'magenta', 'black', 'brown']

# K-Means
X = tripadvisor.values[:, 1:] # 1-ый столбец не нужен
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init= 'k-means++', random_state= 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) # Показывает расстояние между точками и центроидами

X_max = int(input("Max count of classes: "))
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1, X_max), timings= True)
visualizer.fit(X)        # Fit data to visualizer
visualizer.show()

results = {}
for i in range(2, X_max):
    kmeans = KMeans(n_clusters=i, random_state=30)
    labels = kmeans.fit_predict(X)
    db_index = davies_bouldin_score(X, labels)
    results.update({i: db_index})
plt.plot(list(results.keys()), list(results.values()))
plt.title("Davies-Boulding for K-means")
plt.xlabel("Number of clusters")
plt.ylabel("Davies-Boulding Index")
plt.show()
plt.clf()


clusterNum = int(input("Put optimal count of clusters : ")) # Кол-во кластеров
k_means = KMeans(init = "k-means++", n_clusters=clusterNum, n_init=12) # Создаём объект k-means, импортированный
# из библиотеки, вызываем его конструктор и задаём количество кластеров

k_means.fit(X) # Передаём в метод fit наш переделанный массив
labels = k_means.labels_


tripadvisor["Clus_km"] = labels # Добавление новой колонки
print("(1) Age, (2) Education, (3) Years Employed, (4) Income, (5) Other Debt")
AxX = int(input("Какой критерий будет на оси X? : "))
AA = X[:, 0]
AA1 = X[:, 1]
AA2 = X[:, 3]
match(AxX):
    case 1:
        AA1 = X[:, 0]
        XL = 'Age'
        xXx = 0
    case 2:
        AA1 = X[:, 1]
        XL = 'Education'
        xXx = 1
    case 3:
        AA1 = X[:, 2]
        XL = 'Years Employed'
        xXx = 2
    case 4:
        AA1 = X[:, 3]
        XL = 'Income'
        xXx = 3
    case 5:
        AA1 = X[:, 4]
        XL = 'Other Debt'
        xXx = 4

AxY = int(input("Какой критерий будет на оси Y? : "))
match(AxY):
    case 1:
        AA2 = X[:, 0]
        YL = 'Age'
        yYy = 0
    case 2:
        AA2 = X[:, 1]
        YL = 'Education'
        yYy = 1
    case 3:
        AA2 = X[:, 2]
        YL = 'Years Employed'
        yYy = 2
    case 4:
        AA2 = X[:, 3]
        YL = 'Income'
        yYy = 3
    case 5:
        AA2 = X[:, 4]
        YL = 'Other Debt'
        yYy = 4

plt.ylabel(YL, fontsize=16)
area = np.pi * (X[:, 1])**2
plt.scatter(AA1, AA2, s=area, c=labels.astype(float), alpha=0.5)
plt.xlabel(XL, fontsize=18)
plt.title("K-means")
plt.show()
plt.clf()

# Agglomerative Clustering
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
plt.clf()

model = AgglomerativeClustering()
visualizer = KElbowVisualizer(model, k=(1, X_max), timings= False)
visualizer.fit(X)        # Fit data to visualizer
visualizer.show()

results1 = {}
for i in range(2, X_max):
    Agg1 = AgglomerativeClustering(n_clusters=i)
    labels1 = Agg1.fit_predict(X)
    db_index1 = davies_bouldin_score(X, labels1)
    results1.update({i: db_index1})
plt.plot(list(results1.keys()), list(results1.values()))
plt.title("Davies-Bouling for Agglomerative CLustering")
plt.xlabel("Number of clusters")
plt.ylabel("Davies-Boulding Index")
plt.show()
plt.clf()

clusterNum1 = int(input("Put optimal count of clusters : ")) # Кол-во кластеров

hc = AgglomerativeClustering(n_clusters=clusterNum1, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

for i in range(clusterNum1):
    plt.scatter(X[y_hc == i, xXx], X[y_hc == i, yYy], s=100, c=colors[i], label='Cluster ' + str(i+1))
plt.title('Agglomerative Clusters')
plt.xlabel(XL)
plt.ylabel(YL)
plt.legend()
plt.show()
plt.clf()

# DBSCAN
neigh = NearestNeighbors(n_neighbors=11)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X) # Для минимального расстояния между точками
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
i1 = np.arange(len(distances))
knee = KneeLocator(i1, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
fig = plt.figure(figsize=(5, 5))
knee.plot_knee()
plt.xlabel("Points")
plt.ylabel("Distance")
plt.title("For Epsilon")
print("Optimal eps = " + str(distances[knee.knee]))
plt.show()
plt.clf()
optEps = float(input("Put optimal eps = "))

scores = []
centers = list(range(2, 30))
for center in centers:
    scores.append(get_min_samples(X, center, optEps))
plt.plot(centers, scores)
plt.xlabel('min_samples')
plt.ylabel('Silhouette Score')
plt.title("For min_samples")
df3 = pd.DataFrame(centers, columns=['min_samples'])
df3['scores'] = scores
df4 = df3[df3.scores == df3.scores.max()]
print('Optimal min_sample = ' + str(df4['min_samples'].tolist()))
plt.show()
plt.clf()
optSamples = int(input("Put min_samples count = "))

dbscan_optimal = DBSCAN(eps=optEps, min_samples=optSamples)
dbscan_optimal.fit(X)
tripadvisor['DBSCAN_optimal_labels']=dbscan_optimal.labels_
tripadvisor['DBSCAN_optimal_labels'].value_counts()

plt.figure(figsize=(5, 5))
plt.scatter(AA1, AA2, c=tripadvisor['DBSCAN_optimal_labels'], cmap=matplotlib.colors.ListedColormap(colors), s=15)
plt.title('DBSCAN', fontsize=10)
plt.xlabel(XL)
plt.ylabel(YL)
plt.title("DBSCAN")
plt.show()
plt.clf()