import numpy as np
import random
import matplotlib.pyplot as plt

def generate_data():
    # 生成随机数据
    np.random.seed(0)
    mean1 = [0, 0]
    cov1 = [[1, 0], [0, 1]]
    x1, y1 = np.random.multivariate_normal(mean1, cov1, 100).T
    mean2 = [5, 5]
    cov2 = [[1, 0], [0, 1]]
    x2, y2 = np.random.multivariate_normal(mean2, cov2, 100).T
    X = np.vstack((np.hstack((x1, y1)), np.hstack((x2, y2))))
    return X

def kmeans(X, k, max_iters=100):
    # X是数据集，k是聚类数，max_iters是最大迭代次数
    n_samples, n_features = X.shape
    centers = np.zeros((k, n_features))
    for i in range(k):
        centers[i] = X[random.randint(0, n_samples - 1)]
    for i in range(max_iters):
        # 分配样本到最近的中心点
        cluster_labels = np.zeros(n_samples)
        for j in range(n_samples):
            distances = np.linalg.norm(X[j] - centers, axis=1)
            cluster_labels[j] = np.argmin(distances)
        # 重新计算中心点
        for j in range(k):
            cluster_samples = X[cluster_labels == j]
            if len(cluster_samples) > 0:
                centers[j] = np.mean(cluster_samples, axis=0)
    return centers, cluster_labels

def plot_clusters(X, cluster_labels, centers):
    # 绘制聚类结果
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    fig, ax = plt.subplots()
    for i in range(len(centers)):
        cluster_samples = X[cluster_labels == i]
        ax.scatter(cluster_samples[:, 0], cluster_samples[:, 1], c=colors[i % len(colors)], marker='o', label='Cluster {}'.format(i))
    ax.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=150, linewidths=3, label='Centroids')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    X = generate_data()
    k = 2
    centers, cluster_labels = kmeans(X, k)
    plot_clusters(X, cluster_labels, centers)