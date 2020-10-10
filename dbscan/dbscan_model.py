from collections import Counter

import matplotlib.pyplot as plt
import numpy as  np
from sklearn.neighbors.kd_tree import KDTree


class DBScanModel:
    def __init__(self, eps, min_samples, metric='euclidean'):
        """
        :param eps: Радиус, в котором для заданной точки рассматриваются
        её соседи
        :param min_samples: Минимальное число примеров в окрестности точки,
        чтобы не считаться outlier'ом.
        :param metric: Функция расстояния (из числа реализованных в sklearn, но
        можно предоставить и свою, совместимую с sklearn.neighbors.kd_tree.KDTree)
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.kd_tree = None
        self.cluster_assignments = None

    def fit(self, X):
        if type(X) is not np.ndarray:
            X = np.array(X)
        num_examples = X.shape[0]
        # По умолчанию номер кластера = -1. Если номер не поменялся, точка - outlier
        self.cluster_assignments = np.full(shape=num_examples, dtype=np.int, fill_value=-1)
        n_clusters = 0
        # Запоминаем те вершины, из которых ещё не смотрели на соседей.
        not_watched_points = set(range(len(X)))
        # Список соседей вершин, которые мы уже обходили. Нужен для того,
        # чтобы не плодить лишние кластеры в случайных блужданиях по точкам.
        # Обходя вершины этого списка, гарантируем, что эта точка либо останется
        # outlier'ом, либо примкнёт к уже созданному кластеру
        points_stack = []
        self.kd_tree = KDTree(X, metric=self.metric)
        while len(not_watched_points) > 0:
            # Если есть необойдённые соседи обойденных, идём в них
            if len(points_stack) > 0:
                point_id = points_stack.pop()
                # Если точку не обошли, смотрим её соседей
                if point_id in not_watched_points:
                    not_watched_points.remove(point_id)
                # Если же обошли, пропускаем
                else:
                    continue
            else:
                point_id = not_watched_points.pop()
            point = X[point_id]
            point_neighbors_ids_list = self.kd_tree.query_radius(point.reshape(1, -1), self.eps, )[0]
            point_neighbors_clusters = self.cluster_assignments[point_neighbors_ids_list]
            non_outliers_neighbors = point_neighbors_clusters[point_neighbors_clusters != -1]
            # Смотрим, есть ли в окрестности точки вершины с уже присвоенным кластером
            if len(non_outliers_neighbors) > 0:
                non_outliers_counter = Counter(non_outliers_neighbors)
                # Если да, то присваиваем новой вершине кластер большинства соседей
                point_cluster = non_outliers_counter.most_common(1)[0][0]
                self.cluster_assignments[point_id] = point_cluster
            # Создаём новый кластер, если точка не outlier:  если в её окрестности
            # хотя бы self.min_samples других точек
            elif len(point_neighbors_ids_list) > self.min_samples:

                point_cluster = n_clusters
                n_clusters += 1
                self.cluster_assignments[point_id] = point_cluster
            # Составляем список необойдённых соседей рассматриваемой точки
            points_to_traverse = [p for p in point_neighbors_ids_list if p in not_watched_points]
            points_stack.extend(points_to_traverse)

    def fit_predict(self, X):
        self.fit(X)
        return self.cluster_assignments


def main():
    np.random.seed(42)
    num_samples = 1000
    dataset = np.empty(shape=(num_samples, 2), dtype=np.float)
    num_clusters = 10
    samples_per_cluster = num_samples // num_clusters
    for cluster_id in range(num_clusters):
        # генерируем потенциальные кластеры как множество точек, чьи обе координаты
        # нормально распределены вокруг некоторой случайной точки из равномерного распределения
        center_x = np.random.uniform(100 * cluster_id, 100 * (cluster_id + 1))
        center_y = np.random.uniform(300 * (cluster_id % 3), 300 * (cluster_id % 3 + 1))

        for i in range(samples_per_cluster):
            noize_x = np.random.normal(0, 30)
            noize_y = np.random.normal(0, 30)

            dataset[cluster_id * samples_per_cluster + i][0] = center_x + noize_x
            dataset[cluster_id * samples_per_cluster + i][1] = center_y + noize_y
    np.random.shuffle(dataset)
    model = DBScanModel(eps=25, min_samples=3, metric='euclidean')
    model.fit(dataset)
    clusters = model.fit_predict(dataset)
    plt.scatter(dataset[:, 0], dataset[:, 1], c=clusters)
    plt.title('Artificial data')
    plt.show()


if __name__ == '__main__':
    main()
