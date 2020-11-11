import numpy

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


class KmeansModel:
    def __init__(self, n_clusters=8, n_init=10, num_iter=100, random_state=None):
        """
        :param n_clusters: Число кластеров
        :param n_init: Число случайных начальных инициализаций центроидов
        :param num_iter: Число итераций оптимизации центроидов для каждой
        из начальных инициализаций центроидов
        :param random_state: Состояние генератора случайных чисел
        """
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.n_clusters = n_clusters
        self.num_iter = num_iter
        self.cluster_assignments = None
        self.centroids = None
        np.random.seed(random_state)

    def fit(self, X):
        if type(X) is not np.ndarray:
            X = np.array(X)
        num_examples = X.shape[0]
        cluster_distances = np.empty(shape=self.n_clusters, dtype=np.float)
        cluster_assignments = np.empty(shape=num_examples, dtype=np.int)
        best_overall_loss = None
        best_overall_centroids = None
        # Эпохи различных начальных инициализаций цетроидов
        for epoch in tqdm(range(self.n_init)):
            best_epoch_loss = None
            best_epoch_centroids = None

            # Выбираем рандомные точки из данных как начальные центроиды
            random_row_ids = np.random.choice(num_examples, size=self.n_clusters, replace=False)
            centroids = X[random_row_ids, :]

            for iteration_id in range(self.num_iter):
                iteration_loss = 0
                for training_sample_id, data_example in enumerate(X):
                    # Считаем расстояние от каждой точки до центроида
                    cluster_distances[:] = ((centroids - data_example) ** 2).sum(axis=1) ** 0.5
                    iteration_loss += cluster_distances.min()
                    # Выбираем минимальное от точки до центроида - присваиваем
                    # точке принадлежность ближайшему кластеру
                    cluster_assignments[training_sample_id] = cluster_distances.argmin()
                for cluster_id in range(self.n_clusters):
                    centroids[cluster_id] = X[cluster_assignments == cluster_id].mean(axis=0)
                # Если на этой итерации внутри эпохи центроиды стали лучше, запоминаем их
                if best_epoch_loss is None or iteration_loss < best_epoch_loss:
                    best_epoch_loss = iteration_loss
                    best_epoch_centroids = centroids
            # Если на этой эпохе центроиды стали лучше, запоминаем их
            if best_overall_loss is None or best_epoch_loss < best_overall_loss:
                best_overall_loss = best_epoch_loss
                best_overall_centroids = best_epoch_centroids
        self.centroids = best_overall_centroids

    def predict(self, X):
        if type(X) is not np.ndarray:
            X = np.array(X)
        num_examples = X.shape[0]
        cluster_distances = np.empty(shape=self.n_clusters, dtype=np.float)
        cluster_assignments = np.empty(shape=num_examples, dtype=np.int)
        for training_sample_id, data_example in enumerate(X):
            # Считаем расстояние от каждой точки до центроида
            cluster_distances[:] = ((self.centroids - data_example) ** 2).sum(axis=1) ** 0.5
            cluster_assignments[training_sample_id] = cluster_distances.argmin()
        self.cluster_assignments = cluster_assignments
        return cluster_assignments


def is_point_far_enough_from_others(point, other_points, min_distance):
    x, y = point
    for (other_x, other_y) in other_points:
        distance = ((x - other_x) * (x - other_x) + (y - other_y) * (y - other_y)) ** 0.5
        if distance < min_distance:
            return False
    return True


def main():
    np.random.seed(42)
    num_samples = 1000
    num_variables = 2
    scale_x = 100
    scale_y = 300
    x_cluster_center_distance_std = 30
    y_cluster_center_distance_std = 30
    num_clusters = 10
    dataset = numpy.empty(shape=(num_samples, num_variables), dtype=np.float)
    samples_per_cluster = num_samples // num_clusters
    for cluster_id in range(num_clusters):
        # генерируем потенциальные кластеры как множество точек, чьи обе координаты
        # нормально распределены вокруг некоторой случайной точки из равномерного распределения
        center_x = np.random.uniform(scale_x * cluster_id, scale_x * (cluster_id + 1))
        center_y = np.random.uniform(scale_y * (cluster_id % 3), scale_y * (cluster_id % 3 + 1))

        for i in range(samples_per_cluster):
            noize_x = np.random.normal(0, x_cluster_center_distance_std)
            noize_y = np.random.normal(0, y_cluster_center_distance_std)

            dataset[cluster_id * samples_per_cluster + i][0] = center_x + noize_x
            dataset[cluster_id * samples_per_cluster + i][1] = center_y + noize_y
    np.random.shuffle(dataset)
    num_model_clusters = 10
    num_init = 50
    num_iter = 25
    model = KmeansModel(n_clusters=num_model_clusters, n_init=num_init, num_iter=num_iter)
    model.fit(dataset)
    clusters = model.predict(dataset)
    plt.scatter(dataset[:, 0], dataset[:, 1], c=clusters)
    plt.title('Artificial data')
    plt.show()


if __name__ == '__main__':
    main()
