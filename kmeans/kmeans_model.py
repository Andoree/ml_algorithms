import random

import numpy as np


class KmeansModel:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels = None

    def fit(self, X):
        if type(X) is not np.ndarray:
            X = np.array(X)
        num_examples = X.shape[0]
        cluster_distances = np.empty(shape=self.n_clusters, dtype=np.float)
        cluster_assignments = np.empty(shape=num_examples, dtype=np.int)
        for epoch in range(self.n_init):
            random_row_ids = np.random.choice(num_examples, size=self.n_clusters, replace=False)
            centroids = X[random_row_ids, :]
            print(X.shape)
            for i, data_example in enumerate(X):
                cluster_distances[:] = ((centroids - data_example) ** 2).sum(axis=1) ** 0.5
                cluster_assignments[i] = cluster_distances.argmin()
                print(i, cluster_distances, )
            print(cluster_assignments)
            for i in range(self.n_clusters):
                # todo: А может, надо промежуточно хранить кластеры
                # todo: и переприсваивать только в случае если кластеры стали лучше?
                centroids[i] = X[cluster_assignments == i].mean(axis=0)

            # todo: инициализировать центроиды случайными точками из данных
            # todo: Кластеризовать (присвоить точки на основе заданных центроидов)
            # todo: Внутри каждого из кластеров посчитать среднее
            # todo: Заапдейтить центроид на основе среднего
            pass


a = [[2, 4, 5], [126, 7, 6], [17, 7, 1], [2, 2, 1]]
a = np.array(a)

model = KmeansModel(n_clusters=2, n_init=1)
model.fit(a)
