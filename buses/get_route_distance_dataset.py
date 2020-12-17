import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd


def get_points_distance(point_1, point_2):
    """
    Евклидово расстояние между 2 точками.
    :param point_1: tuple (Широта, долгота)
    :param point_2: tuple (Широта, долгота)
    :return:
    """
    dist = ((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2) ** 0.5
    return dist


def data_entry_to_distance(row, fixed_point):
    """
    :param row: Строка Dataframe'а
    :param fixed_point: tuple (широта, долгота) фиксированной точки, до которой
    подсчитывается расстояние
    :return:
    """
    latitude = float(row["Latitude"])
    longitude = float(row["Longitude"])
    dist = get_points_distance((latitude, longitude), fixed_point)
    return dist


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_path', default=r"filtered_data.csv")
    parser.add_argument('--route_number', default='22')
    parser.add_argument('--fixed_point_latitude', type=float, default=55.796276)
    parser.add_argument('--fixed_point_longitude', type=float, default=49.123731)
    parser.add_argument('--output_path', default=r"dataset_dist_22_liberty_square.csv")
    args = parser.parse_args()

    input_path = args.input_path
    route_number = args.route_number
    point_latitude = args.fixed_point_latitude
    point_longitude = args.fixed_point_longitude
    fixed_point = (point_latitude, point_longitude)
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    data_df = pd.read_csv(input_path)
    # Фильтрация по номеру маршрута
    filtered_route_df = data_df[data_df.Marsh == route_number]

    # Парсим TimeNav как datetime
    filtered_route_df['TimeNav'] = pd.to_datetime(filtered_route_df['TimeNav'], format="%d.%m.%Y %H:%M:%S")
    # Сортируем данные по времени
    filtered_route_df.sort_values(by=["TimeNav"], inplace=True)
    # Повышаем временную погрешность измерений - отбрасываем секунды,
    # оставляем измерения только за каждую минуту
    filtered_route_df["TimeNav"] = filtered_route_df["TimeNav"].dt.floor('Min')
    # Считаем расстояние до заданной точки для всех собранных измерений
    filtered_route_df["dist"] = filtered_route_df.apply(lambda row: data_entry_to_distance(row, fixed_point), axis=1)
    # Находим измерение с наименьшим расстоянием до заданной точки для каждой временной
    # единицы. То есть на каждый момент времени находим наиближайший автобус
    aggregated_df = filtered_route_df[["TimeNav", "dist"]].groupby("TimeNav").dist.min()
    """
    На выходе получаем измерения расстояния от остановки "Площадь Свободы" до
    наиближайшего автобуса маршрута №22 за каждую минуту в течение нескольких дней.
    """
    aggregated_df.to_csv(output_path)


if __name__ == '__main__':
    main()
