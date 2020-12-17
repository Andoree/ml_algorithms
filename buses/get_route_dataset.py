import os
from argparse import ArgumentParser

import pandas as pd


def get_points_distance(point_1, point_2):
    dist = ((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2) ** 0.5
    return dist


def data_entry_to_distance(row, fixed_point):
    latitude = float(row["Latitude"])
    longitude = float(row["Longitude"])
    dist = get_points_distance((latitude, longitude), fixed_point)
    return dist


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_path', default=r"dataset/filtered_data.csv")
    parser.add_argument('--route_number', default='22')
    parser.add_argument('--fixed_point_latitude', type=float, default=55.796276)
    parser.add_argument('--fixed_point_longitude', type=float, default=49.123731)
    parser.add_argument('--garage_number', type=int, default=2775)
    parser.add_argument('--time_filter_from', default='12:00')
    parser.add_argument('--time_filter_to', default='22:00')
    parser.add_argument('--output_path', default=r"dataset_route_22_garag_2775.csv")
    args = parser.parse_args()
    # 55.79676800000001, 49.119107 - old
    # 55.796276, 49.123731
    input_path = args.input_path
    route_number = args.route_number
    point_latitude = args.fixed_point_latitude
    point_longitude = args.fixed_point_longitude
    fixed_point = (point_latitude, point_longitude)
    garage_number = args.garage_number
    time_filter_from = args.time_filter_from
    time_filter_to = args.time_filter_to
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    data_df = pd.read_csv(input_path)

    # Фильтрация по номеру маршрута
    filtered_route_df = data_df[data_df.Marsh == route_number]
    # Фильтрация по номеру автобуса
    filtered_route_df = filtered_route_df[filtered_route_df.GaragNumb == garage_number]
    # Парсим TimeNav как datetime
    filtered_route_df['TimeNav'] = pd.to_datetime(filtered_route_df['TimeNav'], format="%d.%m.%Y %H:%M:%S")
    # Сортируем данные по времени
    filtered_route_df.sort_values(by=["TimeNav"], inplace=True)
    filtered_route_df["TimeNav"] = filtered_route_df["TimeNav"].dt.floor('Min')
    # Считаем расстояние до заданной точки
    filtered_route_df["dist"] = filtered_route_df.apply(lambda row: data_entry_to_distance(row, fixed_point), axis=1)
    # Считаем изменение расстояния с момента прошлого измерения
    filtered_route_df['dist_delta'] = (filtered_route_df['dist'].shift(1) - filtered_route_df['dist']).abs()
    # Делаем время сбора данных индексом Dataframe'а
    filtered_route_df.set_index("TimeNav", inplace=True, drop=False)
    # Фильтруем измерения по времени
    filtered_route_df = filtered_route_df.between_time(time_filter_from, time_filter_to, )

    dist_deltas_df = filtered_route_df['dist_delta']
    # Удаляем выбросы - измерения, в которых произошло слишком резкое изменение расстояния
    non_outliers_mask = dist_deltas_df.between(dist_deltas_df.quantile(0.0), dist_deltas_df.quantile(0.98))
    filtered_route_df = filtered_route_df[non_outliers_mask]
    filtered_route_df.to_csv(output_path)


if __name__ == '__main__':
    main()
