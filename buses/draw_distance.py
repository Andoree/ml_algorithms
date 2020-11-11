import datetime

import pandas as pd
from matplotlib import pyplot as plt


def get_points_distance(point_1, point_2):
    dist = ((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2) ** 0.5
    return dist


def data_entry_to_distance(row, fixed_point):
    latitude = float(row["Latitude"])
    longitude = float(row["Longitude"])
    dist = get_points_distance((latitude, longitude), fixed_point)
    return dist


def main():
    input_path = r"filtered_data.csv"
    datetime_format = "%d.%m.%Y %H:%M:%S"
    route_number = '22'
    fixed_point = (55.79676800000001, 49.119107)
    data_df = pd.read_csv(input_path)
    filtered_route_df = data_df[data_df.Marsh == route_number]
    filtered_route_df = filtered_route_df[filtered_route_df.GaragNumb == 2426]
    filtered_route_df['TimeNav'] = pd.to_datetime(filtered_route_df['TimeNav'], format="%d.%m.%Y %H:%M:%S")
    filtered_route_df.sort_values(by = ["TimeNav"],  inplace=True)
    filtered_route_df["dist"] = filtered_route_df.apply(lambda row : data_entry_to_distance(row, fixed_point), axis=1)
    filtered_route_df = filtered_route_df[filtered_route_df.TimeNav < datetime.date(2020,11,10)]
    times_list = filtered_route_df.TimeNav.values
    y_array = filtered_route_df.dist.values
    plt.plot(times_list, y_array)
    plt.show()

    print(filtered_route_df.dtypes)
    #  55°47'34"N   49°7'28"E


if __name__ == '__main__':
    main()
