from argparse import ArgumentParser

import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_path', default=r"full_data.csv")
    parser.add_argument('--output_path', default="filtered_data.csv")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    data_df = pd.read_csv(input_path)
    # Фильтруем данные: отбрасываем ненужные столбцы
    filtered_df = data_df[["Azimuth", "GaragNumb", "Graph", "Latitude", "Longitude", "Marsh", "Speed", "TimeNav"]]
    filtered_df.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()
