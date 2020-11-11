import pandas as pd


def main():
    input_path = r"full_data.csv"
    output_path = "filtered_data.csv"

    data_df = pd.read_csv(input_path)
    filtered_df = data_df[["Azimuth", "GaragNumb", "Graph", "Latitude", "Longitude", "Marsh", "Speed", "TimeNav"]]
    filtered_df.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()
