import os
import pandas as pd

def main():
    data_dir = r"data/"
    output_path = "full_data.csv"
    data_df = None
    for fname in os.listdir(data_dir):
        input_path = os.path.join(data_dir, fname)
        data_delta_df = pd.read_csv(input_path)
        if data_df is None:
            data_df = data_delta_df
        else:
            data_df = data_df.append(data_delta_df)
        data_df.drop_duplicates()
    data_df.to_csv(output_path, index=False)



if __name__ == '__main__':
    main()