import os
from argparse import ArgumentParser

import pandas as pd

def main():
    parser = ArgumentParser()
    parser.add_argument('--input_data_dir', default=r"data/")
    parser.add_argument('--output_path', default="full_data.csv")
    args = parser.parse_args()
    input_data_dir = args.input_data_dir
    output_path = args.output_path

    data_df = None
    for fname in os.listdir(input_data_dir):
        input_path = os.path.join(input_data_dir, fname)
        data_delta_df = pd.read_csv(input_path)
        if data_df is None:
            data_df = data_delta_df
        else:
            data_df = data_df.append(data_delta_df)
        data_df.drop_duplicates()
    data_df.to_csv(output_path, index=False)



if __name__ == '__main__':
    main()