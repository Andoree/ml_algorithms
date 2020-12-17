import os
from argparse import ArgumentParser
from time import sleep

import pandas as pd
import requests


def main():
    parser = ArgumentParser()
    parser.add_argument('--json_url', default=r"http://data.kzn.ru:8082/api/v0/dynamic_datasets/bus.json")
    parser.add_argument('--output_path', default=f"data/bus_data_9.csv")
    args = parser.parse_args()
    json_url = args.json_url
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    num_requests = 100000
    save_every = 250
    prefix = 'run_9'
    data_df = None
    for i in range(num_requests):
        print(f"Request: {i+1} / {num_requests}")
        sleep(5)
        response = requests.get(json_url)
        data = response.json()
        entries_list = []
        for entry in data:
            entry_data_dict = entry["data"]
            entry_data_dict["time"] = entry["updated_at"]
            entries_list.append(entry_data_dict)
        delta_df = pd.DataFrame(entries_list)
        if data_df is None:
            data_df = delta_df
        else:
            data_df = data_df.append(delta_df)
            data_df.drop_duplicates(inplace=True)
        if i % save_every == 0:
            output_path_attrs = output_path.split('.')
            output_path_attrs[0] += f"_{prefix}_{i}"
            checkpoint_path = '.'.join(output_path_attrs)
            data_df.to_csv(checkpoint_path, index=False)
            data_df = None
    data_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()
