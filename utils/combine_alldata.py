import json
import os
from utils.data_utils import save_processed_files
from random import sample
import pandas as pd
from utils.sampleData import sampleData


def read_data(dataset, data_type):
    path1 = f"./data/{dataset}/single_state/{data_type}.json"
    # path2 = f"./data/{dataset}/all_state/{data_type}.json"

    return pd.read_json(path1)


if __name__ == "__main__":
    datasets = ["crosswoz", "risawoz", "sgd", "multiwoz2.0", "multiwoz2.1", "multiwoz2.2"]
    if not os.path.exists("./data/all/single_state"):
        os.makedirs("./data/all/single_state")

    if not os.path.exists("./data/all/all_state"):
        os.makedirs("./data/all/all_state")

    for data_type in ["train"]:
        df = pd.DataFrame()
        slotname_distribution = pd.DataFrame()
        for dataset in datasets:
            d = read_data(dataset, data_type)
            df = pd.concat([df, d])

            tmp = d.groupby("slotname").apply(lambda x: pd.Series({f"{dataset}": len(x)})).describe().transpose()

            slotname_distribution = pd.concat([slotname_distribution, tmp], axis=0)

        df = df.reset_index()
        df.to_json(f"./data/all/single_state/{data_type}.json", force_ascii=False, orient="records", indent=2)

        tmp = df.groupby("slotname").apply(lambda x: pd.Series({"all": len(x)})).describe().transpose()

        slotname_distribution = pd.concat([slotname_distribution, tmp], axis=0).reset_index()

        if not os.path.exists("./result/info"):
            os.makedirs("./result/info")
        slotname_distribution.to_csv("./result/info/slotname_distribution.csv")
