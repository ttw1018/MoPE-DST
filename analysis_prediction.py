import os.path
import json
import pandas as pd
from functools import partial


def read_result(dataset, ratio):
    result_path = f"./result/{ratio}/{dataset}.csv"

    df = pd.read_csv(result_path)

    df = df[~df["checkpoint"].isna()]
    df = df[df["cluster"] == "all"]

    return df


def get_error_distrubution(data):
    def cal_domain_acc(df):
        acc = df["correct"].sum() / df["correct"].count()

        def cal_slotname_acc(df):
            acc = df["correct"].sum() / df["correct"].count()
            return pd.Series({"slotname_acc": acc})

        df = df.groupby("slotname").apply(cal_slotname_acc)

        df["domain_acc"] = acc

        return df

    domain_slotname_acc = data.groupby("domain").apply(cal_domain_acc).reset_index()

    return domain_slotname_acc


def get_error_info(dataset, ratio, run_type="few-shot"):
    if run_type == "zero-shot":
        df = read_result(dataset, "zero-shot")
    else:
        df = read_result(dataset, ratio)

    schema_path = f"./data/{dataset}/schema.json"
    assert os.path.exists(schema_path), f"{schema_path} is not found"

    with open(schema_path, "r") as fin_schema:
        schema = json.load(fin_schema)

    slotname2domain = {
        n: k for k, v in schema.items() for n in v
    }

    for info in df.iterrows():
        turn_acc = info[1]["turn acc"]
        n_clusters = info[1]["n_clusters"]

        prediction_path = f"prediction/{dataset}/prediction/4-0-{turn_acc:.2f}.json"
        assert os.path.exists(prediction_path), f"{prediction_path} is not found"
        datas = pd.read_json(prediction_path)

        slotname2cluster = pd.read_csv(f"./save/{dataset}/{ratio}/4/{n_clusters}-cluster/{n_clusters}-cluster-test.csv")
        slotname2cluster = {v["slotname"]: v["cluster"] for k, v in slotname2cluster.iterrows()}

        if set(slotname2cluster.keys()) != set(datas["slotname"]):
            continue

        def add_domain(slotname2domain, series):
            if "domain" not in series:
                series["domain"] = slotname2domain[series["slotname"]]
            return series

        def add_cluster(slotname2cluster, series):
            if "cluster" not in series:
                series["cluster"] = slotname2cluster[series["slotname"]]
            return series

        datas = datas.apply(partial(add_domain, slotname2domain), axis=1)
        error_distribution = get_error_distrubution(datas)

        error_distribution = error_distribution.apply(partial(add_cluster, slotname2cluster), axis=1)

        if not os.path.exists("./result/analysis"):
            os.makedirs("./result/analysis")

        error_distribution.to_csv(f"./result/analysis/{dataset}-{ratio}-{n_clusters}cluster-error-distribution.csv")


if __name__ == "__main__":
    # few_shot_datasets = ["crosswoz", "multiwoz2.2", "sgd"]
    # few_shot_datasets = ["crosswoz", "multiwoz2.2"]
    # few_shot_ratios = ["1", "5", "10", "100"]
    #
    # for dataset in few_shot_datasets:
    #     for ratio in few_shot_ratios:
    #         get_error_info(dataset, ratio)

    zero_shot_datasets = ["multiwoz2.1"]
    zero_shot_ratios = ["attraction", "hotel", "restaurant", "taxi", "train"]

    for dataset in zero_shot_datasets:
        for ratio in zero_shot_ratios:
            get_error_info(dataset, ratio, "zero-shot")
