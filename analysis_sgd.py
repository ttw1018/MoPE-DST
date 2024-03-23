import pandas as pd
from glob import glob
import re
import os


def add_info(path, info):
    return re.sub("(.*)(\.json)", fr"\1-{info}\2", path)

def cal_acc(df):
    def compare(data):
        if data["correct"].sum() == data["correct"].count():
            return 1
        else:
            return 0
    slot_acc = df["correct"].sum() / df["correct"].count()
    turn_data = df.groupby(["dialogue_id", "turn"]).apply(compare)
    turn_acc = turn_data.sum() / turn_data.count()
    dialogue_data = df.groupby(["dialogue_id"]).apply(compare)
    dialogue_acc = dialogue_data.sum() / dialogue_data.count()

    return pd.Series({
        "slot_acc": slot_acc,
        "turn_acc": turn_acc,
        "dialogue_acc": dialogue_acc
    })

def get_zero_shot_result(path, unseen_domain):
    if not os.path.exists(path):
        print(f"{path} is not exist")
        return

    df = pd.read_json(path)

    df["domain"] = df.apply(lambda x: "unseen" if x["domain"] in unseen_domain else "seen", axis=1)

    res = df.groupby("domain").apply(cal_acc)

    print(res)


if __name__ == "__main__":
    unseen_domain = ["alarm", "messaging", "payment", "train"]

    path = glob(f"./prediction/sgd/prediction/1/*.json")
    for p in path:
        print(p)
        get_zero_shot_result(p, unseen_domain)
