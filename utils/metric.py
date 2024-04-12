import os
import pdb
import re
from copy import deepcopy

import pandas as pd

from utils.logger import result_logger
from utils.save_result import *


class ValueMetric:
    def __init__(self, args, tokenizer):
        self.args = args
        self.labels = []
        self.tokenizer = tokenizer
        self.predict_error = []
        self.turn = []
        self.dialogue_id = []
        self.input_context = []
        self.dialogue_history = []
        self.correct = []
        self.slotname = []
        self.datas = []
        self.total = 0
        self.right = 0
        self.step = 1
        self.slot_acc = -1
        self.JGA = -1
        self.label_acc = -1
        self.none_acc = -1

    def add(self, data):
        self.datas.extend(data)

    def compute(self, uuid, step=None, cluster_id=None):

        data = pd.DataFrame(self.datas)

        # if not self.args.use_all_state:
        #     data = data.groupby("dialogue_id").apply(self.update_dialogue_state).reset_index(drop=True)

        data = data.apply(self.check, axis=1)

        slot_acc, label_acc, none_acc = self.compute_slot_acc(data)
        JGA = self.compute_joint_goal_acc(data)

        slot_acc = float(f"{slot_acc:.2f}")
        label_acc = float(f"{label_acc:.2f}")
        none_acc = float(f"{none_acc:.2f}")
        JGA = float(f"{JGA:.2f}")

        self.slot_acc = slot_acc
        self.label_acc = label_acc
        self.none_acc = none_acc
        self.JGA = JGA

        result = f"slot acc: {slot_acc:.2f}  JGA acc: {JGA:.2f} label acc: {label_acc:.2f} none acc: {none_acc:.2f}"
        result_logger.info(
            f"{self.args.checkpoint} {self.args.dataset} {self.args.n_clusters} ratio {len(data)}"
            f"({self.args.data_ratio}) sample {self.args.num_sample} none_rate {self.args.none_rate} {result}")

        result_csv_dir = self.args.result_csv_dir
        if not os.path.exists(result_csv_dir):
            os.makedirs(result_csv_dir)
        result_csv_path = os.path.join(result_csv_dir, f"{self.args.dataset}.csv")

        if self.args.zero_shot:
            tmp_result = pd.DataFrame({"time": [time.strftime("%y-%m-%d %H:%M:%S", time.localtime())],
                                       "dataset": [self.args.dataset],
                                       "feature": [self.args.cluster_feature],
                                       "n_clusters": [self.args.n_clusters],
                                       "num sample": [self.args.num_sample],
                                       "history turn": [self.args.history_turn],
                                       "domain": [self.args.exclude_domain],
                                       "slot acc": [f"{slot_acc:.2f}"],
                                       "turn acc": [f"{JGA:.2f}"],
                                       "checkpoint": self.args.checkpoint})
            if step:
                insert_zero_shot_dev_result(uuid, step, cluster_id, slot_acc, label_acc, none_acc, JGA)
            else:
                insert_zero_shot_result(uuid, self.args, slot_acc, label_acc, none_acc, JGA)

            if not step and self.args.dataset == "sgd":
                self.args.exclude_domain = "all"
                # unseen_domain = {"Alarm_1", "Trains_1", "Payment_1", "Messaging_1"}
                for k, df in data.groupby("domain"):
                    new_args = deepcopy(self.args)
                    new_args.exclude_domain = k
                    slot_acc, label_acc, none_acc = self.compute_slot_acc(df)
                    JGA = self.compute_joint_goal_acc(df)
                    insert_zero_shot_result(uuid, new_args, slot_acc, label_acc, none_acc, JGA)
        else:
            tmp_result = pd.DataFrame({"time": [time.strftime("%y-%m-%d %H:%M:%S", time.localtime())],
                                       "dataset": [self.args.dataset],
                                       "feature": [self.args.cluster_feature],
                                       "n_clusters": [self.args.n_clusters],
                                       "num sample": [self.args.num_sample],
                                       "history turn": [self.args.history_turn],
                                       "data ratio": [self.args.data_ratio],
                                       "slot acc": [f"{slot_acc:.2f}"],
                                       "turn acc": [f"{JGA:.2f}"],
                                       "checkpoint": self.args.checkpoint})
            if step:
                insert_few_shot_dev_result(uuid, step, cluster_id, slot_acc, label_acc, none_acc, JGA)
            else:
                insert_few_shot_result(uuid, self.args, slot_acc, label_acc, none_acc, JGA)

        if os.path.exists(result_csv_path) and step == None:
            previous_result = pd.read_csv(result_csv_path, index_col=0)
            tmp_result = pd.concat([previous_result, tmp_result]).reset_index(drop=True)
            tmp_result.to_csv(result_csv_path, index_label="index")

        if self.args.save_prediction:
            prediction_dir = os.path.join(self.args.prediction_save_path,
                                          f"{self.args.dataset}/prediction/{self.args.exclude_domain if self.args.zero_shot else self.args.data_ratio}")
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir)

            data.to_json(os.path.join(prediction_dir, f"{uuid}-{JGA:.2f}.json"), indent=2, orient="records",
                         force_ascii=False)

            error_dir = os.path.join(self.args.prediction_save_path,
                                     f"{self.args.dataset}/error/{self.args.exclude_domain if self.args.zero_shot else self.args.data_ratio}")
            if not os.path.exists(error_dir):
                os.makedirs(error_dir)

            errors = data[data["correct"] == 0]
            errors.to_json(os.path.join(error_dir, f"{uuid}-{JGA:.2f}.json"), indent=2, orient="records",
                           force_ascii=False)

    @staticmethod
    def update_dialogue_state(df: pd.DataFrame):
        current_state = {}
        data = []
        for turn, turn_df in df.groupby("turn"):
            for _, info in turn_df.iterrows():
                slotname = info["slotname"]
                predict_state = info["predict_state"]
                if slotname not in current_state:
                    current_state[slotname] = predict_state
                else:
                    if predict_state == "none" and current_state[slotname] != "none":
                        info["predict_state"] = current_state[slotname]
                    else:
                        if current_state[slotname] != info["predict_state"]:
                            current_state[slotname] = info["predict_state"]
                data.append(info)
        data = pd.concat(data, axis=1, ignore_index=True).T.reset_index(drop=True)
        return data

    @staticmethod
    def compare(data):
        if data["correct"].sum() == data["correct"].count():
            return 1
        else:
            return 0

    def cal_acc(self, df):
        if len(df) == 0:
            return 0
        return len(df[df["correct"] == 1]) / len(df) * 100

    def compute_slot_acc(self, data):
        if len(data) == 0:
            return 0
        return self.cal_acc(data), self.cal_acc(data[data["labels"] != "none"]), self.cal_acc(data[data["labels"] ==
                                                                                                   "none"])

    def compute_joint_goal_acc(self, data):
        df = data.groupby(["dialogue_id", "turn"]).apply(self.compare)
        if len(df) == 0:
            return 0
        result = df.sum() / df.count() * 100
        return result

    @staticmethod
    def normalize(s: str):
        s = s.replace("，", ",")
        s = s.replace("。", ".")
        s = s.replace("）", ")")
        s = s.replace("（", "(")
        s = s.replace("；", ";")
        s = re.sub("[ -]", "", s)
        return s

    @staticmethod
    def check(data):
        predict = data["predict_state"].lower()
        label = data["labels"].lower()
        predict = ValueMetric.normalize(predict)
        label = ValueMetric.normalize(label)

        predict = predict.rstrip(",;.，。；、")

        if "，" in label:
            label = set(label.split("，"))
        elif ";" in label:
            label = set(label.split(";"))
        elif "," in label:
            label = set(label.split(","))

        if isinstance(label, set):
            if "，" in predict:
                predict = set(predict.split("，"))
            elif ";" in predict:
                predict = set(predict.split(";"))
            elif "," in predict:
                predict = set(predict.split(","))
            else:
                predict = {predict}

        if predict == label:
            data["correct"] = 1
        else:
            data["correct"] = 0

        return data
